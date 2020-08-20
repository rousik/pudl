import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pudl.constants as pc
from prefect import Task
from pudl.workflow.task import PudlTableReference, Stage

logger = logging.getLogger(__name__)


class EntityExtractionMapper(Task):
    """Extracts rows of entity-relevant columns from given dataframe."""

    def __init__(self, entity_extractor_class, **kwargs):
        self.extractor = entity_extractor_class()
        name = self.extractor.NAME
        super().__init__(name=f'{name}: extract entity rows', **kwargs)

    def run(self, df, name='undef'):
        return self.extractor.extract_entity_rows(df, name)


class EntityExtractionReduce(Task):
    """Combines entity-relevant rows into annual and entity dataframes."""

    def __init__(self, entity_extractor_class, df_persistence, **kwargs):
        self.extractor = entity_extractor_class()
        self.df_persistence = df_persistence
        name = self.extractor.NAME
        # TODO(rousik): remove hardcoded eia dataset
        self.annual_ref = PudlTableReference(
            f'{name}_annual_eia', dataset='eia', stage=Stage.EXTRACTED_ENTITY)
        self.entity_ref = PudlTableReference(
            f'{name}_entity_eia', dataset='eia', stage=Stage.EXTRACTED_ENTITY)
        super().__init__(name=f'{name}: build entity frames', **kwargs)

    def run(self, entity_row_dfs):
        if (self.df_persistence.exists(self.annual_ref)
                and self.df_persistence.exists(self.entity_ref)):
            return (self.df_persistence.get(self.annual_ref),
                    self.df_persistence.get(self.entity_ref))

        entity_df, annual_df = self.extractor.harvest_entity(
            entity_row_dfs)
        logger.info(
            f'{self.extractor.NAME}: extracted {annual_df.size} annual entities.')
        # TODO(rousik): skip creating dataframes if they are empty or column-less
        # TODO(rousik): write consistency data to disk as well
        consistency_ref = PudlTableReference(
            f'{self.extractor.NAME}_entity_consistency', dataset='eia',
            stage=Stage.DEBUG)
        self.df_persistence.set(consistency_ref, self.extractor.consistency)
        self.df_persistence.set(self.annual_ref, annual_df)
        self.df_persistence.set(self.entity_ref, entity_df)
        return (annual_df.copy(), entity_df.copy())


class EntityExtractor(object):
    """Extracts entities from given list of input tables.

    Given list of input tables, this will determine whether entity is present in
    a given table (entity is associated with list of specific columns).

    We expect that tables may contain noisy data so we will construct the most
    consistent version of the entity (across its various attributes) and/or
    yearly-entity (for entities, that vary across years).

    At this point, columns associated with the entity can be dropped from the input
    tables and replaces with key-columns pointing to the extracted entities.

    TODO(rousik): make this documentation be better worded.
    """

    NAME = None
    """Name of the entity that is being extracted here."""

    ID_COLUMNS = []
    """List of columns that hold identifiers for this entity."""

    STATIC_COLUMNS = []
    """Columns that contain attributes that is expected to not change across years."""

    ANNUAL_COLUMNS = []
    """List of columns containing attributes that are expected to change across years."""

    DTYPES = {}
    """Mapping of columns to dtypes. This is applied after entity extraction."""

    def __init__(self):
        self.entity_df = None
        self.annual_df = None
        self.consistency = pd.DataFrame(
            columns=['column', 'consistent_ratio', 'wrongos', 'total'])

    def all_entity_columns(self, exclude_report_date=False):
        """Returns list of columns associated with this entity (with the exception of id cols)."""
        cols = self.ID_COLUMNS + self.STATIC_COLUMNS + self.ANNUAL_COLUMNS
        if not exclude_report_date:
            cols.append('report_date')
        return cols

    @ classmethod
    def drop_entity_columns(cls, df):
        """Drop STATIC and ANNUAL entity columns from df."""
        candidates = list(cls.STATIC_COLUMNS) + list(cls.ANNUAL_COLUMNS)
        logger.info(f'Candidate columns for {cls.NAME} is {candidates}')
        return df.drop(columns=[col for col in candidates if col in df.columns])

    @ classmethod
    def contains_entity(cls, df):
        """Returns True if df contains entity (it has ID_COLUMNS)."""
        base = set(cls.ID_COLUMNS)
        base.add('report_date')
        return base.issubset(df.columns)

    def extract_entity_rows_for_map(self, input_df_map):
        """Extracts rows of entity-relevant information from dict of input dataframes.

        For each of the named input_df, this method will extract columns that are relevant to this
        entity, and adds input_df table name to the table column. Resulting dataframe is appended
        to self.entity_rows.
        """
        for table_name, input_df in input_df_map.items():
            if 'annual' in table_name:
                # Skip over annual tables that may have been added by other processes.
                # TODO(rousik): maybe we can avoid feeding those tables to the harvester
                logger.error(
                    f'{table_name} should not be passed to entity extraction.')
                continue
            if not self.contains_entity(input_df):
                continue
            self.entity_rows.append(
                self.self.extract_entity_rows(input_df, table_name))

    def extract_entity_rows(self, input_df, table_name):
        """Extracts entity records from a given input_df. Stores table_name in table column."""
        if not self.contains_entity(input_df):
            return None

        cols = [c for c in self.all_entity_columns() if c in input_df.columns]
        df = input_df[cols].dropna(subset=self.ID_COLUMNS)
        df['table'] = table_name
        return df

    @classmethod
    def get_combined_entity_rows(cls, entity_rows):
        """Returns single dataframe constructed from self.entity_rows."""
        entity_rows = [df for df in entity_rows if df is not None]
        compiled_df = pd.concat(entity_rows, axis=0,
                                ignore_index=True, sort=True)
        # strip month and date from the date so we can have annual records
        compiled_df['report_date'] = compiled_df['report_date'].map(
            lambda x: datetime(x.year, 1, 1))
        compiled_df = compiled_df.astype(cls.DTYPES)
        return compiled_df

    @classmethod
    def column_consistency(cls, entity_df, column, strictness=.7):
        """
        Find the occurence of plants & the consistency of records.

        We need to determine how consistent a reported value is in the records
        across all of the years or tables that the value is being reported, so we
        want to compile two key numbers: the number of occurances of the entity and
        the number of occurances of each reported record for each entity. With that
        information we can determine if the reported records are strict enough.

        Args:
            entity_df: pd.DataFrame containing extracted entities.
            column: name of the columns we need to calculate consistency for.
            strictness: minimal consistency required for the entity to be considered
              good enough.

        Returns:
            pandas.DataFrame: this dataframe will be a transformed version of
            compiled_df with NaNs removed and with new columns with information
            about the consistency of the reported values.
        """
        cols_to_consit = list(cls.ID_COLUMNS)
        if column in cls.ANNUAL_COLUMNS:
            cols_to_consit.append('report_date')

        # select only the colums you want and drop the NaNs
        # we want to drop the NaNs because
        # TODO(rousik): ideally, we would break the dependency on pc.column_dtypes
        # and embed that directly in the transformers?
        col_df = entity_df[cls.ID_COLUMNS +
                           ['report_date', column, 'table']].copy()
        if pc.column_dtypes["eia"][column] == pd.StringDtype():
            nan_str_mask = (col_df[column] == "nan").fillna(False)
            col_df.loc[nan_str_mask, column] = pd.NA
        col_df = col_df.dropna()

        if len(col_df) == 0:
            col_df[f'{column}_consistent'] = pd.NA
            col_df['entity_occurences'] = pd.NA
            col_df = col_df.drop(columns=['table'])
            return col_df
        # determine how many times each entity occurs in col_df
        occur = (
            col_df.
            groupby(by=cols_to_consit).
            agg({'table': "count"}).
            reset_index().
            rename(columns={'table': 'entity_occurences'})
        )

        # add the occurrences into the main dataframe
        col_df = col_df.merge(occur, on=cols_to_consit)

        # determine how many instances of each of the records in col exist
        consist_df = (
            col_df.
            groupby(by=cols_to_consit + [column]).
            agg({'table': 'count'}).
            reset_index().
            rename(columns={'table': 'record_occurences'})
        )
        # now in col_df we have # of times an entity occurred accross the tables
        # and we are going to merge in the # of times each value occured for each
        # entity record. When we merge the consistency in with the occurances, we
        # can determine if the records are more than 70% consistent across the
        # occurances of the entities.
        col_df = col_df.merge(consist_df, how='outer').drop(columns=['table'])
        # change all of the fully consistent records to True
        col_df[f'{column}_consistent'] = (
            col_df['record_occurences'] /
            col_df['entity_occurences'] > strictness)

        return col_df

    def harvest_entity(self, entity_rows_dfs):
        # TODO(rousik): we may want to explicitly pass arguments here.

        # We assume that extract_entity_rows has been already called.
        compiled_df = self.get_combined_entity_rows(entity_rows_dfs)

        # Debug print input tables for this entity
        input_tables = ', '.join(sorted(compiled_df.table.unique()))
        logger.info(
            f'Entity {self.NAME} extracted from the following tables: {input_tables}')

        # compile annual ids
        annual_id_df = compiled_df[
            ['report_date'] + self.ID_COLUMNS].copy().drop_duplicates()
        annual_id_df.sort_values(['report_date'] + self.ID_COLUMNS,
                                 inplace=True, ascending=False)

        # create the annual and entity dfs
        entity_id_df = annual_id_df.drop(
            ['report_date'], axis=1).drop_duplicates(subset=self.ID_COLUMNS)

        entity_df = entity_id_df.copy()
        annual_df = annual_id_df.copy()

        for col in self.STATIC_COLUMNS + self.ANNUAL_COLUMNS:
            col_df = self.column_consistency(compiled_df, col)

            # pull the correct values out of the df and merge w/ the plant ids
            # TODO(rousik): cols_to_consit leaks out of column_consistency, maybe
            # roll this block into the column_consistency
            if col in self.ANNUAL_COLUMNS:
                cols_to_consit = self.ID_COLUMNS + ['report_date']
            else:
                cols_to_consit = list(self.ID_COLUMNS)

            col_correct_df = (
                col_df[col_df[f'{col}_consistent']].
                drop_duplicates(
                    subset=(cols_to_consit + [f'{col}_consistent']))
            )
            # we need this to be an empty df w/ columns bc we are going to use it
            if col_correct_df.empty:
                col_correct_df = pd.DataFrame(columns=col_df.columns)

            if col in self.STATIC_COLUMNS:
                clean_df = entity_id_df.merge(
                    col_correct_df, on=self.ID_COLUMNS, how='left')
                clean_df = clean_df[self.ID_COLUMNS + [col]]
                entity_df = entity_df.merge(clean_df, on=self.ID_COLUMNS)

            if col in self.ANNUAL_COLUMNS:
                clean_df = annual_id_df.merge(
                    col_correct_df, on=(self.ID_COLUMNS + ['report_date']), how='left')
                clean_df = clean_df[self.ID_COLUMNS + ['report_date', col]]
                annual_df = annual_df.merge(
                    clean_df, on=(self.ID_COLUMNS + ['report_date']))

            # TODO(rousik): here's where special_case_cols handling was. Figure out how to put
            # it back.
            # get the still dirty records by using the cleaned ids w/null values
            # we need the plants that have no 'correct' value so
            # we can't just use the col_df records when the consistency is not True
            # dirty_df = col_df.merge(
            #    clean_df[clean_df[col].isnull()][self.ID_COLUMNS])
            # if col in special_case_cols.keys():
            #    clean_df = special_case_cols[col][0](
            #        dirty_df, clean_df, entity_id_df, entity_id, col,
            #        cols_to_consit, special_case_cols[col][1])

            # this next section is used to print and test whether the harvested
            # records are consistent enough
            total = len(col_df.drop_duplicates(subset=cols_to_consit))
            # if the total is 0, the ratio will error, so assign null values.
            if total == 0:
                ratio = np.NaN
                wrongos = np.NaN
                logger.debug(f"       Zero records found for {col}")
            if total > 0:
                ratio = (
                    len(col_df[(col_df[f'{col}_consistent'])].
                        drop_duplicates(subset=cols_to_consit)) / total
                )
                wrongos = (1 - ratio) * total
                logger.debug(
                    f"       Ratio: {ratio:.3}  "
                    f"Wrongos: {wrongos:.5}  "
                    f"Total: {total}   {col}"
                )
                if ratio < 0.9:
                    raise AssertionError(
                        f'Harvesting of {col} for entity {self.NAME} is too inconsistent at {ratio:.3}.')
            # add to a small df to be used in order to print out the ratio of
            # consistent records
            self.consistency = self.consistency.append({
                'column': col,
                'consistent_ratio': ratio,
                'wrongos': wrongos,
                'total': total}, ignore_index=True)

        # Calculate overall entity consistency
        mcs = self.consistency['consistent_ratio'].mean()
        logger.info(
            f"Average consistency of static {self.NAME} values is {mcs:.2%}")
        self.entity_df = entity_df
        self.annual_df = annual_df
        return (self.entity_df, self.annual_df)

    @classmethod
    def add_entity_extraction_tasks(cls, flow, df_persistence):
        """Adds the entity extraction tasks into the flow.

        It will pull tasks that emit Stage.FINAL tables and use
        these as inputs for the entity extraction.
        """
        mapper = EntityExtractionMapper(cls)
        reducer = EntityExtractionReduce(cls, df_persistence)
        with flow:
            final_tables = flow.get_tasks(tags=[Stage.FINAL.name])
            table_names = [ft.name for ft in final_tables]
            entity_rows = mapper.map(final_tables, name=table_names)
            reducer(entity_rows)
