"""Routines specific to cleaning up EIA Form 923 data."""

import logging

import numpy as np
import pandas as pd
import pudl
import pudl.constants as pc
from pudl.transform.generic import TableTransformer
from pudl.workflow.task import Stage, emits, reads, transformer

logger = logging.getLogger(__name__)


# TODO(rousik): this creates eia923/tf task which will fail, but ideally
# we will not create these tasks for these intermediate class instantiations.

class Tf(TableTransformer):
    """Generic logic for eia923 table cleanup/transformations."""
    DATASET = 'eia923'

    @staticmethod
    def early_clean(df):
        df = _yearly_to_monthly_records(df, pc.month_dict_eia923)
        return df.pipe(pudl.helpers.fix_eia_na).pipe(pudl.helpers.convert_to_date)


###############################################################################
###############################################################################
# HELPER FUNCTIONS
###############################################################################
###############################################################################


def _yearly_to_monthly_records(df, md):
    """Converts an EIA 923 record of 12 months of data into 12 monthly records.

    Much of the data reported in EIA 923 is monthly, but all 12 months worth of
    data is reported in a single record, with one field for each of the 12
    months.  This function converts these annualized composite records into a
    set of 12 monthly records containing the same information, by parsing the
    field names for months, and adding a month field.  Non - time series data
    is retained in the same format.

    Args:
        df (pandas.DataFrame): A pandas DataFrame containing the annual
            data to be converted into monthly records.
        md (dict): a dictionary with the integers 1-12 as keys, and the
            patterns used to match field names for each of the months as
            values. These patterns are also used to rename the columns in
            the dataframe which is returned, so they need to match the entire
            portion of the column name that is month specific.

    Returns:
        pandas.DataFrame: A dataframe containing the same data as was passed in
        via df, but with monthly records instead of annual records.

    """
    yearly = df.copy()
    all_years = pd.DataFrame()

    for y in yearly.report_year.unique():
        this_year = yearly[yearly.report_year == y].copy()
        monthly = pd.DataFrame()
        for m in md:
            # Grab just the columns for the month we're working on.
            this_month = this_year.filter(regex=md[m]).copy()
            # Drop this month's data from the yearly data frame.
            this_year.drop(this_month.columns, axis=1, inplace=True)
            # Rename this month's columns to get rid of the month reference.
            this_month.columns = this_month.columns.str.replace(md[m], '')
            # Add a numerical month column corresponding to this month.
            this_month['report_month'] = m
            # Add this month's data to the monthly DataFrame we're building.
            monthly = pd.concat([monthly, this_month], sort=True)

        # Merge the monthly data we've built up with the remaining fields in
        # the data frame we started with -- all of which should be independent
        # of the month, and apply across all 12 of the monthly records created
        # from each of the # initial annual records.
        this_year = this_year.merge(monthly, left_index=True, right_index=True)
        # Add this new year's worth of data to the big dataframe we'll return
        all_years = pd.concat([all_years, this_year], sort=True)

    return all_years


# TODO(rousik): break the diamond dependency between fuel_receipts_costs and coalmines
# by constructing temporary table that reads fuel_receipts_costs, applies the cleanup
# and then have both Coalmine and FuelReceiptsCosts read from that.

# TODO(rousik): Coalmine should really be moved to entity extraction for cleaner
# logic here.


@transformer
class Coalmine(Tf):
    @reads(Tf.table_ref('fuel_receipts_costs', Stage.CLEAN))
    @emits(Stage.TIDY)
    def tidy(df):
        return df

    COLUMNS_TO_KEEP = [
        'mine_name',
        'mine_type_code',
        'state',
        'county_id_fips',
        'mine_id_msha']

    @staticmethod
    def early_clean(cmi_df):
        # _yearly_to_monthly_records fails when it is run more than once because
        # it needs report_year column. Because we are reading CLEAN table as our
        # input, we should not do any early_clean() here.
        # TODO(rousik): Maybe we can make _yearly_to_monthly_records a no-op
        # when relevant columns are not found to avoid this potential issue.
        return cmi_df

    @staticmethod
    def late_clean(cmi_df):
        # If we actually *have* an MSHA ID for a mine, then we have a totally
        # unique identifier for that mine, and we can safely drop duplicates and
        # keep just one copy of that mine, no matter how different all the other
        # fields associated with the mine info are... Here we split out all the
        # coalmine records that have an MSHA ID, remove them from the CMI
        # data frame, drop duplicates, and then bring the unique mine records
        # back into the overall CMI dataframe...
        cmi_with_msha = cmi_df[cmi_df['mine_id_msha'] > 0]
        cmi_with_msha = cmi_with_msha.drop_duplicates(
            subset=['mine_id_msha', ])
        cmi_df.drop(cmi_df[cmi_df['mine_id_msha'] > 0].index)
        cmi_df.append(cmi_with_msha)
        cmi_df.drop_duplicates()

        # drop null values if they occur in vital fields....
        cmi_df.dropna(subset=['mine_name', 'state'], inplace=True)
        return cmi_df

    @emits(Stage.TRANSFORMED)
    def transform(cmi_df):
        # we need an mine id to associate this coalmine table with the frc
        # table. In order to do that, we need to create a clean index, like
        # an autoincremeted id column in a db, which will later be used as a
        # primary key in the coalmine table and a forigen key in the frc table

        # first we reset the index to get a clean index
        cmi_df = cmi_df.reset_index()
        # then we get rid of the old index
        cmi_df = cmi_df.drop(labels=['index'], axis=1)
        # then name the index id
        cmi_df.index.name = 'mine_id_pudl'
        # then make the id index a column for simpler transferability
        return cmi_df.reset_index()


@transformer
class FuelReceiptsCosts(Tf):
    def clean(cmi_df):
        """Cleans up the coalmine_eia923 table.

        This function does most of the coalmine_eia923 table transformation. It is
        separate from the coalmine() transform function because of the peculiar
        way that we are normalizing the fuel_receipts_costs_eia923() table.

        All of the coalmine information is originally coming from the EIA
        fuel_receipts_costs spreadsheet, but it really belongs in its own table.
        We strip it out of FRC, and create that separate table, but then we need
        to refer to that table through a foreign key. To do so, we actually merge
        the entire contents of the coalmine table into FRC, including the surrogate
        key, and then drop the data fields.

        For this to work, we need to have exactly the same coalmine data fields in
        both the new coalmine table, and the FRC table. To ensure that's true, we
        isolate the transformations here in this function, and apply them to the
        coalmine columns in both the FRC table and the coalmine table.

        Args:
            cmi_df (pandas.DataFrame): A DataFrame to be cleaned, containing
                coalmine information (e.g. name, county, state)

        Returns:
            pandas.DataFrame: A cleaned DataFrame containing coalmine information.
        """
        # Because we need to pull the mine_id_msha field into the FRC table,
        # but we don't know what that ID is going to be until we've populated
        # this table... we're going to functionally end up using the data in
        # the coalmine info table as a "key."  Whatever set of things we
        # drop duplicates on will be the defacto key.  Whatever massaging we do
        # of the values here (case, removing whitespace, punctuation, etc.) will
        # affect the total number of "unique" mines that we end up having in the
        # table... and we probably want to minimize it (without creating
        # collisions).  We will need to do exactly the same transofrmations in the
        # FRC ingest function before merging these values in, or they won't match
        # up.
        cmi_df = (
            cmi_df.assign(
                # Map mine type codes, which have changed over the years, to a few
                # canonical values:
                mine_type_code=lambda x: x.mine_type_code.replace(
                    {'[pP]': 'P', 'U/S': 'US', 'S/U': 'SU', 'Su': 'S'},
                    regex=True),
                # replace 2-letter country codes w/ ISO 3 letter as appropriate:
                state=lambda x: x.state.replace(pc.coalmine_country_eia923),
                # remove all internal non-alphanumeric characters:
                mine_name=lambda x: x.mine_name.replace(
                    '[^a-zA-Z0-9 -]', '', regex=True),
                # Homogenize the data type that we're finding inside the
                # county_id_fips field (ugh, Excel sheets!).  Mostly these are
                # integers or NA values, but for imported coal, there are both
                # 'IMP' and 'IM' string values.
                county_id_fips=lambda x: x.county_id_fips.replace(
                    '[a-zA-Z]+', value=np.nan, regex=True
                )
            )
            # No leading or trailing whitespace:
            .pipe(pudl.helpers.strip_lower, columns=["mine_name"])
            .astype({"county_id_fips": float})
            .astype({"county_id_fips": pd.Int64Dtype()})
            .fillna({"mine_type_code": pd.NA})
            .astype({"mine_type_code": pd.StringDtype()})
        )
        return cmi_df

    @reads(Tf.table_ref('fuel_receipts_costs', Stage.CLEAN),
           Coalmine.get_stage(Stage.TRANSFORMED))
    def transformed(frc_df, cmi_df):
        """Transforms the fuel_receipts_costs dataframe.

        Fuel cost is reported in cents per mmbtu. Converts cents to dollars.
        """
        # Drop fields we're not inserting into the fuel_receipts_costs_eia923
        # table.
        cols_to_drop = ['plant_name_eia',
                        'plant_state',
                        'operator_name',
                        'operator_id',
                        'mine_id_msha',
                        'mine_type_code',
                        'state',
                        'county_id_fips',
                        'mine_name',
                        'regulated',
                        'reporting_frequency']

        # This type/naming cleanup function is separated out so that we can be
        # sure it is applied exactly the same both when the coalmine_eia923 table
        # is populated, and here (since we need them to be identical for the
        # following merge)
        frc_df = (
            frc_df.
            merge(cmi_df, how='left',
                  on=['mine_name', 'state', 'mine_id_msha',
                      'mine_type_code', 'county_id_fips']).
            drop(cols_to_drop, axis=1).
            # Replace the EIA923 NA value ('.') with a real NA value.
            pipe(pudl.helpers.fix_eia_na).
            # These come in ALL CAPS from EIA...
            pipe(pudl.helpers.strip_lower, columns=['supplier_name']).
            pipe(pudl.helpers.fix_int_na, columns=['contract_expiration_date', ]).
            assign(
                # Standardize case on transportaion codes -- all upper case!
                primary_transportation_mode_code=lambda x: x.primary_transportation_mode_code.str.upper(),
                secondary_transportation_mode_code=lambda x: x.secondary_transportation_mode_code.str.upper(),
                fuel_cost_per_mmbtu=lambda x: x.fuel_cost_per_mmbtu / 100,
                fuel_group_code=lambda x: x.fuel_group_code.str.lower().str.replace(' ', '_'),
                fuel_type_code_pudl=lambda x: pudl.helpers.cleanstrings_series(
                    x.energy_source_code, pc.energy_source_eia_simple_map),
                fuel_group_code_simple=lambda x: pudl.helpers.cleanstrings_series(
                    x.fuel_group_code, pc.fuel_group_eia923_simple_map),
                contract_expiration_month=lambda x: x.contract_expiration_date.apply(
                    lambda y: y[:-2] if y != '' else y)).
            assign(
                # These assignments are separate b/c they exp_month is altered 2x
                contract_expiration_month=lambda x: x.contract_expiration_month.apply(
                    lambda y: y if y != '' and int(y) <= 12 else ''),
                contract_expiration_year=lambda x: x.contract_expiration_date.apply(
                    lambda y: '20' + y[-2:] if y != '' else y)).
            # Now that we will create our own real date field, so chuck this one.
            drop('contract_expiration_date', axis=1).
            pipe(pudl.helpers.convert_to_date,
                 date_col='contract_expiration_date',
                 year_col='contract_expiration_year',
                 month_col='contract_expiration_month').
            pipe(pudl.helpers.convert_to_date).
            pipe(pudl.helpers.cleanstrings,
                 ['natural_gas_transport_code',
                  'natural_gas_delivery_contract_type_code'],
                 [{'firm': ['F'], 'interruptible': ['I']},
                  {'firm': ['F'], 'interruptible': ['I']}],
                 unmapped='')
        )

        # Remove known to be invalid mercury content values. Almost all of these
        # occur in the 2012 data. Real values should be <0.25ppm.
        bad_hg_idx = frc_df.mercury_content_ppm >= 7.0
        frc_df.loc[bad_hg_idx, "mercury_content_ppm"] = np.nan
        return frc_df
###############################################################################
###############################################################################
# DATATABLE TRANSFORM FUNCTIONS
###############################################################################
###############################################################################


@transformer
class Plants(Tf):
    """Builds eia923/plants table."""
    # There are other fields being compiled in the plant_info_df from all of
    # the various EIA923 spreadsheet pages. Do we want to add them to the
    # database model too? E.g. capacity_mw, operator_name, etc.
    COLUMNS_TO_KEEP = [
        'plant_id_eia',
        'combined_heat_power',
        'plant_state',
        'eia_sector',
        'naics_code',
        'reporting_frequency',
        'census_region',
        'nerc_region',
        'capacity_mw',
        'report_year']

    # Apply transformation lambdas on the specified columns
    COLUMN_CLEANUP_OPERATIONS = [
        (lambda col: col.replace({'M': 'monthly', 'A': 'annual'}),
         ['reporting_frequency']),

        # Since this is a plain Yes/No variable -- just make it a real sa.Boolean.
        (pudl.helpers.convert_to_boolean,
         ['combined_heat_power']),

        # Get rid of excessive whitespace introduced to break long lines (ugh)
        (lambda col: col.str.replace(' ', ''),
         ['census_region']),
    ]

    COLUMN_DTYPES = {'plant_id_eia': int}

    def transformed(df):
        return df.drop_duplicates(subset='plant_id_eia')


@transformer
class GenerationFuel(Tf):
    """Builds eia923/generation_fuel table."""

    COLUMNS_TO_DROP = [
        'combined_heat_power',
        'plant_name_eia',
        'operator_name',
        'operator_id',
        'plant_state',
        'census_region',
        'nerc_region',
        'naics_code',
        'eia_sector',
        'sector_name',
        'fuel_unit',
        'total_fuel_consumption_quantity',
        'electric_fuel_consumption_quantity',
        'total_fuel_consumption_mmbtu',
        'elec_fuel_consumption_mmbtu',
        'net_generation_megawatthours']

    def table_specific_clean(df):
        # Remove "State fuel-level increment" records... which don't pertain to
        # any particular plant (they have plant_id_eia == operator_id == 99999)
        df = df[df.plant_id_eia != 99999]
        return df

    def transformed(gf_df):
        gf_df['fuel_type_code_pudl'] = pudl.helpers.cleanstrings_series(gf_df.fuel_type,
                                                                        pc.fuel_type_eia923_gen_fuel_simple_map)


@transformer
class BoilerFuel(Tf):

    COLUMNS_TO_DROP = [
        'combined_heat_power',
        'plant_name_eia',
        'operator_name',
        'operator_id',
        'plant_state',
        'census_region',
        'nerc_region',
        'naics_code',
        'eia_sector',
        'sector_name',
        'fuel_unit',
        'total_fuel_consumption_quantity']

    @staticmethod
    def late_clean(bf_df):
        # Drop fields we're not inserting into the boiler_fuel_eia923 table.
        bf_df.dropna(subset=['boiler_id', 'plant_id_eia'], inplace=True)

        bf_df['fuel_type_code_pudl'] = pudl.helpers.cleanstrings_series(
            bf_df.fuel_type_code,
            pc.fuel_type_eia923_boiler_fuel_simple_map)

        # Convert Year/Month columns into a single Date column...
        return pudl.helpers.convert_to_date(bf_df)


@transformer
class Generation(Tf):

    @reads(Tf.table_ref('generator'))
    @emits(Stage.TIDY)
    def tidy(df):
        return df
    # TODO(rousik): these tidy-shims when tables are renamed are silly, maybe
    # we can solve this better, e.g. by renaming the page names in the excel
    # spreadsheet metadata.

    # Or, alternatively, we could just have class level
    #
    # RAW_SOURCE = Tf.table_ref('generator')
    #
    # which tells us which RAW source we should be reading.

    COLUMNS_TO_DROP = [
        'combined_heat_power',
        'plant_name_eia',
        'operator_name',
        'operator_id',
        'plant_state',
        'census_region',
        'nerc_region',
        'naics_code',
        'eia_sector',
        'sector_name',
        'net_generation_mwh_year_to_date']

    def late_clean(df):
        df = df.dropna(subset=['generator_id']).pipe(
            pudl.helpers.convert_to_date)
        # There are a few hundred (out of a few hundred thousand) records which
        # have duplicate records for a given generator/date combo. However, in all
        # cases one of them has no data (net_generation_mwh) associated with it,
        # so it's pretty clear which one to drop.
        unique_subset = ["report_date", "plant_id_eia", "generator_id"]
        dupes = df[df.duplicated(subset=unique_subset, keep=False)]
        return df.drop(dupes.net_generation_mwh.isna().index)
