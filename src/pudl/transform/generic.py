"""Module implementing generic table transformations. """

import logging

import pudl
import pudl.workflow.task as task
from pudl.workflow.task import Stage, emits

logger = logging.getLogger(__name__)


class TableTransformer(task.PudlTableTransformer):
    """This class implements common functionality for table transformation process.

    COLUMN_DTYPES map should contain column name to dtype mapping. This will be used
    to create DTYPE_ASSIGNED stage that is fed to custom transformations.

    COLUMN_CLEANUP_OPERATIONS is list of (operator, columndef) tuples and is suitable
    for simple one-liner transformations that should be applied to individual columns
    or list of columns.
    - operator is a callable that takes single pd.DataFrame argument. It will contain
      column values to transform.
    - columndef is either column name or list of column names on which the operator
      should be applied.

    COLUMNS_TO_KEEP can contain list of columns that should be retained. It will only
    be used if it is non-empty. This is going to be done during cleanup phase.

    COLUMNS_TO_DROP can contain list of columns that should be dropped. It will only
    be used if it is non-empty. This is going to be done during cleanup phase.

    Cleanup stage is composed of the following steps:
    1. apply column restrictions using COLUMNS_TO_KEEP and COLUMNS_TO_DROP
    2. call early_clean(df) method that subclasses can modify
    3. apply operations from COLUMN_CLEANUP_OPERATIONS list
    4. call late_clean(df) method that subclasses can modify
    """

    COLUMNS_TO_KEEP = []
    """If non-empty, only listed columns will be retained."""

    COLUMNS_TO_DROP = []
    """If non-empty, listed columns will be dropped."""

    COLUMN_DTYPES = {}
    """Dtypes that should be assigned to columns before transform stage."""

    COLUMN_CLEANUP_OPERATIONS = []
    """Contains tuples of (operator, columndef).

    columndef is either single column name or list of columns.

    operator is a method that takes DataFrame (with a single column) and transforms it.
    These operators will be applied in a loop as follows:

    for col_name in columndef:
      df[col_name] = operator(df[col_name])
    """

    @classmethod
    @emits(Stage.CLEAN)
    def generic_table_cleanup(cls, df):
        try:
            if cls.COLUMNS_TO_KEEP:
                logger.info(
                    f'{cls.__name__} specifies COLUMNS_TO_KEEP = {cls.COLUMNS_TO_KEEP}')
                df = df[cls.COLUMNS_TO_KEEP]
            if cls.COLUMNS_TO_DROP:
                logger.info(
                    f'{cls.__name__} specifies COLUMNS_TO_DROP = {cls.COLUMNS_TO_DROP}')
                df.drop(cls.COLUMNS_TO_DROP, axis=1, inplace=True)
        except (AttributeError, KeyError) as err:
            logger.error(
                f'Column cleanup for {cls.__name__} has failed: {err}')
            logger.warning(
                f'Columns present in the input df: {", ".join(sorted(df.columns))}')
            raise err
        df = cls.early_clean(df)
        for operator, columndef in cls.COLUMN_CLEANUP_OPERATIONS:
            cols = columndef
            if type(columndef) == str:
                cols = [columndef]
            for c in cols:
                df[c] = operator(df[c])
        df = cls.late_clean(df)
        return df

    @classmethod
    @emits(Stage.ASSIGN_DTYPE)
    def assign_dtype(cls, df):
        return df.astype(cls.COLUMN_DTYPES)

    @staticmethod
    def early_clean(df):
        return df

    @staticmethod
    def late_clean(df):
        return df

    @classmethod
    @emits(Stage.FINAL)
    def assign_final_dtypes(cls, df):
        ds = cls.DATASET
        if ds[:3] == 'eia':
            ds = 'eia'
        # TODO(rousik): the above eiaNNN --> eia conversion is dirty, figure out
        # a better way to do this.
        return pudl.helpers.convert_cols_dtypes(df, ds, cls.get_table_name())
