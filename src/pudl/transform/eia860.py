"""Module to perform data cleaning functions on EIA860 data tables."""

import logging

import numpy as np
import pandas as pd

import pudl
import pudl.constants as pc
import pudl.workflow.task as task
from pudl.workflow.task import reads

# TODO(rousik): simplify the above import

logger = logging.getLogger(__name__)


# TODO: we should just add _eia860 to all of the transformed tables
# ownership
# plants
# boiler_generator_assn
# utilities

Tf = task.PudlTableTransformer.for_dataset('eia860')

# TODO(rousik): it is easy to mess up types in COLUMN_DTYPES
# and/or COLUMN_CLEANUP_OPERATIONS. We need to ensure that the contents
# are valid and should probably run the validation on all known objects.


class CommonCleanupMixin(object):
    """This mixin implements some convenience operations for transforming eia tables.

    This introduces standardized clean() method which:
    1. applies fix_eia_na and convert_to_date helpers
    2. cleans individual columns using COLUMN_CLEANUP_OPERATIONS
    3. then calls table_specific_clean() which can be overriden.

    COLUMN_CLEANUP_OPERATIONS should contain (operator, columndef) tuples where
    columndef is either single column name (str) or list of column names and operator
    is a lambda function that takes pd.DataFrame and returns pd.DataFrame.

    The expectation here is that operator will be given a single-column (based on
    the columndef), apply changes and the results will be assigned to the column.

    After the cleanup phase, dtypes are assigned to specific columns using the
    mapping in COLUMN_DTYPES {column name: type}.

    Ultimately, no-op transformed() method is defined that can be overriden by
    subclasses.
    """
    # TODO(rousik): this could be easily used as a common ancestor class for
    # variety of transformers. If we use @emits annotation instead of finding
    # methods by name we might avoid the use of Mixin, although we would still
    # need a way to skip over the non-instantiated class.

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
    def clean(cls, df):
        df = df.pipe(pudl.helpers.fix_eia_na).pipe(
            pudl.helpers.convert_to_date)
        try:
            for operator, columndef in cls.COLUMN_CLEANUP_OPERATIONS:
                cols = columndef
                if type(columndef) == str:
                    cols = [columndef]
                for c in cols:
                    df[c] = operator(df[c])
            df = cls.table_specific_clean(df)
        except ValueError as err:
            logging.error(f'Bad transformation in {cls.__name__}: {err}')
            raise err
        return df

    @classmethod
    def table_specific_clean(cls, df):
        return df

    @classmethod
    def assign_dtype(self, df):
        return df.astype(self.COLUMN_DTYPES)

    def transformed(df):
        return df


class Ownership(CommonCleanupMixin, Tf):

    COLUMN_DTYPES = {
        "owner_utility_id_eia": pd.Int64Dtype(),
        "utility_id_eia": pd.Int64Dtype(),
        "plant_id_eia": pd.Int64Dtype(),
        "owner_state": pd.StringDtype(),
    }

    def table_specific_clean(df):
        # The fix we're making here is only known to be valid for 2011 -- if we
        # get older data... then we need to to revisit the cleaning function and
        # make sure it also applies to those earlier years.
        if min(df.report_date.dt.year) < min(pc.working_years["eia860"]):
            raise ValueError(
                f"EIA 860 transform step is only known to work for "
                f"year {min(pc.working_years['eia860'])} and later, but found data "
                f"from year {min(df.report_date.dt.year)}."
            )

        # Prior to 2012, ownership was reported as a percentage, rather than
        # as a proportion, so we need to divide those values by 100.
        df.loc[df.report_date.dt.year < 2012, 'fraction_owned'] = \
            df.loc[df.report_date.dt.year < 2012, 'fraction_owned'] / 100
        return df


class Generators(CommonCleanupMixin, Tf):
    @reads(
        Tf.table_ref('generator_proposed'),
        Tf.table_ref('generator_existing'),
        Tf.table_ref('generator_retired'))
    def tidy(gp_df, ge_df, gr_df):
        """Creates generators table."""
        # Groupby objects were creating chained assignment warning that is N/A
        pd.options.mode.chained_assignment = None

        # There are three sets of generator data reported in the EIA860 table,
        # planned, existing, and retired generators. We're going to concatenate
        # them all together into a single big table, with a column that indicates
        # which one of these tables the data came from, since they all have almost
        # exactly the same structure
        gp_df['operational_status'] = 'proposed'
        ge_df['operational_status'] = 'existing'
        gr_df['operational_status'] = 'retired'
        dfs = pd.concat([ge_df, gp_df, gr_df], sort=True)
        return dfs.dropna(subset=['generator_id', 'plant_id_eia'])

    COLUMN_CLEANUP_OPERATIONS = [
        # A subset of the columns have zero values, where NA is appropriate:
        (lambda df_col: df_col.replace(to_replace=[" ", 0], value=np.nan), [
            'planned_retirement_month',
            'planned_retirement_year',
            'planned_uprate_month',
            'planned_uprate_year',
            'other_modifications_month',
            'other_modifications_year',
            'planned_derate_month',
            'planned_derate_year',
            'planned_repower_month',
            'planned_repower_year',
            'planned_net_summer_capacity_derate_mw',
            'planned_net_summer_capacity_uprate_mw',
            'planned_net_winter_capacity_derate_mw',
            'planned_net_winter_capacity_uprate_mw',
            'planned_new_capacity_mw',
            'nameplate_power_factor',
            'minimum_load_mw',
            'winter_capacity_mw',
            'summer_capacity_mw',
        ]),

        # A subset of the columns have "X" values, where other columns_to_fix
        # have "N" values. Replacing these values with "N" will make for uniform
        # values that can be converted to Boolean True and False pairs.
        (lambda df_col: df_col.replace(to_replace='X', value='N'),
         ['duct_burners', 'bypass_heat_recovery', 'syncronized_transmission_grid']),

        # A subset of the columns have "U" values, presumably for "Unknown," which
        # must be set to None in order to convert the columns to datatype Boolean.
        (lambda df_col: df_col.replace(to_replace='U', value=None),
         ['multiple_fuels', 'switch_oil_gas']),

        (pudl.helpers.convert_to_boolean,
         [
             'duct_burners',
             'multiple_fuels',
             'deliver_power_transgrid',
             'syncronized_transmission_grid',
             'solid_fuel_gasification',
             'pulverized_coal_tech',
             'fluidized_bed_tech',
             'subcritical_tech',
             'supercritical_tech',
             'ultrasupercritical_tech',
             'carbon_capture',
             'stoker_tech',
             'other_combustion_tech',
             'cofire_fuels',
             'switch_oil_gas',
             'bypass_heat_recovery',
             'associated_combined_heat_power',
             'planned_modifications',
             'other_planned_modifications',
             'uprate_derate_during_year',
             'previously_canceled',
         ]),
    ]

    COLUMN_DTYPES = {
        'plant_id_eia': int,
        'generator_id': str,
        'unit_id_eia': str,
        'utility_id_eia': int,
    }

    def transformed(df):
        return (df.pipe(pudl.helpers.month_year_to_date).
                assign(fuel_type_code_pudl=lambda x: pudl.helpers.cleanstrings_series(
                    x['energy_source_code_1'], pc.fuel_type_eia860_simple_map)).
                pipe(pudl.helpers.strip_lower,
                     columns=['rto_iso_lmp_node_id',
                              'rto_iso_location_wholesale_reporting_id']).
                pipe(pudl.helpers.convert_to_date))
        # TODO(rousik): do we need to call convert_to_date again here?


class Plant(CommonCleanupMixin, Tf):
    COLUMN_CLEANUP_OPERATIONS = [
        # A subset of the columns have "X" values, where other columns_to_fix
        # have "N" values. Replacing these values with "N" will make for uniform
        # values that can be converted to Boolean True and False pairs.
        (lambda c_df: c_df.replace(to_replace='X', value='N'),
         ['ash_impoundment', 'natural_gas_storage', 'liquefied_natural_gas_storage']),

        (pudl.helpers.convert_to_boolean, [
            "ferc_cogen_status",
            "ferc_small_power_producer",
            "ferc_exempt_wholesale_generator",
            "ash_impoundment",
            "ash_impoundment_lined",
            "energy_storage",
            "natural_gas_storage",
            "liquefied_natural_gas_storage",
            "net_metering",
        ]),
    ]

    COLUMN_DTYPES = {
        "plant_id_eia": int,
        "utility_id_eia": int,
        "primary_purpose_naics_id": "Int64",
        "ferc_cogen_docket_no": str,
        "ferc_exempt_wholesale_generator_docket_no": str,
        "ferc_small_power_producer_docket_no": str,
        "street_address": str,
        "zip_code": str,
    }

    def table_specific_clean(df):
        df = df.drop("iso_rto", axis="columns")

        # Spelling, punctuation, and capitalization of county names can vary from
        # year to year. We homogenize them here to facilitate correct value
        # harvesting.
        df['county'] = (
            df.county.
            str.replace(r'[^a-z,A-Z]+', ' ').
            str.strip().
            str.lower().
            str.replace(r'\s+', ' ').
            str.title()
        )
        return df

    def transformed(df):
        return df.pipe(pudl.helpers.convert_to_date)


class BoilerGeneratorAssn(CommonCleanupMixin, Tf):

    COLUMN_DTYPES = {
        # We need to cast the generator_id column as type str because sometimes
        # it is heterogeneous int/str which make drop_duplicates fail.
        'generator_id': str,
        'boiler_id': str,
        'utility_id_eia': int,
    }

    def tidy(df):
        """Extracts columns of interest from the boiler table."""
        return df[[
            'report_year',
            'utility_id_eia',
            'plant_id_eia',
            'boiler_id',
            'generator_id']]

    def clean(df):
        # There are some bad (non-data) lines in some of the boiler generator
        # data files (notes from EIA) which are messing up the import. Need to
        # identify and drop them early on.
        df['utility_id_eia'] = df['utility_id_eia'].astype(str)
        df = df[df.utility_id_eia.str.isnumeric()]

        df['plant_id_eia'] = df['plant_id_eia'].astype(int)
        return df

    def transformed(df):
        """Transforms boiler_generator_assn table."""
        # This drop_duplicates isn't removing all duplicates
        df = df.drop_duplicates()
        df = df.dropna()
        df = pudl.helpers.convert_to_date(df)
        return df


class Utilities(CommonCleanupMixin, Tf):
    @reads(Tf.table_ref('utility'))
    def tidy(df):
        """Pulls utility:RAW into utilities:TIDY."""
        return df

    COLUMN_CLEANUP_OPERATIONS = [
        (lambda col: col.str.upper().replace({
            'QB': 'QC',  # Wrong abbreviation for Quebec
            'Y': 'NY',  # Typo
        }), 'state'),
        (pudl.helpers.convert_to_boolean, [
            'plants_reported_owner',
            'plants_reported_operator',
            'plants_reported_asset_manager',
            'plants_reported_other_relationship'
        ]),
    ]

    COLUMN_DTYPES = {'utility_id_eia': int}

    def transformed(df):
        return pudl.helpers.convert_to_date(df)
