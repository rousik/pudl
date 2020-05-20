"""Routines used for extracting the raw FERC 714 data."""

import logging
import pathlib
import zipfile

import pandas as pd

import pudl.constants as pc

logger = logging.getLogger(__name__)

TABLE_FNAME = {
    "id_certification_ferc714": "Part 1 Schedule 1 - Identification Certification.csv",
    "gen_plants_ba_ferc714": "Part 2 Schedule 1 - Balancing Authority Generating Plants.csv",
    "demand_monthly_ba_ferc714": "Part 2 Schedule 2 - Balancing Authority Monthly Demand.csv",
    "net_energy_load_ba_ferc714": "Part 2 Schedule 3 - Balancing Authority Net Energy for Load.csv",
    "adjacency_ba_ferc714": "Part 2 Schedule 4 - Adjacent Balancing Authorities.csv",
    "interchange_ba_ferc714": "Part 2 Schedule 5 - Balancing Authority Interchange.csv",
    "lambda_hourly_ba_ferc714": "Part 2 Schedule 6 - Balancing Authority Hourly System Lambda.csv",
    "lambda_description_ferc714": "Part 2 Schedule 6 - System Lambda Description.csv",
    "description_pa_ferc714": "Part 3 Schedule 1 - Planning Area Description.csv",
    "demand_forecast_pa_ferc714": "Part 3 Schedule 2 - Planning Area Forecast Demand.csv",
    "demand_hourly_pa_ferc714": "Part 3 Schedule 2 - Planning Area Hourly Demand.csv",
    "respondent_id_ferc714": "Respondent IDs.csv",
}
"""Dictionary mapping PUDL tables to filenames within the FERC 714 zipfile."""

TABLE_ENCODING = {
    "id_certification_ferc714": "iso-8859-1",
    "gen_plants_ba_ferc714": "iso-8859-1",
    "demand_monthly_ba_ferc714": None,
    "net_energy_load_ba_ferc714": None,
    "adjacency_ba_ferc714": "iso-8859-1",
    "interchange_ba_ferc714": "iso-8859-1",
    "lambda_hourly_ba_ferc714": None,
    "lambda_description_ferc714": "iso-8859-1",
    "description_pa_ferc714": "iso-8859-1",
    "demand_forecast_pa_ferc714": None,
    "demand_hourly_pa_ferc714": None,
    "respondent_id_ferc714": None,
}
"""Dictionary describing the character encodings of the FERC 714 CSV files."""


def _get_zpath(pudl_table, pudl_settings):
    """Given a table and pudl_settings, return a Path to the requested file."""
    return zipfile.Path(
        pathlib.Path(pudl_settings["data_dir"],
                     "local/ferc714/ferc714.zip"),
        TABLE_FNAME[pudl_table]
    )


def extract(tables=pc.pudl_tables["ferc714"], pudl_settings=None):
    """
    Extract the raw FERC Form 714 dataframes from their original CSV files.

    Args:
        ferc714_tables (iterable): The set of tables to be extracted.
        pudl_settings (dict): A PUDL settings dictionary.

    Returns:
        dict: A dictionary of dataframes, with raw FERC 714 table names as the
        keys, and minimally processed pandas.DataFrame instances as the values.

    """
    raw_dfs = {}
    for table in tables:
        if table not in pc.pudl_tables["ferc714"]:
            raise ValueError(
                f"No extract function found for requested FERC Form 714 data "
                f"table {table}!"
            )
        logger.info(f"Reading {table} from CSV into pandas DataFrame.")
        with _get_zpath(table, pudl_settings).open() as f:
            raw_dfs[table] = pd.read_csv(f, encoding=TABLE_ENCODING[table])
    return raw_dfs
