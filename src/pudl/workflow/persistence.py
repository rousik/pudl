"""Implements dataframe persistence."""
import logging
import os

import pandas as pd

from pudl.workflow.task import fq_name, is_fq_name

logger = logging.getLogger(__name__)


class AbstractPersistence(object):
    """Persists panda dataframes as parquets.

    Use fully qualified name ${dataset}/${df_name}:${stage} to
    store or retrieve dataframes.

    Dataframes are identified using their fully-qualified names:

      ${dataset}/${dataframe_name}:${stage}

    They are simply persisted on disk in the feather format.
    """

    def __init__(self, working_dir_path):
        self._working_dir_path = working_dir_path
        os.makedirs(working_dir_path, exist_ok=True)

    def df_path(self, df_name):
        assert is_fq_name(df_name), f"df name not fully qualified: {df_name}"
        p = os.path.join(self._working_dir_path,
                         fq_name(df_name).replace(':', '.'))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def get(self, ref):
        """Retrieve dataframe from a feather file."""
        raise NotImplementedError('Abstract method not implemented.')

    def set(self, ref):
        raise NotImplementedError('Abstract method not implemented.')

    def exists(self, ref):
        return os.path.exists(self.df_path(ref.name()))


class Pickle(AbstractPersistence):
    def get(self, ref):
        # TODO: time these methods, emit summary of time spent (de)serializing.
        return pd.read_pickle(self.df_path(ref.name()), compression=None)

    def set(self, ref, df):
        return df.to_pickle(self.df_path(ref.name()), compression=None)
