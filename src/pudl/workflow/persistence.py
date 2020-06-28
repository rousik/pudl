"""Implements dataframe persistence."""
import logging
import os
import time

import pandas as pd
from pudl.workflow.task import PudlTableReference

logger = logging.getLogger(__name__)


class Pickle(object):
    # TODO(rousik): add call-count and time-spent for get/set methods
    # to analyze overhead.
    # Time accumulator that allows calling like:
    # with self.named_timer('get'):
    #   return ...
    # and this will increment call counter by one and calculate time
    # it took for this block to execute.
    def __init__(self, working_dir_path):
        self._working_dir_path = working_dir_path
        self._pickle_timing = {}  # method_name -> time spent in these
        os.makedirs(working_dir_path, exist_ok=True)

    def _add_timing(self, method_name, duration):
        self._pickle_timing[method_name] = (
            self._pickle_timing.get(method_name, 0.0) + duration)

    def _df_path(self, df_name):
        return os.path.join(self._working_dir_path, df_name.replace(':', '.'))

    def get(self, table_ref):
        assert isinstance(table_ref, PudlTableReference)
        if not self.exists(table_ref):
            raise RuntimeError(f'Pickled file for {table_ref} does not exist.')
        t1 = time.monotonic()
        data = pd.read_pickle(self._df_path(
            table_ref.name()), compression=None)
        self._add_timing('get', time.monotonic() - t1)
        return data

    def set(self, table_ref, df):
        assert isinstance(table_ref, PudlTableReference)
        p = self._df_path(table_ref.name())
        os.makedirs(os.path.dirname(p), exist_ok=True)
        t1 = time.monotonic()
        result = df.to_pickle(p, compression=None)
        self._add_timing('set', time.monotonic() - t1)
        return result

    def exists(self, table_ref):
        assert isinstance(table_ref, PudlTableReference)
        return os.path.exists(self._df_path(table_ref.name()))

    def get_overhead(self):
        """Return number of seconds spent de/serializing the data."""
        return dict(self._pickle_timing)
