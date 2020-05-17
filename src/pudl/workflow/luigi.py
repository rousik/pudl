import logging
import os

import luigi
import pandas as pd

from pudl.workflow.task import Registry, fq_name, is_fq_name

logger = logging.getLogger(__name__)


class DataFramePersistence(object):
    """Dataframe persistence layer based on on-disk feather files.

    Use fully qualified name ${dataset}/${df_name}:${stage} to
    store or retrieve dataframes.

    Dataframes are identified using their fully-qualified names:

      ${dataset}/${dataframe_name}:${stage}

    They are simply persisted on disk in the feather format.
    """

    def __init__(self, working_dir_path):
        self.working_dir_path = working_dir_path

    def df_path(self, df_name):
        assert is_fq_name(df_name), f"df name not fully qualified: {df_name}"
        return os.path.join(self.working_dir_path, fq_name(df_name))

    def get(self, df_name):
        """Retrieve dataframe from a feather file."""
        # TODO(jaro): time this method
        return pd.read_feather(self.df_path(df_name))

    def set(self, df_name, df):
        """Store dataframe into a feather file."""
        # TODO(jaro): time this method
        return df.to_feather(self.df_path(df_name))

    def get_target(self, df_name):
        """Constructs LocalTarget for a given df."""
        luigi.LocalTarget(self.df_path(df_name))


class DataFrameTask(luigi.Task):
    df_persistence = luigi.Parameter()
    input_dfs = luigi.Parameter()
    output_df = luigi.Parameter()
    function = luigi.Parameter()

    def input(self):
        return [self.df_persistence.get_task(df) for df in self.input_dfs]

    def output(self):
        return [self._df_target(self.output_df)]

    def run(self):
        dfs = [self.df_persistence.get(df) for df in self.input_dfs]
        out = self.function(*dfs)
        self.df_persistence.set(self.output_df, out)

    # no unpicklable properties are df_persistence and function


def build_luigi_graph(input_dfs, workspace_dir=None):
    df_cache = DataFramePersistence(workspace_dir)
    luigi_tasks = []

    # set up tasks for externally originating dataframes (input_dfs)
    for df_name, df in input_dfs.items():
        df_name = fq_name(df_name)

        class ExternalDataFrame(luigi.ExternalTask):
            def output():
                return df_cache.get_target(df_name)
        luigi_tasks.append(ExternalDataFrame())
        df_cache.set(df_name, df)

    for meta_task in Registry.tasks():
        luigi_tasks.append(DataFrameTask(
            df_persistence=df_cache,
            input_dfs=meta_task.inputs,
            output_df=meta_task.output,
            function=meta_task.function))

    return luigi.build(luigi_tasks, workers=1, local_scheduler=True)
