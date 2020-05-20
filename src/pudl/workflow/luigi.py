import logging

import luigi
from pudl.workflow import persistence
from pudl.workflow.task import Registry, fq_name

logger = logging.getLogger(__name__)


class DataFrameTask(luigi.Task):
    df_persistence = luigi.Parameter(significant=False)
    # TODO(rousik): ^ we could replace this just with a workspace_dir
    input_dfs = luigi.Parameter()
    output_df = luigi.Parameter()
    function = luigi.Parameter(significant=False)
    # TODO(rousik): ^ this is not exactly insignificant
    # TODO(rousik): if we mark input_dfs as insignificant, we could probably
    # use def requires(self): instead of input_dfs

    def input(self):
        # TODO: we should be using tasks, not outputs here
        return [self.df_persistence.get_task(df) for df in self.input_dfs]

    def output(self):
        return [luigi.LocalTarget(self.df_persistence.df_path(self.output_df))]

    def run(self):
        dfs = [self.df_persistence.get(df) for df in self.input_dfs]
        out = self.function(*dfs)
        self.df_persistence.set(self.output_df, out)

    # no unpicklable properties are df_persistence and function


def build_luigi_graph(input_dfs, workspace_dir=None):
    df_cache = persistence.Parquet(workspace_dir)
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

    logger.info(f'Constructing luigi graph from {len(luigi_tasks)} tasks.')
    return luigi.build(luigi_tasks, workers=1, local_scheduler=True)
