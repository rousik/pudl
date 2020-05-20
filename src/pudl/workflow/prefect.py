import logging

import prefect

from pudl.workflow import persistence, task

logger = logging.getLogger(__name__)


# use fq_df_name as a task name/slug
# TODO: we could consider equipping DataFrameTask with persistence hook
# that saves/loads artifacts to parquet files if we want to.
class DataFrameTask0(prefect.Task):
    def __init__(self, fcn, output, persistence=None, **kwargs):
        self.fcn = fcn
        self.df_persistence = persistence
        self.output = output
        super().__init__(name=output.name(),
                         tags=[output.stage.name], **kwargs)

    def persist(self, df):
        if not self.df_persistence:
            return df
        self.df_persistence.set(self.output, df)
        return df

    def run(self):
        if self.df_persistence and self.df_persistence.exists(self.output):
            return self.df_persistence.get(self.output)
        else:
            return self.persist(self.fcn())

    def get_result(self):
        if not self.df_persistence:
            raise NotImplementedError(
                'Cannot retrieve dataframe without persistence.')
        return self.df_persistence.get(self.output)


class DataFrameTask1(DataFrameTask0):
    def run(self, df):
        if self.df_persistence and self.df_persistence.exists(self.output):
            return self.df_persistence.get(self.output)
        else:
            return self.persist(self.fcn(df.copy()))


class DataFrameTask2(DataFrameTask0):
    def run(self, df1, df2):
        if self.df_persistence and self.df_persistence.exists(self.output):
            return self.df_persistence.get(self.output)
        else:
            return self.persist(self.fcn(df1.copy(), df2.copy()))


class DataFrameTask3(DataFrameTask0):
    def run(self, df1, df2, df3):
        if self.df_persistence and self.df_persistence.exists(self.output):
            return self.df_persistence.get(self.output)
        else:
            return self.persist(self.fcn(df1.copy(), df2.copy(), df3.copy()))


def get_task_obj(num_arguments):
    """Get the n-ary DataFrameTask."""
    if num_arguments > 3 or num_arguments < 0:
        raise NotImplementedError(
            'Tasks with more than 3 inputs not supported.')
    classes = [DataFrameTask0, DataFrameTask1, DataFrameTask2, DataFrameTask3]
    return classes[num_arguments]


def build_flow(tmp_dir, task_filter=None, raise_on_unmet_dependencies=False):
    prefect_tasks = {}  # output -> prefect_task
    df_persistence = persistence.Pickle(tmp_dir)

    # join things using Registry (old style) and the PudlTableTransformers
    all_meta_tasks = []
    all_meta_tasks.extend(task.Registry.tasks())
    for tf_class in task.PudlTableTransformer.get_subclasses_with_transformations():
        all_meta_tasks.extend(tf_class.generate_tasks())

    pruned_tasks = prune_invalid_tasks(all_meta_tasks)
    if len(pruned_tasks) < len(all_meta_tasks):
        known_inputs = set(t.output for t in pruned_tasks)
        num_pruned = len(all_meta_tasks) - len(pruned_tasks)
        logger.warning(f'{num_pruned} tasks pruned due to invalid inputs.')
        for t in all_meta_tasks:
            if t not in pruned_tasks:
                bad_inputs = sorted(str(i) for i in set(
                    t.inputs).difference(known_inputs))
                logger.warning(
                    f'Task generating {t.output} has bad inputs: {bad_inputs}')
        if raise_on_unmet_dependencies:
            raise task.InvalidDependencies(
                f'{num_pruned} tasks have unknown dependencies.')

    for meta_task in pruned_tasks:
        if task_filter and not task_filter(meta_task.output):
            continue
        # TODO(rousik): this filtering is kind of crude and good for testing only, eliminate.
        t = get_task_obj(len(meta_task.inputs))(
            meta_task.function,
            meta_task.output,
            persistence=df_persistence,
            max_retries=0)
        prefect_tasks[meta_task.output] = {
            'task': t,
            'inputs': meta_task.inputs
        }

    # TODO(rousik): nuke this thing, we don't need this print once we are confident
    # things work well.
    for df in sorted(x.name() for x in list(prefect_tasks)):
        print(df)

    # Add all the tasks into prefect flow now
    with prefect.Flow('pudl-etl') as flow:
        for out, p_task in prefect_tasks.items():
            input_tasks = [prefect_tasks[i]['task'] for i in p_task['inputs']]
            logging.info(f'Binding {len(input_tasks)} inputs to task {out}')
            flow.add_task(p_task['task'])
            p_task['task'].bind(*input_tasks)
        return flow


def prune_invalid_tasks(tasks):
    """Remove tasks that transitively depend on unknown inputs."""
    good_tasks = list(tasks)
    while True:
        # debug stuffsies
        known_outputs = set(t.output for t in good_tasks)
        still_good = [t for t in good_tasks if not set(
            t.inputs).difference(known_outputs)]
        if len(still_good) == len(good_tasks):
            return good_tasks
        else:
            good_tasks = still_good
