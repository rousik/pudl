"""Task abstraction functions.

This allows wrapping PUDL transformation into tasks that can be executed
by a library of sorts.
"""

import re


def fq_name(df_name, stage='raw', force_stage=False):
    """Constructs fully qualified dataframe name.

    Args:
      df_name (str): dataframe name with or without a stage
      stage (str): stage that should be appended to the name.
      force_stage (bool): if True then the stage will be set
        even if it is present in df_name. Otherwise the stage
        will be preserved.
    """
    if ':' in df_name and not force_stage:
        return df_name
    else:
        base_name = df_name.split(':')[0]
        return f'{base_name}:{stage}'


def is_fq_name(name):
    return re.match(r'\w+/\w+:\w+')


class DataFrameTaskDecorator(object):
    """Single input single output transformer."""

    def __init__(self, inputs=[], output=None):
        assert output and is_fq_name(output), f'Invalid output: {output}'
        assert inputs, 'Need at least one input'
        for i in inputs:
            assert is_fq_name(
                i), f'Input {i} not fully qualified dataframe name.'
        self.inputs = inputs
        self.output = output

    def __call__(self, function):
        self.function = function
        Registry.add_task(self)
        # TODO: we could consider wrapping the function in a simple wrapper
        # that can pull raw/transformed inputs from first/second argument
        # and set the transformed output in the second argument map.

        # This way the old way of PUDL-ing would still work :-)
        return function


def transforms(df_name, output_stage='transformed', input_stage='raw'):
    """Reads dataframe and emits dataframe with a given stage."""
    return DataFrameTaskDecorator(
        inputs=[fq_name(df_name, input_stage)],
        output=fq_name(df_name, output_stage, force_stage=True))


def transforms_single(in_df_name, out_df_name):
    assert is_fq_name(in_df_name), f'{in_df_name} not fully qualified'
    assert is_fq_name(out_df_name), f'{out_df_name} not fully qualified'
    return DataFrameTaskDecorator(inputs=[in_df_name], output=out_df_name)


class Registry(object):
    all_tasks = []

    @classmethod
    def add_task(cls, task):
        cls.all_tasks.append(task)

    @classmethod
    def tasks(cls):
        return list(cls.all_tasks)

# In order to execute things we now need to:

# - construct external inputs (from extract phase)
# - construct luigi.Task objects for each abstract task in Registry
# - initialize working dir for the etl run (random or given)
