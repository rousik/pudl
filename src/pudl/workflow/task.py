"""Task abstraction functions.

This allows wrapping PUDL transformation into tasks that can be executed
by a library of sorts.
"""

import logging
import re
from collections import namedtuple
from enum import Enum, auto

logger = logging.getLogger(__name__)


class Stage(Enum):
    """This defines order and names of all known stages.

    For most part, pudl transformations will take table in a given stage,
    apply changes and emit table in the subsequent stage.

    We do not need to implement all stages for each table.
    """
    RAW = auto()
    TIDY = auto()
    CLEAN = auto()
    TRANSFORMED = auto()


class PudlTableReference(object):
    """Pointer to a specific stage of a pudl table."""

    def __init__(self, table_name, dataset=None, stage=None):
        self.table_name = table_name
        self.dataset = dataset
        self.stage = stage

    def __str__(self):
        return f'ref<{self.name()}>'

    def __hash__(self):
        return hash((self.dataset, self.table_name, self.stage))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.table_name == other.table_name and
                    self.dataset == other.dataset and
                    self.stage == other.stage)
        else:
            return False

    def name(self):
        """Returns string represenation of this reference."""
        return f'{self.dataset}/{self.table_name}:{self.stage.name}'

    @staticmethod
    def from_fq_name(fq_name):
        """Returns reference instance from fully qualified name."""
        # TODO(rousik): this is used to support legacy decorator. Kill that.
        base, stage = fq_name.split(':')
        ds, table = base.split('/')
        stage = Stage[stage.upper()]
        return PudlTableReference(table, ds, stage)


class reads(object):
    """Decorator for indicating when additional PudlTableReferences are needed."""

    def __init__(self, *table_references):
        self.references = table_references

    def __call__(self, fcn):
        fcn.pudl_table_references = self.references
        return fcn


class PudlTableTransformer(object):
    """Base class for defining pudl table transformer objects.

    By default, name of this class is converted from CamelCase to
    snake_case and used as a table_name.
    """
    DATASET = None

    @classmethod
    def for_dataset(cls, dataset):
        """Returns instance of PudlTableTransformer with dataset fixed to a given value.

        Use this to simplify creation of transformers associated with a dataset, e.g.:

        ```
        tf = PudlTableTransformer.for_dataset('eia820')

        class MyTransformer(tf):
            # do stuff
        ```
        """
        class TransformerForDataset(cls):
            DATASET = dataset
        return TransformerForDataset

    @classmethod
    def table_ref(cls, table_name, stage=Stage.RAW):
        """Returns PudlTableReference for self.DATASET."""
        return PudlTableReference(cls.DATASET, table_name, stage=stage)

    @classmethod
    def get_table_name(cls):
        cls_name = cls.__name__
        return re.sub(r'(?<!^)(?=[A-Z])', '_', cls_name).lower()

    @classmethod
    def get_stage(cls, stage):
        return PudlTableReference(cls.DATASET, cls.get_table_name(), stage)

    @classmethod
    def has_transformations(cls):
        """Returns True if any transformer methods corresponding to known Stage exist."""
        for attr in dir(cls):
            if attr.lower() in Stage.__members__:
                return True
        return False

    @classmethod
    def get_subclasses_with_transformations(cls):
        """Returns all subclasses that have defined transformations."""
        res = set()
        for sc in cls.__subclasses__():
            if sc.has_transformations():
                res.add(sc)
                res.update(sc.get_subclasses_with_transformations())
        return res

    @classmethod
    def generate_tasks(cls):
        attrs = dir(cls)
        last_stage = Stage.RAW
        tasks = []
        known_stages = []
        for name, stage in Stage.__members__.items():
            if name.tolower() in attrs:
                known_stages.append(stage)
                inputs = []
                fcn = getattr(cls, name.tolower())
                if hasattr(fcn, 'pudl_table_references'):
                    # @reads decorator defines what the inputs should be.
                    inputs = fcn.pudl_table_references
                else:
                    inputs = [cls.get_stage(last_stage)]

                tasks.append(
                    PudlTask(inputs=inputs, output=cls.get_stage(stage), function=fcn))
                last_stage = stage
        return tasks


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
    return re.match(r'\w+/\w+:\w+', name)


PudlTask = namedtuple('PudlTask', ['inputs', 'output', 'function'])
"""Generic task for pudl dataframe transformations."""


class DataFrameTaskDecorator(object):
    """Single input single output transformer."""

    def __init__(self, inputs=[], output=None):
        assert output and is_fq_name(output), f'Invalid output: {output}'
        assert inputs, 'Need at least one input'
        for i in inputs:
            assert is_fq_name(
                i), f'Input {i} not fully qualified dataframe name.'
        # Convert fq_names to PudlTableReferences
        self.inputs = [PudlTableReference.from_fq_name(i) for i in inputs]
        self.output = PudlTableReference.from_fq_name(output)

    def __call__(self, fcn):
        Registry.add_task(PudlTask(inputs=self.inputs,
                                   output=self.output, function=fcn))
        # TODO(rousik): we could consider wrapping the function in a simple wrapper
        # that can pull raw/transformed inputs from first/second argument
        # and set the transformed output in the second argument map.

        # This way the old way of PUDL-ing would still work :-)
        return fcn


def transforms(df_name, output_stage='transformed', input_stage='raw'):
    """Reads dataframe and emits dataframe with a given stage."""
    return DataFrameTaskDecorator(
        inputs=[fq_name(df_name, input_stage)],
        output=fq_name(df_name, output_stage, force_stage=True))


def transforms_single(in_df_name, out_df_name):
    assert is_fq_name(in_df_name), f'{in_df_name} not fully qualified'
    assert is_fq_name(out_df_name), f'{out_df_name} not fully qualified'
    return DataFrameTaskDecorator(inputs=[in_df_name], output=out_df_name)


def transforms_many(in_df_names, out_df_name, input_stage='raw', output_stage='transformed'):
    assert type(in_df_names) == list, 'in_df_names needs to be list'
    assert type(out_df_name) == str, 'out_f_name needs to be string'
    fq_in = [fq_name(x, input_stage) for x in in_df_names]
    fq_out = fq_name(out_df_name, output_stage)
    return DataFrameTaskDecorator(inputs=fq_in, output=fq_out)


class Registry(object):
    _all_tasks = []

    @classmethod
    def add_task(cls, task):
        cls._all_tasks.append(task)

    @classmethod
    def tasks(cls):
        return list(cls._all_tasks)

# In order to execute things we now need to:

# - construct external inputs (from extract phase)
# - construct luigi.Task objects for each abstract task in Registry
# - initialize working dir for the etl run (random or given)
