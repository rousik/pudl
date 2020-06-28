"""Workflow task abstraction layer.

This module provides lightweight API for building table transformation tasks
that can be passed to a worfklow automation library for execution.

Table transformations are functions that take a certain number of pandas
data frames holding table data and emits one panda data frame with results.

Each table usually undergoes series of transformation steps that move the
table between pre-defined stages.

PudlTableTransformer is an object that is responsible for transforming
a given table through the known stages.
"""

# TODO(rousik): Get rid of some existing magic, such as:
# - instead of relying on class-hierarchy, explicitly decorate classes
#   that will do the transformations.
# - possibly: explicitly indicate the table name given object is responsible
#   for. Right now we do CamelCase to camel_case conversion which is a bit
#   mysterious.
# - Extractors should also be marked with "this makes tasks" annotation or
#   object mixin, which should be used separately from PudlTableTransformer
#   descendance.

import logging
import re
from collections import namedtuple
from enum import Enum, auto

import pudl.constants as pc

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
    ASSIGN_DTYPE = auto()
    TRANSFORMED = auto()
    FINAL = auto()
    # TODO(rousik): do we want to provide documentation/description for
    # these stages?


class PudlTableReference(object):
    """Instance of this class points to a pudl table in a specific stage.

    Each table is uniquely identified by its dataset (eia860, eia923, ...),
    the table name (e.g. generators) and stage (see Stage enum above).
    """

    def __init__(self, table_name, dataset=None, stage=None):
        self.table_name = table_name
        self.dataset = dataset
        self.stage = stage
        if dataset not in pc.data_sources:
            raise KeyError(
                f'Unsupported data_source {dataset} used when creating tableref.')

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


class reads(object):
    """Decorator for indicating when additional PudlTableReferences are needed."""

    def __init__(self, *table_references):
        self.references = table_references

    def __call__(self, fcn):
        fcn.pudl_table_references = self.references
        return fcn


class emits(object):
    """Annotates method with emits_stage."""
    # TODO(rousik): we could consider annotating with full tableref

    def __init__(self, stage):
        self.stage = stage

    def __call__(self, fcn):
        fcn.emits_stage = self.stage
        return fcn


def transformer(cls):
    """Mark the class for inclusion in the workflow."""
    cls.is_table_transformer = True
    return cls


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
        return PudlTableReference(table_name, dataset=cls.DATASET, stage=stage)

    @classmethod
    def get_table_name(cls):
        cls_name = cls.__name__
        return re.sub(r'(?<!^)(?=[A-Z])', '_', cls_name).lower()

    @classmethod
    def get_stage(cls, stage):
        return cls.table_ref(cls.get_table_name(), stage)

    @classmethod
    def marked_as_transformer(cls):
        """Returns True if this instance has been decorated with @transformer.

        This decoration indicates that transformations within should be run
        within the workflow.
        """
        return hasattr(cls, 'is_table_transformer')

    @classmethod
    def get_transformer_subclasses(cls):
        """Returns all subclasses that have methods annotated with @emits."""
        res = set()
        for sc in cls.__subclasses__():
            if sc.marked_as_transformer():
                res.add(sc)
            res.update(sc.get_transformer_subclasses())
        return res

    @classmethod
    def get_transformations(cls):
        """Returns {Stage: method} map."""
        tfs = {}
        for attr_name in dir(cls):
            fcn = getattr(cls, attr_name)
            emits_stage = getattr(fcn, 'emits_stage', None)
            if emits_stage:
                tfs[emits_stage] = fcn
        return tfs

    @classmethod
    def generate_tasks(cls):
        last_stage = Stage.RAW
        tasks = []
        stage_functions = cls.get_transformations()

        # Process transformations in the Stage order.
        # TODO(rousik): we could potentially associate Stage.TRANSFORMED
        # with the last available stage there is.
        for stage in Stage.__members__.values():
            fcn = stage_functions.get(stage, None)
            if not fcn:
                continue

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


PudlTask = namedtuple('PudlTask', ['inputs', 'output', 'function'])
"""Generic task for pudl dataframe transformations."""
