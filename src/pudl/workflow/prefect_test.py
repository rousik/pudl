"""Unit tests for the workflow prefect integration."""

import unittest

import pudl.workflow.prefect as pf
import pudl.workflow.task as task


class TestFlowBuilding(unittest.TestCase):

    def test_prune_invalid_tasks_all_good(self):
        a = task.PudlTask(inputs=[], output='a', function=None)
        b = task.PudlTask(inputs=[], output='b', function=None)
        c = task.PudlTask(inputs=['a', 'b'], output='c', function=None)
        self.assertEquals([a, b, c], pf.prune_invalid_tasks([a, b, c]))

    def test_find_invalid_tasks_direct_deps(self):
        a = task.PudlTask(inputs=[], output='a', function=None)
        b = task.PudlTask(inputs=['x'], output='b', function=None)
        self.assertEquals([a], pf.prune_invalid_tasks([a, b]))

    def test_find_invalid_tasks_indirect(self):
        a = task.PudlTask(inputs=[], output='a', function=None)
        b = task.PudlTask(inputs=['x'], output='b', function=None)
        c = task.PudlTask(inputs=['b'], output='c', function=None)
        self.assertEquals([a], pf.prune_invalid_tasks([a, b, c]))
