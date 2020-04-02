"""This module provides methods for timing the execution of the pipeline."""

# TODO: provide mechanisms for storing all timing records and dumping them into CSV
# at the end of the program run. Ideally, in the form of function_name, start_time, end_time, duration

import csv
import logging
import time

logger = logging.getLogger(__name__)


class TimerScope(object):
    _csv_log = None

    def __init__(self, name):
        self._name = name

    def __enter__(self):
        self._start_time = time.monotonic()
        return self

    def __exit__(self, _type, _value, _traceback):
        end_time = time.monotonic()
        duration = end_time - self._start_time

        if self._csv_log:
            self._csv_log.writerow(
                {'scope': self._name,
                 'start_time': self._start_time,
                 'end_time': end_time,
                 'duration': duration,
                 })

    def sub_timer(self, name):
        """Constructs inner timing scope by appending /name to the outer scope."""
        return TimerScope(f'{self._name}/{name}')

    @classmethod
    def enable_csv_logging(cls, filename):
        """When called, enables persistent logging of timing records into given file."""
        f = open(filename, 'w', newline='')
        cls._csv_log = csv.DictWriter(
            f, fieldnames=['scope', 'start_time', 'end_time', 'duration'])
        cls._csv_log.writeheader()


def timed_as(name):
    def timed_decorator(method):
        def inner_fcn(*args, **kw):
            with TimerScope(name):
                return method(*args, **kw)
        return inner_fcn
    return timed_decorator


# Quick decorator using default method name
def timed(method):
    return timed_as(method.__name__)(method)
