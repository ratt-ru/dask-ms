# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from threading import Lock
import weakref

import concurrent.futures as cf

from daskms.utils import table_path_split

log = logging.getLogger(__name__)

_executor_cache = weakref.WeakValueDictionary()
_executor_lock = Lock()


STANDARD_EXECUTOR = "__standard_executor__"


class ExecutorMetaClass(type):
    """ https://en.wikipedia.org/wiki/Multiton_pattern """
    def __call__(cls, key=STANDARD_EXECUTOR):
        with _executor_lock:
            try:
                return _executor_cache[key]
            except KeyError:
                instance = type.__call__(cls, key)
                _executor_cache[key] = instance
                return instance


def _finalise(tpe, lock, finalisers):
    with lock:
        cf.wait([tpe.submit(fn, *args, **kwargs) for
                 (fn, args, kwargs) in finalisers])
        tpe.shutdown(wait=True)


class Executor(object, metaclass=ExecutorMetaClass):
    def __init__(self, key=STANDARD_EXECUTOR):
        # Initialise a single thread
        self.impl = impl = cf.ThreadPoolExecutor(1)
        self.key = key
        self.lock = lock = Lock()
        self.finalisers = finalisers = []

        # Register a finaliser shutting down the
        # ThreadPoolExecutor on garbage collection
        weakref.finalize(self, _finalise, impl, lock, finalisers)

    def add_finaliser(self, fn, *args, **kwargs):
        with self.lock:
            self.finalisers.append((fn, args, kwargs))

    def shutdown(self, *args, **kwargs):
        return self.impl.shutdown(*args, **kwargs)

    def __reduce__(self):
        return (Executor, (self.key,))

    def __repr__(self):
        return "Executor(%s)" % self.key

    __str__ = __repr__


def executor_key(table_name):
    """
    Product an executor key from table_name
    """

    # Remove any path separators
    root, table_name, subtable = table_path_split(table_name)
    return str(Path(root, table_name))
