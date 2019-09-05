# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


def executor_delete_reference(ex, threadpool_executor):
    # http://pydev.blogspot.com/2015/01/creating-safe-cyclic-reference.html
    # To avoid cyclic references, ex may not be used within _callback
    def _callback(ref):
        # This is possibly a bad idea because the ThreadPoolExecutor
        # puts None on queues to signal to threads that they should
        # exit. However, if the callback is called, nothing should be
        # referring to the executor anymore so it should be OK (TM).
        # For more information, please reread:
        # https://codewithoutrules.com/2017/08/16/concurrency-python/
        try:
            threadpool_executor.shutdown(wait=True)
        except Exception:
            log.exception("Error shutting down executor in _callback")

    return weakref.ref(ex, _callback)


class Executor(object, metaclass=ExecutorMetaClass):
    def __init__(self, key=STANDARD_EXECUTOR):
        # Initialise a single thread
        self.impl = impl = cf.ThreadPoolExecutor(1)
        self.key = key
        self.__del_ref = executor_delete_reference(self, impl)

    def shutdown(self, *args, **kwargs):
        return self.impl.shutdown(*args, **kwargs)

    def __reduce__(self):
        return (Executor, ())

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
