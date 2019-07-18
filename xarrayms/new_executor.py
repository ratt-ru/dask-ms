# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from threading import Lock
import weakref

import concurrent.futures as cf

log = logging.getLogger(__name__)

_executor_cache = weakref.WeakValueDictionary()
_executor_lock = Lock()


class ExecutorMetaClass(type):
    def __call__(cls, *args, **kwargs):
        with _executor_lock:
            try:
                return _executor_cache["key"]
            except KeyError:
                instance = type.__call__(cls, *args, **kwargs)
                _executor_cache["key"] = instance
                return instance


def executor_delete_reference(ex, threadpool_executor):
    # http://pydev.blogspot.com/2015/01/creating-safe-cyclic-reference.html
    # To avoid cyclic references, self may not be used within _callback
    def _callback(ref):
        try:
            threadpool_executor.shutdown(wait=True)
        except Exception:
            log.exception("Error shutting down executor in _callback")

    return weakref.ref(ex, _callback)


class Executor(ExecutorMetaClass("base", (object,), {})):
    def __init__(self):
        # Initialise a single thread
        self.ex = ex = cf.ThreadPoolExecutor(1)
        self.__del_ref = executor_delete_reference(self, ex)

    def submit(self, *args, **kwargs):
        return self.ex.submit(*args, **kwargs)

    def shutdown(self, *args, **kwargs):
        return self.ex.shutdown(*args, **kwargs)

    def __reduce__(self):
        return (Executor, ())

