# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from threading import Lock
import weakref

import concurrent.futures as cf
import six

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


@six.add_metaclass(ExecutorMetaClass)
class Executor(object):
    def __init__(self):
        # Initialise a single thread
        self.impl = impl = cf.ThreadPoolExecutor(1)
        self.__del_ref = executor_delete_reference(self, impl)

    def shutdown(self, *args, **kwargs):
        return self.impl.shutdown(*args, **kwargs)

    def __reduce__(self):
        return (Executor, ())
