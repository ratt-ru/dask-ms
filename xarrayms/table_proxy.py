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


_table_cache = weakref.WeakValueDictionary()
_table_lock = Lock()


class TableProxyMetaClass(type):
    def __call__(cls, *args, **kwargs):
        key = (cls,) + args + (frozenset(kwargs.items()),)

        with _table_lock:
            try:
                return _table_cache[key]
            except KeyError:
                instance = type.__call__(cls, *args, **kwargs)
                _table_cache[key] = instance
                return instance


def proxy_delete_reference(table_proxy, ex, table):
    # http://pydev.blogspot.com/2015/01/creating-safe-cyclic-reference.html
    # To avoid cyclic references, self may not be used within _callback
    def _callback(ref):
        try:
            ex.submit(table.close).result()
        except Exception:
            log.exception("Error closing table in _callback")

    return weakref.ref(table_proxy, _callback)


def _map_create_proxy(cls, factory, args, kwargs):
    """ Support pickling of kwargs in TableProxy.__reduce__ """
    return cls(factory, *args, **kwargs)


class TableProxy(TableProxyMetaClass("base", (object,), {})):
    def __init__(self, factory, *args, **kwargs):
        self._ex = ex = Executor()
        self._table = table = ex.submit(factory, *args, **kwargs).result()
        self._factory = factory
        self._args = args
        self._kwargs = kwargs
        self.__del_ref = proxy_delete_reference(self, ex, table)

    def __reduce__(self):
        """ Defer to _map_create_proxy to support kwarg pickling """
        return (_map_create_proxy, (TableProxy, self._factory,
                                    self._args, self._kwargs))

    def close(self):
        try:
            self._ex.submit(self._table.close).result()
        except Exception:
            log.exception("Exception closing TableProxy")

    def __enter__(self):
        return self

    def __exit__(self, evalue, etype, etraceback):
        self.close()
