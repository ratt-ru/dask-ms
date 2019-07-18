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


class Executor(ExecutorMetaClass("base", (object,), {})):
    def __init__(self):
        # Initialise a single thread
        self.ex = ex = cf.ThreadPoolExecutor(1)

        # http://pydev.blogspot.com/2015/01/creating-safe-cyclic-reference.html
        def _callback(ref):
            # to avoid cyclic references, self may
            # not be used within this function
            try:
                ex.shutdown(wait=True)
            except Exception:
                log.exception("Error shutting down executor in _callback")

        self.__del_ref = weakref.ref(self, _callback)

    def submit(self, *args, **kwargs):
        return self.ex.submit(*args, **kwargs)

    def shutdown(self, *args, **kwargs):
        return self.ex.shutdown(*args, **kwargs)

    def __reduce__(self):
        return (Executor, ())  # No args/kwargs, they're ignored in __init__


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

        # http://pydev.blogspot.com/2015/01/creating-safe-cyclic-reference.html
        def _callback(ref):
            # to avoid cyclic references, self may
            # not be used within this function
            try:
                ex.submit(table.close).result()
            except Exception:
                log.exception("Error closing table in _callback")

        self.__del_ref = weakref.ref(self, _callback)

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
