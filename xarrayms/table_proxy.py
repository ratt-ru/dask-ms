# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import logging
from threading import Lock
import weakref

import concurrent.futures as cf
import pyrap.tables as pt


log = logging.getLogger(__name__)


class ExecutorSingleton(object):
    _ex_lock = Lock()
    _ex_instance = None
    _ex_del_ref = None

    @classmethod
    def instance(cls):
        # Double-locking
        if cls._ex_instance is None:
            with cls._ex_lock:
                if cls._ex_instance is None:
                    cls._ex_instance = ex = cf.ThreadPoolExecutor(1)

                    def _callback(ref):
                        try:
                            ex.shutdown(wait=True)
                        except Exception:
                            log.exception("Error closing executor "
                                          "in weakref callback")
                        else:
                            print("Shutdown executor")

                    cls._ex_del_ref = weakref.ref(cls, _callback)

        return cls._ex_instance

    @classmethod
    def close(cls):
        if cls._ex_instance is not None:
            with cls._ex_lock:
                cls._ex_instance.shutdown(wait=True)


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
        f = ExecutorSingleton.instance().submit(factory, *args, **kwargs)

        self._table = table = f.result()
        self._factory = factory
        self._args = args
        self._kwargs = kwargs

        # http://pydev.blogspot.com/2015/01/creating-safe-cyclic-reference.html
        def _callback(ref):
            # to avoid cyclic references, self may
            # not be used within this function
            try:
                ExecutorSingleton.instance().submit(table.close).result()
            except Exception:
                log.exception("Error closing table in weakref callback")

        self.__del_ref = weakref.ref(self, _callback)

    def __reduce__(self):
        """ Defer to _map_create_proxy to support kwarg pickling """
        return (_map_create_proxy, (TableProxy, self._factory,
                                    self._args, self._kwargs))

    def close(self):
        self._table.close()

    def __enter__(self):
        return self

    def __exit__(self, evalue, etype, etraceback):
        self.close()


atexit.register(ExecutorSingleton.close)
