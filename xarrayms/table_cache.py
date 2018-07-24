import atexit
from collections import defaultdict
from contextlib import contextmanager

try:
    from dask.utils import SerializableLock as Lock
except ImportError:
    from threading import Lock

import pyrap.tables as pt


class TableCache(object):
    __lock = Lock()
    __instance = None
    __refcount = defaultdict(lambda: 0)
    __maxsize = 100
    __cache = {}

    def __init__(self, __maxsize=100):
        if self.__instance is not None:
            raise RuntimeError("Singleton already initialised")

    @classmethod
    def instance(cls):
        if cls.__instance is None:
            with cls.__lock:
                if cls.__instance is None:
                    cls.__instance = cls()

        return cls.__instance

    @classmethod
    @contextmanager
    def open(cls, table, **kwargs):
        key = (table, frozenset(kwargs.items()))
        with cls.__lock:
            try:
                table = cls.__cache[key]
            except KeyError:
                table = pt.table(table, **kwargs)
                table.unlock()
                cls.__cache[key] = table

            cls.__refcount[key] += 1

            # TODO(sjperkins)
            # Clear old tables intelligently
            if len(cls.__cache) > cls.__maxsize:
                raise ValueError("Cache size exceeded")

        try:
            yield table
        finally:
            with cls.__lock:
                # TODO(sjperkins)
                # Intelligently close tables whose __refcount drops to zero
                cls.__refcount[key] -= 1

    @classmethod
    def clear(cls):
        with cls.__lock:
            cls.__cache.clear()


atexit.register(lambda: TableCache.instance().clear())
