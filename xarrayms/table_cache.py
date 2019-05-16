import atexit
from collections import defaultdict
from contextlib import contextmanager
import logging
from pprint import pformat

try:
    from dask.utils import SerializableLock as Lock
except ImportError:
    from threading import Lock

import pyrap.tables as pt

log = logging.getLogger(__name__)

NOLOCK = 0
READLOCK = 1
WRITELOCK = 2


class MismatchedLocks(Exception):
    pass


class TableWrapper(object):
    """
    Wrapper around the casacore table locking mechanism.

    The following invariants should always hold

    .. code-block:: python

        l = TableWrapper(table)
        l.table.haslock(l.write) is (l.readlocks + l.writelocks > 0)
        l.write is (l.writelocks > 0)

    """  #
    def __init__(self, table_name, table_kwargs):
        self.table_name = table_name
        self.table_kwargs = table_kwargs
        self.table = pt.table(table_name, **table_kwargs)
        self.lock = Lock()
        self.readlocks = 0
        self.writelocks = 0
        self.write = False
        self.writeable = self.table.iswritable()
        self.refcount = 0

    def close(self):
        if self.table is not None:
            self.table.close()

    def acquire(self, locktype):
        if locktype == READLOCK:
            # No locks at all, acquire readlock
            if self.readlocks + self.writelocks == 0:
                self.table.lock(write=False)

            self.readlocks += 1
        elif locktype == WRITELOCK:
            if not self.writeable:
                raise ValueError("Table is not writeable")

            # Acquire writelock if we had none previously
            if self.writelocks == 0:
                self.table.lock(write=True)
                self.write = True

            self.writelocks += 1
        elif locktype == NOLOCK:
            pass
        else:
            raise ValueError("Invalid lock type %d" % locktype)

    def release(self, locktype):
        if locktype == READLOCK:
            self.readlocks -= 1

            if self.readlocks == 0:
                if self.writelocks > 0:
                    # Should be write-locked, check the invariant
                    assert self.write is True
                else:
                    # Release all locks
                    self.table.unlock()
                    self.write = False
            elif self.readlocks < 0:
                raise MismatchedLocks("mismatched readlocks")

        elif locktype == WRITELOCK:
            self.writelocks -= 1

            if self.writelocks == 0:
                if self.readlocks > 0:
                    # Downgrade from write to read lock if
                    # there are any remaining readlocks
                    self.write = False
                    self.table.lock(write=False)
                else:
                    # Release all locks
                    self.write = False
                    self.table.unlock()
            elif self.writelocks < 0:
                raise MismatchedLocks("mismatched writelocks")

        elif locktype == NOLOCK:
            pass
        else:
            raise ValueError("Invalid lock type %d" % locktype)


class TableCache(object):
    __cache_lock = Lock()
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
            with cls.__cache_lock:
                if cls.__instance is None:
                    cls.__instance = cls()

        return cls.__instance

    @classmethod
    def register(cls, table_name, table_kwargs):
        # key = hash((table_name, frozenset(table_kwargs.items())))
        key = table_name

        with cls.__cache_lock:
            try:
                table_wrapper = cls.__cache[key]
            except KeyError:
                table_wrapper = TableWrapper(table_name, table_kwargs)
                cls.__cache[key] = table_wrapper

            table_wrapper.refcount += 1

        if table_wrapper.table_kwargs != table_kwargs:
            log.warn("Table kwarg mismatch\n%s\nvs\n%s" % (
                                pformat(table_kwargs),
                                pformat(table_wrapper.table_kwargs)))

        return key

    @classmethod
    def deregister(cls, key):
        with cls.__cache_lock:
            try:
                table_wrapper = cls.__cache[key]
            except KeyError:
                return

            if table_wrapper.refcount == 1:
                table_wrapper.close()
                del cls.__cache[key]

    @classmethod
    @contextmanager
    def acquire(cls, key, locktype):
        with cls.__cache_lock:
            try:
                table_wrapper = cls.__cache[key]
            except KeyError:
                raise KeyError("No key '%s' registered in table cache" % key)

        with table_wrapper.lock:
            table_wrapper.acquire(locktype)

            try:
                yield table_wrapper.table
            finally:
                table_wrapper.release(locktype)

    @classmethod
    def clear(cls):
        with cls.__cache_lock:
            for k, table_wrapper in cls.__cache.items():
                table_wrapper.close()

            cls.__cache.clear()


atexit.register(TableCache.clear)
