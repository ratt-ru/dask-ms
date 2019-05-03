import atexit
from collections import defaultdict
from contextlib import contextmanager

try:
    from dask.utils import SerializableLock as Lock
except ImportError:
    from threading import Lock

import pyrap.tables as pt

NOLOCK = 0
READLOCK = 1
WRITELOCK = 2


class MismatchedLocks(Exception):
    pass


class LockContext(object):
    def __init__(self, rwlock, locktype):
        self._rwlock = rwlock
        self._locktype = locktype

    def __enter__(self):
        self._rwlock.acquire(self._locktype)
        return self._rwlock.table

    def __exit__(self, type, value, tb):
        if tb is None:
            self._rwlock.release(self._locktype)


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
        self.readlocks = 0
        self.writelocks = 0
        self.write = False
        self.writeable = self.table.iswritable()
        self.refcount = 0

    def locked_table(self, locktype):
        return LockContext(self, locktype)

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

        self.refcount += 1

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

        # TODO(sjperkins)
        # Intelligently close tables whose refcount drops to zero
        self.refcount -= 1


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
    @contextmanager
    def open(cls, key, locktype, table_name, table_kwargs):
        with cls.__cache_lock:
            try:
                table_wrapper = cls.__cache[key]
            except KeyError:
                table_wrapper = TableWrapper(table_name, table_kwargs)
                cls.__cache[key] = table_wrapper

            # TODO(sjperkins)
            # Clear old tables intelligently
            if len(cls.__cache) > cls.__maxsize:
                raise ValueError("Cache size exceeded")

            table_wrapper.acquire(locktype)

        try:
            yield table_wrapper.table
        finally:
            with cls.__cache_lock:
                table_wrapper.release(locktype)

    @classmethod
    def clear(cls):
        with cls.__cache_lock:
            cls.__cache.clear()


atexit.register(lambda: TableCache.instance().clear())
