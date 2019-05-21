from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
from contextlib import contextmanager

from concurrent.futures import ThreadPoolExecutor

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


class TableWrapper(object):
    """
    Wrapper around the casacore table locking mechanism.

    The following invariants should always hold

    .. code-block:: python

        l = TableWrapper(table)
        l.table.haslock(l.write) is (l.readlocks + l.writelocks > 0)
        l.write is (l.writelocks > 0)

    """  #

    def __init__(self, table_name):
        self.table_name = table_name
        self.table_kwargs = {'readonly': False, 'ack': False}
        self.table = pt.table(table_name, **self.table_kwargs)
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


class TableProxy(object):
    """
    An object referencing a CASA table, suitable for embedding within
    a dask graph. The most frequently used table methods are proxied
    on this object, submitting work on a single IO thread in the
    TableExecutor class and returning Futures.
    """
    def __init__(self, table_name):
        self.table_name = table_name
        TableExecutor.instance().submit(TableExecutor.register, table_name)

    def close(self):
        """
        Returns a future indicating that the TableProxy is no longer
        interested in this table
        """
        return TableExecutor.instance().submit(TableExecutor.deregister,
                                               self.table_name).result()

    def getcol(self, *args, **kwargs):
        """ Returns a future calling a getcol on the table """
        return TableExecutor.instance().submit(TableExecutor.getcol,
                                               self.table_name,
                                               *args, **kwargs)

    def getcolnp(self, *args, **kwargs):
        """ Returns a future calling a getcolnp on the table """
        return TableExecutor.instance().submit(TableExecutor.getcolnp,
                                               self.table_name,
                                               *args, **kwargs)

    def putcol(self, *args, **kwargs):
        """ Returns a future calling a putcol on the table """
        return TableExecutor.instance().submit(TableExecutor.putcol,
                                               self.table_name,
                                               *args, **kwargs)


class TableExecutor(object):
    """
    Singleton class providing Measurement Set IO operations
    within a single thread
    """
    __pool = None
    __pool_lock = Lock()

    __cache = {}

    @classmethod
    def instance(cls):
        if cls.__pool is None:
            with cls.__pool_lock:
                if cls.__pool is None:
                    cls.__pool = ThreadPoolExecutor(1)

        return cls.__pool

    @classmethod
    def register(cls, table_name):
        """
        Registers a table with the Executor table cache.
        This must only be called from the ThreadPoolExecutor
        """
        try:
            wrapper = cls.__cache[table_name]
        except KeyError:
            wrapper = TableWrapper(table_name)
            cls.__cache[table_name] = wrapper

        wrapper.refcount += 1

        return True

    @classmethod
    def deregister(cls, table_name):
        """
        Deregisters a table with the Executor table cache.
        This must only be called from the ThreadPoolExecutor
        """
        try:
            wrapper = cls.__cache[table_name]
        except KeyError:
            return False

        if wrapper.refcount == 1:
            del cls.__cache[table_name]
            wrapper.close()

        wrapper.refcount -= 1

        return True

    @staticmethod
    def _close_tables(table_cache):
        for name, wrapper in table_cache.items():
            wrapper.close()

        table_cache.clear()

    @classmethod
    def close(cls, wait=False):
        """ Closes the pool and associated table cache """
        with cls.__pool_lock:
            if cls.__pool is not None:
                cls.__pool.submit(cls._close_tables, cls.__cache)
                cls.__pool.shutdown(wait=wait)
                cls.__pool = None

    @classmethod
    @contextmanager
    def _locked_table(cls, table_name, locktype):
        """
        Context Manager which yields a table, guarded by
        lock and unlock on either side.
        """
        try:
            wrapper = cls.__cache[table_name]
        except KeyError:
            raise ValueError("Table '%s' not registered")

        try:
            wrapper.acquire(locktype)
            yield wrapper.table
        finally:
            wrapper.release(locktype)

    @classmethod
    def getcolnp(cls, table_name, *args, **kwargs):
        """
        Proxies :func:`~pyrap.tables.getcolnp`
        This must only be called from the ThreadPoolExecutor
        """
        with cls._locked_table(table_name, READLOCK) as table:
            return table.getcolnp(*args, **kwargs)

    @classmethod
    def getcol(cls, table_name, *args, **kwargs):
        """
        Proxies :func:`~pyrap.tables.getcol`
        This must only be called from the ThreadPoolExecutor
        """
        with cls._locked_table(table_name, READLOCK) as table:
            return table.getcol(*args, **kwargs)

    @classmethod
    def putcol(cls, table_name, *args, **kwargs):
        """
        Proxies :func:`~pyrap.tables.putcol`
        This must only be called from the ThreadPoolExecutor
        """
        with cls._locked_table(table_name, WRITELOCK) as table:
            return table.putcol(*args, **kwargs)


atexit.register(TableExecutor.close, wait=False)
