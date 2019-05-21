from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
from collections import defaultdict
from contextlib import contextmanager
import logging
import threading

from concurrent.futures import ThreadPoolExecutor

try:
    from dask.utils import SerializableLock as Lock
except ImportError:
    from threading import Lock

import pyrap.tables as pt

NOLOCK = 0
READLOCK = 1
WRITELOCK = 2

log = logging.getLogger(__name__)


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


_thread_local = threading.local()


def _table_create(table_name):
    """ Associate this thread with a specific table """
    _thread_local.table_name = table_name
    return True


def _table_close(table_name):
    """ Close this thread's table """
    try:
        wrapper = _thread_local.wrapper
    except AttributeError:
        pass
    else:
        try:
            wrapper.close()
        except Exception:
            log.exception("Error closing table")

        del _thread_local.wrapper

    return True


def _check_table_name(table_name):
    """ Sanity check """
    if _thread_local.table_name != table_name:
        raise ValueError("Thread local table name '%s' "
                         "does not match table name '%s' "
                         "of calling thread." % (
                            _thread_local.table_name, table_name))


@contextmanager
def _locked_table(locktype):
    """
    Context Manager which yields a table, guarded by
    lock and unlock on either side.
    """

    # Get the wrapper or create it
    try:
        w = _thread_local.wrapper
    except AttributeError:
        _thread_local.wrapper = w = TableWrapper(_thread_local.table_name)

    try:
        w.acquire(locktype)
        yield w.table
    finally:
        w.release(locktype)


def _getcol(*args, **kwargs):
    with _locked_table(READLOCK) as table:
        return table.getcol(*args, **kwargs)


def _getcolnp(*args, **kwargs):
    with _locked_table(READLOCK) as table:
        return table.getcolnp(*args, **kwargs)


def _putcol(*args, **kwargs):
    with _locked_table(WRITELOCK) as table:
        return table.putcol(*args, **kwargs)


class TableExecutor(object):
    """
    Singleton class providing CASA Table IO operations
    isolated within a single thread per table
    """
    __cache = {}
    __refcounts = defaultdict(lambda: 0)
    __cache_lock = Lock()

    @classmethod
    def register(cls, table_name):
        """
        Registers a table with the Executor table cache.
        """
        with cls.__cache_lock:
            # Create a new executor or bump the reference count
            # on the new one
            try:
                executor = cls.__cache[table_name]
            except KeyError:
                cls.__cache[table_name] = executor = ThreadPoolExecutor(1)
                cls.__refcounts[table_name] = 1
                return executor.submit(_table_create, table_name)
            else:
                cls.__refcounts[table_name] += 1
                return executor.submit(lambda: True)

    @classmethod
    def deregister(cls, table_name):
        """
        Deregisters a table with the Executor table cache.
        """
        with cls.__cache_lock:
            try:
                executor = cls.__cache[table_name]
            except KeyError:
                raise KeyError("Table '%s' not registered with the executor")
            else:
                cls.__refcounts[table_name] -= 1

                if cls.__refcounts[table_name] == 0:
                    f = executor.submit(_table_close, table_name)
                    executor.shutdown(wait=False)
                    del cls.__cache[table_name]
                    del cls.__refcounts[table_name]
                    return f
                elif cls.__refcounts[table_name] < 0:
                    raise ValueError("Invalid condition")
                else:
                    return executor.submit(lambda: True)

    @classmethod
    def close(cls, wait=False):
        """ Closes the pool and associated table cache """
        with cls.__cache_lock:
            for table_name, executor in cls.__cache.items():
                executor.submit(_table_close, table_name)
                executor.shutdown(wait=wait)

            cls.__cache.clear()
            cls.__refcounts.clear()

    @classmethod
    def getcol(cls, table_name, *args, **kwargs):
        """ Returns a future calling a getcol on the table """
        with cls.__cache_lock:
            try:
                executor = cls.__cache[table_name]
            except KeyError:
                raise ValueError("Table '%s' not registered" % table_name)

            return executor.submit(_getcol, *args, **kwargs)

    @classmethod
    def getcolnp(cls, table_name, *args, **kwargs):
        """ Returns a future calling a getcolnp on the table """
        with cls.__cache_lock:
            try:
                executor = cls.__cache[table_name]
            except KeyError:
                raise ValueError("Table '%s' not registered" % table_name)

            return executor.submit(_getcolnp, *args, **kwargs)

    @classmethod
    def putcol(cls, table_name, *args, **kwargs):
        """ Returns a future calling a putcol on the table """
        with cls.__cache_lock:
            try:
                executor = cls.__cache[table_name]
            except KeyError:
                raise ValueError("Table '%s' not registered" % table_name)

            return executor.submit(_putcol, *args, **kwargs)


atexit.register(TableExecutor.close, wait=False)


class TableProxy(object):
    """
    An object referencing a CASA table, suitable for embedding within
    a dask graph. The most frequently used table methods are proxied
    on this object, submitting work on a single IO thread associated
    with the Table in the TableExecutor class and returning Futures.

    .. code-block:: python

        proxy = TableProxy("WSRT.MS")
        future = proxy.getcol("TIME", startrow=10, nrow=10)
        time = future.result()

    """
    def __init__(self, table_name):
        self.table_name = table_name

        if TableExecutor.register(table_name).result() is False:
            raise ValueError("Table '%s' registration failed" %
                             self.table_name)

    def close(self):
        """ Closes this TableProxy's link to the CASA table """
        if TableExecutor.deregister(self.table_name).result() is False:
            raise ValueError("Table '%s' deregistration failed" %
                             self.table_name)

    def getcol(self, *args, **kwargs):
        """ Proxies :meth:`pyrap.tables.table.getcol` """
        return TableExecutor.getcol(self.table_name, *args, **kwargs)

    def getcolnp(self, *args, **kwargs):
        """ Proxies :meth:`pyrap.tables.table.getcolnp` """
        return TableExecutor.getcolnp(self.table_name, *args, **kwargs)

    def putcol(self, *args, **kwargs):
        """ Proxies :meth:`pyrap.tables.table.putcol` """
        return TableExecutor.putcol(self.table_name, *args, **kwargs)
