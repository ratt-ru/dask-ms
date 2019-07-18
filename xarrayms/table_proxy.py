# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from threading import Lock
import weakref

from xarrayms.new_executor import Executor

log = logging.getLogger(__name__)

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


NOLOCK = 0
READLOCK = 1
WRITELOCK = 2


class MismatchedLocks(Exception):
    pass


class TableProxy(TableProxyMetaClass("base", (object,), {})):
    def __init__(self, factory, *args, **kwargs):
        self._ex = ex = Executor()
        table = ex.submit(factory, *args, **kwargs).result()
        self._factory = factory
        self._args = args
        self._kwargs = kwargs

        # Ensure tables are closed when the object is deleted
        self._del_ref = proxy_delete_reference(self, ex, table)

        # Private, should be inaccessible
        self._table = table
        self._readlocks = 0
        self._writelocks = 0
        self._write = False
        self._writeable = table.iswritable()

    def __reduce__(self):
        """ Defer to _map_create_proxy to support kwarg pickling """
        return (_map_create_proxy, (TableProxy, self._factory,
                                    self._args, self._kwargs))

    def close(self):
        """" Closes the table immediately """
        try:
            self._ex.submit(self._table.close).result()
        except Exception:
            log.exception("Exception closing TableProxy")

    def nrows(self):
        """ Proxy nrows """
        return self._ex.submit(self._table.nrows)

    def getcol(self, *args, **kwargs):
        """ Proxy getcol """
        return self._ex.submit(self.__getcol_impl, args, kwargs)

    def __getcol_impl(self, args, kwargs):
        """ getcol implementation with readlocking """
        try:
            self._acquire(READLOCK)
            return self._table.getcol(*args, **kwargs)
        finally:
            self._release(READLOCK)

    def __enter__(self):
        return self

    def __exit__(self, evalue, etype, etraceback):
        self.close()

    def _acquire(self, locktype):
        """ Acquire a lock on the table """
        if locktype == READLOCK:
            # No locks at all, acquire readlock
            if self._readlocks + self._writelocks == 0:
                self._table.lock(write=False)

            self._readlocks += 1
        elif locktype == WRITELOCK:
            if not self._writeable:
                raise ValueError("Table is not writeable")

            # Acquire writelock if we had none previously
            if self._writelocks == 0:
                self._table.lock(write=True)
                self._write = True

            self._writelocks += 1
        elif locktype == NOLOCK:
            pass
        else:
            raise ValueError("Invalid lock type %d" % locktype)

    def _release(self, locktype):
        """ Release a lock on the table """
        if locktype == READLOCK:
            self._readlocks -= 1

            if self._readlocks == 0:
                if self._writelocks > 0:
                    # Should be write-locked, check the invariant
                    assert self._write is True
                else:
                    # Release all locks
                    self._table.unlock()
                    self._write = False
            elif self._readlocks < 0:
                raise MismatchedLocks("mismatched readlocks")

        elif locktype == WRITELOCK:
            self._writelocks -= 1

            if self._writelocks == 0:
                if self._readlocks > 0:
                    # Downgrade from write to read lock if
                    # there are any remaining readlocks
                    self._write = False
                    self._table.lock(write=False)
                else:
                    # Release all locks
                    self._write = False
                    self._table.unlock()
            elif self._writelocks < 0:
                raise MismatchedLocks("mismatched writelocks")

        elif locktype == NOLOCK:
            pass
        else:
            raise ValueError("Invalid lock type %d" % locktype)
