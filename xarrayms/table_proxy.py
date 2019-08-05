# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from threading import Lock
import weakref

from dask.base import normalize_token
import pyrap.tables as pt
from xarrayms.table_executor import Executor
from xarrayms.utils import with_metaclass

log = logging.getLogger(__name__)

_table_cache = weakref.WeakValueDictionary()
_table_lock = Lock()

# CASA Table Locking Modes
NOLOCK = 0
READLOCK = 1
WRITELOCK = 2

# List of CASA Table methods to proxy and the appropriate locking mode
_proxied_methods = [
    # Queries
    ("nrows", NOLOCK),
    ("colnames", NOLOCK),
    ("getcoldesc", NOLOCK),
    ("getdminfo", NOLOCK),
    # Reads
    ("getcol", READLOCK),
    ("getcolnp", READLOCK),
    ("getvarcol", READLOCK),
    ("getcellslice", READLOCK),
    # Writes
    ("putcol", WRITELOCK),
    ("putcolnp", WRITELOCK),
    ("putvarcol", WRITELOCK),
    ("putcellslice", WRITELOCK)]


_PROXY_DOCSTRING = ("""
Proxies calls to pyrap.tables.table.%s
via a concurrent.futures.ThreadPoolExecutor

Returns
-------
future : concurrent.futures.Future
    Future containing the result of the call
""")


def proxied_method_factory(method, locktype):
    """
    Proxy pyrap.tables.table.method calls.

    Creates a private implementation function which performs
    the call locked according to to ``locktype``.

    The private implementation is accessed by a public ``method``
    which submits a call to the implementation
    on a concurrent.futures.ThreadPoolExecutor.
    """

    def _impl(self, args, kwargs):
        """ Calls method on the table, acquiring the appropriate lock """
        self._acquire(locktype)

        try:
            return getattr(self._table, method)(*args, **kwargs)
        finally:
            self._release(locktype)

    _impl.__name__ = method + "_impl"

    def public_method(self, *args, **kwargs):
        """
        Submits _impl(args, kwargs) to the executor
        and returns a Future
        """
        return self._ex.submit(_impl, self, args, kwargs)

    public_method.__name__ = method
    public_method.__doc__ = _PROXY_DOCSTRING % method

    return public_method


def _hasher(args):
    """ Recursively hash data structures -- handles list and dicts """
    if isinstance(args, (tuple, list, set)):
        return hash(tuple(_hasher(v) for v in args))
    elif isinstance(args, dict):
        return hash(tuple((k, _hasher(v)) for k, v in sorted(args.items())))
    else:
        return hash(args)


class TableProxyMetaClass(type):
    def __new__(cls, name, bases, dct):
        for method, locktype in _proxied_methods:
            proxy_method = proxied_method_factory(method, locktype)
            proxy_method.__name__ = method
            proxy_method.__doc__ = _PROXY_DOCSTRING
            dct[method] = proxy_method

        return type.__new__(cls, name, bases, dct)

    def __call__(cls, *args, **kwargs):
        key = _hasher((cls,) + args + (kwargs,))

        with _table_lock:
            try:
                return _table_cache[key]
            except KeyError:
                instance = type.__call__(cls, *args, **kwargs)
                _table_cache[key] = instance
                return instance


def _close_table(table):
    tabstr = hash(str(table))
    log.debug("Closing %s", tabstr)
    try:
        table.close()
    except Exception:
        log.exception("Error closing %s", tabstr)
        raise
    finally:
        log.debug("Finished closing %s", tabstr)


def proxy_delete_reference(table_proxy, ex, table):
    # http://pydev.blogspot.com/2015/01/creating-safe-cyclic-reference.html
    # To avoid cyclic references, table_proxy may not be used within _callback
    # Something wierd was happening on kernsuite 3 that caused this to fail
    # Upgrading to kernsuite 5 fixed things. See the following commit
    # https://github.com/ska-sa/xarray-ms/pull/41/commits/af5126acf1646887ca59ce14680093988d32e333
    def _callback(ref):
        try:
            ex.impl.submit(_close_table, table).result()
        except Exception:
            log.exception("Error closing table in _callback")

    return weakref.ref(table_proxy, _callback)


def _map_create_proxy(cls, factory, args, kwargs):
    """ Support pickling of kwargs in TableProxy.__reduce__ """
    return cls(factory, *args, **kwargs)


class MismatchedLocks(Exception):
    pass


def taql_factory(query, style='Python', tables=[]):
    """ Calls pt.taql, converting TableProxy's in tables to pyrap tables """
    tabs = [t._table for t in tables]

    for t in tables:
        t._acquire(READLOCK)

    try:
        return pt.taql(query, style=style, tables=tabs)
    finally:
        for t in tables:
            t._release(READLOCK)


@with_metaclass(TableProxyMetaClass)
class TableProxy(object):
    def __init__(self, factory, *args, **kwargs):
        # Save the arguments as keys for pickling and tokenising
        self._factory = factory
        self._args = args
        self._kwargs = kwargs

        ex = Executor()
        table = ex.impl.submit(factory, *args, **kwargs).result()

        # Ensure tables are closed when the object is deleted
        self._del_ref = proxy_delete_reference(self, ex, table)

        self._ex = ex.impl

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

    def __enter__(self):
        return self

    def __exit__(self, evalue, etype, etraceback):
        pass

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

    def __repr__(self):
        return "TableProxy(%s, %s, %s)" % (
                    self._factory.__name__,
                    ",".join(str(s) for s in self._args),
                    ",".join("%s=%s" % (str(k), str(v))
                             for k, v in self._kwargs.items()))

    __str__ = __repr__


@normalize_token.register(TableProxy)
def _normalize_table_proxy_tokens(tp):
    """ Generate tokens based on TableProxy arguments """
    return (tp._factory, tp._args, tp._kwargs)
