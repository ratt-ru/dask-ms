# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from threading import Lock
import weakref

from dask.base import normalize_token
import numpy as np
import six
import pyrap.tables as pt
from daskms.table_executor import Executor

log = logging.getLogger(__name__)

_table_cache = weakref.WeakValueDictionary()
_table_lock = Lock()

# CASA Table Locking Modes
NOLOCK = 0
READLOCK = 1
WRITELOCK = 2

_LOCKTYPE_STRINGS = {
    0: 'NOLOCK',
    1: 'READLOCK',
    2: 'WRITELOCK'
}


# List of CASA Table methods to proxy and the appropriate locking mode
_proxied_methods = [
    # Queries
    ("nrows", READLOCK),
    ("colnames", READLOCK),
    ("getcoldesc", READLOCK),
    ("getdminfo", READLOCK),
    # Modification
    ("addrows", WRITELOCK),
    ("addcols", WRITELOCK),
    # Reads
    ("getcol", READLOCK),
    ("getcolnp", READLOCK),
    ("getvarcol", READLOCK),
    ("getcell", READLOCK),
    ("getcellslice", READLOCK),
    # Writes
    ("putcol", WRITELOCK),
    ("putcolnp", WRITELOCK),
    ("putvarcol", WRITELOCK),
    ("putcellslice", WRITELOCK)]


_PROXY_DOCSTRING = ("""
Proxies calls to :func:`~pyrap.tables.table.%s`
via a :class:`~concurrent.futures.ThreadPoolExecutor`

Returns
-------
future : :class:`concurrent.futures.Future`
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
        self._acquire(locktype)

        try:
            return getattr(self._table, method)(*args, **kwargs)
        finally:
            self._release(locktype)

    _impl.__name__ = method + "_impl"
    _impl.__doc__ = ("Calls table.%s, wrapped in a %s." %
                     (method, _LOCKTYPE_STRINGS[locktype]))

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
    elif isinstance(args, np.ndarray):
        # NOTE(sjperkins)
        # https://stackoverflow.com/a/16592241/1611416
        # Slowish, but we shouldn't be passing
        # huge numpy arrays in the TableProxy constructor
        return hash(args.tostring())
    else:
        return hash(args)


class TableProxyMetaClass(type):
    """
    https://en.wikipedia.org/wiki/Multiton_pattern

    """
    def __new__(cls, name, bases, dct):
        for method, locktype in _proxied_methods:
            proxy_method = proxied_method_factory(method, locktype)
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


def proxy_delete_reference(table_proxy, table):
    # http://pydev.blogspot.com/2015/01/creating-safe-cyclic-reference.html
    # To avoid cyclic references, table_proxy may not be used within _callback
    def _callback(ref):
        # We close the table **without** using the executor due to
        # reentrancy issues with Python queues and garbage collection
        # https://codewithoutrules.com/2017/08/16/concurrency-python/
        # There could be internal casacore issues here, due to accessing
        # the table from a different thread, but test cases are passing
        tabstr = hash(str(table))
        log.debug("Begin closing %s", tabstr)

        try:
            table.close()
        except Exception:
            log.exception("Error closing %s", tabstr)
            raise
        finally:
            log.debug("Finished closing %s", tabstr)

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


@six.add_metaclass(TableProxyMetaClass)
class TableProxy(object):
    """
    Proxies calls to a :class:`pyrap.tables.table` object via
    a :class:`concurrent.futures.ThreadPoolExecutor`.
    """

    def __init__(self, factory, *args, **kwargs):
        # Save the arguments as keys for pickling and tokenising
        self._factory = factory
        self._args = args
        self._kwargs = kwargs

        ex = Executor()
        table = ex.impl.submit(factory, *args, **kwargs).result()

        # Ensure tables are closed when the object is deleted
        self._del_ref = proxy_delete_reference(self, table)

        # Store a reference to the Executor wrapper class
        # so that the Executor is retained while this TableProxy
        # still lives
        self._ex_wrapper = ex
        # Reference to the internal ThreadPoolExecutor
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

    def __runner(self, fn, locktype, args, kwargs):
        self._acquire(locktype)

        try:
            return fn(self._table, *args, **kwargs)
        finally:
            self._release(locktype)

    def submit(self, fn, locktype, *args, **kwargs):
        """
        Submits :code:`fn(table, *args, **kwargs)` within
        the executor, returning a Future.

        Parameters
        ----------
        fn : callable
            Function with signature :code:`fn(table, *args, **kwargs)`
        locktype : {NOLOCK, READLOCK, WRITELOCK}
            Type of lock to acquire before and release
            after calling `fn`
        *args :
            Arguments passed to `fn`
        **kwargs :
            Keyword arguments passed to `fn`

        Returns
        -------
        future : :class:`concurrent.futures.Future`
            Future containing the result of :code:`fn(table, *args, **kwargs)`
        """
        return self._ex.submit(self.__runner, fn, locktype, args, kwargs)

    def _acquire(self, locktype):
        """
        Acquire a lock on the table

        Notes
        -----
        This should **only** be called from within the associated Executor
        """
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
        """
        Release a lock on the table

        Notes
        -----
        This should **only** be called from within the associated Executor
        """
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
