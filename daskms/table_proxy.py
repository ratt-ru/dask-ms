# -*- coding: utf-8 -*-

from itertools import zip_longest
import logging
from threading import Lock
import weakref

from dask.base import normalize_token
import pyrap.tables as pt
from daskms.table_executor import Executor, STANDARD_EXECUTOR
from daskms.utils import arg_hasher

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
    ("iswritable", READLOCK),
    # Modification
    ("addrows", WRITELOCK),
    ("addcols", WRITELOCK),
    ("setmaxcachesize", WRITELOCK),
    # Reads
    ("getcol", READLOCK),
    ("getcolslice", READLOCK),
    ("getcolnp", READLOCK),
    ("getvarcol", READLOCK),
    ("getcell", READLOCK),
    ("getcellslice", READLOCK),
    ("getkeywords", READLOCK),
    ("getcolkeywords", READLOCK),
    # Writes
    ("putcol", WRITELOCK),
    ("putcolslice", WRITELOCK),
    ("putcolnp", WRITELOCK),
    ("putvarcol", WRITELOCK),
    ("putcellslice", WRITELOCK),
    ("putkeyword", WRITELOCK),
    ("putkeywords", WRITELOCK),
    ("putcolkeywords", WRITELOCK)]


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

    if locktype == NOLOCK:
        def _impl(table_future, args, kwargs):
            try:
                return getattr(table_future.result(), method)(*args, **kwargs)
            except Exception:
                if logging.DEBUG >= log.getEffectiveLevel():
                    log.exception("Exception in %s", method)
                raise

    elif locktype == READLOCK:
        def _impl(table_future, args, kwargs):
            table = table_future.result()
            table.lock(write=False)

            try:
                return getattr(table, method)(*args, **kwargs)
            except Exception:
                if logging.DEBUG >= log.getEffectiveLevel():
                    log.exception("Exception in %s", method)
                raise
            finally:
                table.unlock()

    elif locktype == WRITELOCK:
        def _impl(table_future, args, kwargs):
            table = table_future.result()
            table.lock(write=True)

            try:
                return getattr(table, method)(*args, **kwargs)
            except Exception:
                if logging.DEBUG >= log.getEffectiveLevel():
                    log.exception("Exception in %s", method)
                raise
            finally:
                table.unlock()

    else:
        raise ValueError(f"Invalid locktype {locktype}")

    _impl.__name__ = method + "_impl"
    _impl.__doc__ = ("Calls table.%s, wrapped in a %s." %
                     (method, _LOCKTYPE_STRINGS[locktype]))

    def public_method(self, *args, **kwargs):
        """
        Submits _impl(args, kwargs) to the executor
        and returns a Future
        """
        return self._ex.submit(_impl, self._table_future, args, kwargs)

    public_method.__name__ = method
    public_method.__doc__ = _PROXY_DOCSTRING % method

    return public_method


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
        key = arg_hasher((cls,) + args + (kwargs,))

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


class MismatchedLocks(Exception):
    pass


def taql_factory(query, style='Python', tables=(), readonly=True):
    """ Calls pt.taql, converting TableProxy's in tables to pyrap tables """
    tables = [t._table_future.result() for t in tables]

    if isinstance(readonly, (tuple, list)):
        it = zip_longest(tables, readonly[:len(tables)],
                         fillvalue=readonly[-1])
    elif isinstance(readonly, bool):
        it = zip(tables, (readonly,)*len(tables))
    else:
        raise TypeError("readonly must be a bool or list of bools")

    for t, ro in it:
        t.lock(write=ro is False)

    try:
        return pt.taql(query, style=style, tables=tables)
    finally:
        for t in tables:
            t.unlock()


def _nolock_runner(table_future, fn, args, kwargs):
    try:
        return fn(table_future.result(), *args, **kwargs)
    except Exception:
        if logging.DEBUG >= log.getEffectiveLevel():
            log.exception("Exception in %s", fn.__name__)
        raise


def _readlock_runner(table_future, fn, args, kwargs):
    table = table_future.result()
    table.lock(write=False)

    try:
        return fn(table, *args, **kwargs)
    except Exception:
        if logging.DEBUG >= log.getEffectiveLevel():
            log.exception("Exception in %s", fn.__name__)
        raise
    finally:
        table.unlock()


def _writelock_runner(table_future, fn, args, kwargs):
    table = table_future.result()
    table.lock(write=True)

    try:
        result = fn(table, *args, **kwargs)
        table.flush()
    except Exception:
        if logging.DEBUG >= log.getEffectiveLevel():
            log.exception("Exception in %s", fn.__name__)
        raise
    else:
        return result
    finally:
        table.unlock()


def _iswriteable(table_future):
    return table_future.result().iswritable()


def _table_future_close(table_future):
    try:
        table_future.result().close()
    except BaseException as e:
        print(str(e))


# This finalizer submits the table closing operation
# on the Executor associated with the TableProxy
# Note that this operation **must** happen before
# shutdown is called on the Executor's internal
# ThreadPoolExecutor object or deadlock can occur
# See https://codewithoutrules.com/2017/08/16/concurrency-python/
# for further insight.
# The fact that the Executor is an argument to weakref.finalize implies
# that the Executor's finalizer, and by further implication,
# the ThreadPoolExecutor.shutdown has not yet been called.
def _table_future_finaliser(ex, table_future, args, kwargs):
    """
    Closes the table object in the thread it was created
    on Python >= 3.7 and in the garbage collection thread
    in Python 3.6.

    Parameters
    ----------
    ex : :class:`daskms.table_executor.Executor`
        Executor on which table operations should be performed
    table_future : :class:`concurrent.futures.Future`
        Future referring to the CASA table object. Should've
        been created in ``ex``.
    args : tuple
        Arguments used to create the table future by
        :class:`daskms.table_proxy.TableProxy`.
    kwargs : kwargs
        Keyword arguments used to create the table future by
        :class:`daskms.table_proxy.TableProxy`.

    Notes
    -----
    ``args`` and ``kwargs`` aren't used by the function but should
    be passed through so that any dependencies are garbage collected
    before the TableProxy is. Primarily this is to ensure that
    taql_proxy's are garbage collected before their dependant
    tables.

    """
    _table_future_close(table_future)


class TableProxy(object, metaclass=TableProxyMetaClass):
    """
    Proxies calls to a :class:`pyrap.tables.table` object via
    a :class:`concurrent.futures.ThreadPoolExecutor`.
    """

    def __init__(self, factory, *args, **kwargs):
        """
        Parameters
        ----------
        factory : callable
            Function which creates the CASA table
        *args : tuple
            Positional arguments passed to factory.
        **kwargs : dict
            Keyword arguments passed to factory.
        __executor_key__ : str, optional
            Executor key. Identifies a unique threadpool
            in which table operations will be performed.
        """

        # Save the arguments as keys for pickling and tokenising
        self._factory = factory
        self._args = args
        self._kwargs = kwargs

        # NOTE(sjperkins)
        # Copy the kwargs and remove (any) __executor_key__
        # This is smelly but we do this to maintain
        # key uniqueness derived from
        # the *args and **kwargs in the MetaClass
        # as well as uniqueness when pickling/unpickling
        # A named keyword is possible but
        # TableProxy(*args, *kwargs)
        # doesn't produce the same unique key as
        # TableProxy(*args, ex_key=..., **kwargs)
        kwargs = kwargs.copy()
        self._ex_key = kwargs.pop("__executor_key__", STANDARD_EXECUTOR)

        # Store a reference to the Executor wrapper class
        # so that the Executor is retained while this TableProxy
        # still lives
        self._ex_wrapper = ex = Executor(key=self._ex_key)
        self._table_future = table = ex.impl.submit(factory, *args, **kwargs)

        weakref.finalize(self, _table_future_finaliser, ex, table,
                         args, kwargs)

        # Reference to the internal ThreadPoolExecutor
        self._ex = ex.impl

        # Private, should be inaccessible
        self._write = False
        self._writeable = ex.impl.submit(_iswriteable, table).result()

        should_be_writeable = not kwargs.get('readonly', True)

        if self._writeable is False and should_be_writeable:
            # NOTE(sjperkins)
            # This seems to happen if you've opened a WSRT.MS with
            # readonly=False, and then you try to open WSRT.MS::SUBTABLE
            # with readonly=False.
            # Solution is to open WSRT.MS/SUBTABLE to avoid the locking,
            # which may introduce it's own set of issues
            raise RuntimeError("%s was opened as readonly=False but "
                               "table.iswritable()==False" % table.name())

    @property
    def executor_key(self):
        return self._ex_key

    def __reduce__(self):
        """ Defer to _map_create_proxy to support kwarg pickling """
        return (_map_create_proxy, (TableProxy, self._factory,
                                    self._args, self._kwargs))

    def __enter__(self):
        return self

    def __exit__(self, evalue, etype, etraceback):
        pass

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
        if locktype == NOLOCK:
            return self._ex.submit(_nolock_runner, self._table_future,
                                   fn, args, kwargs)
        elif locktype == READLOCK:
            return self._ex.submit(_readlock_runner, self._table_future,
                                   fn, args, kwargs)
        elif locktype == WRITELOCK:
            return self._ex.submit(_writelock_runner, self._table_future,
                                   fn, args, kwargs)
        else:
            raise ValueError(f"Invalid locktype {locktype}")

    def __repr__(self):
        return "TableProxy[%s](%s, %s, %s)" % (
            self._ex_key,
            self._factory.__name__,
            ",".join(str(s) for s in self._args),
            ",".join("%s=%s" % (str(k), str(v))
                     for k, v in self._kwargs.items()))

    __str__ = __repr__


@normalize_token.register(TableProxy)
def _normalize_table_proxy_tokens(tp):
    """ Generate tokens based on TableProxy arguments """
    return (tp._factory, tp._args, tp._kwargs)
