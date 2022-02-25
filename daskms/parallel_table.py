import logging
import threading
from daskms.table_proxy import (TableProxy,
                                TableProxyMetaClass,
                                _proxied_methods,
                                NOLOCK,
                                READLOCK,
                                WRITELOCK,
                                _LOCKTYPE_STRINGS,
                                _PROXY_DOCSTRING,
                                STANDARD_EXECUTOR)
from weakref import finalize, WeakValueDictionary
from daskms.utils import arg_hasher

_table_cache = WeakValueDictionary()
_table_lock = threading.Lock()

log = logging.getLogger(__name__)


_parallel_methods = [
    "getcol",
    "getcolnp",
    "getcolslice",
    "getvarcol",
    "getcell",
    "getcellslice",
    "getkeywords",
    "getcolkeywords"
]


def _parallel_table_finalizer(table_cache):

    for table in table_cache.cache.values():
        table.close()


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
            if isinstance(table_future, TIDCache):
                table = table_future.get()
            else:
                table = table_future.result()

            try:
                return getattr(table, method)(*args, **kwargs)
            except Exception:
                if logging.DEBUG >= log.getEffectiveLevel():
                    log.exception("Exception in %s", method)
                raise

    elif locktype == READLOCK:
        def _impl(table_future, args, kwargs):
            if isinstance(table_future, TIDCache):
                table = table_future.get()
            else:
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
            if isinstance(table_future, TIDCache):
                table = table_future.get()
            else:
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

    if method in _parallel_methods:
        def public_method(self, *args, **kwargs):
            """
            Submits _impl(args, kwargs) to the executor
            and returns a Future
            """
            return self._ex.submit(_impl, self._cached_tables, args, kwargs)
    else:
        def public_method(self, *args, **kwargs):
            """
            Submits _impl(args, kwargs) to the executor
            and returns a Future
            """
            return self._ex.submit(_impl, self._table_future, args, kwargs)

    public_method.__name__ = method
    public_method.__doc__ = _PROXY_DOCSTRING % method

    return public_method


class ParallelTableProxyMetaClass(TableProxyMetaClass):
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


class ParallelTableProxy(TableProxy, metaclass=ParallelTableProxyMetaClass):

    @classmethod
    def _from_args_kwargs(cls, factory, args, kwargs):
        """Support pickling of kwargs in ParallelTableProxy.__reduce__."""
        return cls(factory, *args, **kwargs)

    def __init__(self, factory, *args, **kwargs):

        super().__init__(factory, *args, **kwargs)

        # This is a bit hacky, as noted by Simon in TableProxy. Maybe storing
        # a sanitised version would be better?
        kwargs = self._kwargs.copy()
        kwargs.pop("__executor_key__", STANDARD_EXECUTOR)

        self._cached_tables = TIDCache(factory, *args, **kwargs)

        finalize(self, _parallel_table_finalizer, self._cached_tables)

    def __reduce__(self):
        """ Defer to _from_args_kwargs to support kwarg pickling """
        return (
            self._from_args_kwargs,
            (self._factory, self._args, self._kwargs)
        )


class TIDCache(object):

    def __init__(self, fn, *args, **kwargs):
        """A cache keyed on thread ID.

        When a key is not found, it is added with a value given by
        fn(*args, **kwargs).
        """

        self.cache = {}
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def get(self):
        """Return or create fn(*args, **args) for the calling thread."""

        thread_id = threading.get_ident()

        try:
            item = self.cache[thread_id]
        except KeyError:
            print(f"Opening item in {thread_id}.")
            self.cache[thread_id] = item = self.fn(*self.args, **self.kwargs)

        return item
