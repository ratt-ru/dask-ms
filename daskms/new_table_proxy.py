from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import ClassVar, Mapping, Any, Tuple

from dask.base import normalize_token

from daskms.multiton import FrozenKey, normalise_args, MultitonMetaclass

from cacheout import LRUCache

FactoryFunctionT = Callable[..., Any]


FIVE_MINUTES = 5 * 60.0


class ExecutorMultiton(metaclass=MultitonMetaclass):
    pass


# CASA Table Locking Modes
NOLOCK = 0
READLOCK = 1
WRITELOCK = 2


log = logging.getLogger(__name__)


def _no_lock_impl(table, method, args, kwargs):
    try:
        return getattr(table, method)(*args, **kwargs)
    except Exception:
        if logging.DEBUG >= log.getEffectiveLevel():
            log.exception(f"Exception in {method}")
        raise


def _read_lock_impl(table, method, args, kwargs):
    table.lock(write=False)

    try:
        return getattr(table, method)(*args, **kwargs)
    except Exception:
        if logging.DEBUG >= log.getEffectiveLevel():
            log.exception(f"Exception in {method}")
        raise
    finally:
        table.unlock()


def _write_lock_impl(table, method, args, kwargs):
    table.lock(write=True)

    try:
        return getattr(table, method)(*args, **kwargs)
    except Exception:
        if logging.DEBUG >= log.getEffectiveLevel():
            log.exception(f"Exception in {method}")
        raise
    finally:
        table.unlock()


def proxied_method_factory(method, locktype):
    if locktype == NOLOCK:
        impl = _no_lock_impl
    elif locktype == READLOCK:
        impl = _read_lock_impl
    elif locktype == WRITELOCK:
        impl = _write_lock_impl
    else:
        raise ValueError(f"Invalid locktype {locktype}")

    def public_method(self, *args, **kwargs):
        f"""Submits {method}(args, kwargs) to the executor and returns the Future result"""
        return self._pool.instance.submit(
            impl, self.instance, method, args, kwargs
        ).result()

    return public_method


def dummy_init(*args) -> None:
    pass


# List of CASA Table methods to  proxy and the appropriate locking mode
METHOD_LOCKS = [
    # Queries
    ("nrows", READLOCK),
    ("colnames", READLOCK),
    ("getcoldesc", READLOCK),
    ("getdminfo", READLOCK),
    ("iswritable", READLOCK),
    # Modification
    ("addrows", WRITELOCK),
    ("addcols", WRITELOCK),
    ("setmaxcachesize", READLOCK),
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
    ("putcolkeywords", WRITELOCK),
]


def on_get_keep_alive(key, value, exists):
    """Re-insert on get to update the TTL"""
    if exists:
        # Re-insert to update the TTL
        TableProxy._CACHE.set(key, value)


def on_delete(key, value, cause):
    """Invoke any instance close methods on deletion"""
    if hasattr(value, "close") and callable(value.close):
        value.close()


class TableProxy:
    """Hashable and pickleable factory class
    for creating and caching an object instance"""

    _CACHE: ClassVar[LRUCache] = LRUCache(
        maxsize=100, ttl=5 * 60, on_get=on_get_keep_alive, on_delete=on_delete
    )
    _factory: FactoryFunctionT
    _args: Tuple[Any, ...]
    _kw: Mapping[str, Any]
    _executor_key: str
    _key: FrozenKey

    def __init__(
        self,
        factory: FactoryFunctionT,
        *args: Any,
        __executor_key__: str = "DEFAULT",
        **kw: Any,
    ):
        self._factory = factory
        self._args, self._kw = normalise_args(factory, args, kw)
        self._key = FrozenKey(
            factory, *self._args, __executor_key__=__executor_key__, **self._kw
        )
        self._executor_key = __executor_key__
        self._pool = ExecutorMultiton(
            ThreadPoolExecutor,
            initializer=dummy_init,
            initargs=(__executor_key__,),
            max_workers=1,
        )

    @staticmethod
    def from_reduce_args(
        factory: FactoryFunctionT,
        args: Tuple[Any, ...],
        executor_key: str,
        kw: Mapping[str, Any],
    ) -> TableProxy:
        return TableProxy(factory, *args, __executor_key__=executor_key, **kw)

    def __reduce__(
        self,
    ) -> Tuple[Callable, Tuple[Callable, Tuple[Any, ...], str, Mapping[str, Any]]]:
        return (
            self.from_reduce_args,
            (self._factory, self._args, self._executor_key, self._kw),
        )

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TableProxy):
            return NotImplemented
        return self._key == other._key

    @staticmethod
    def _create_instance(self) -> Any:
        return self._factory(*self._args, **self._kw)

    @property
    def instance(self) -> Any:
        """Create the object instance represented by the TableProxy,
        or retrieved the cache instance"""
        return self._CACHE.get(self, self._create_instance)

    def release(self) -> bool:
        """Evict any cached instance associated with this TableProxy"""
        return self._CACHE.delete(self) > 0

    def submit(self, fn: Callable, locktype: int, *args, **kw):
        if locktype == NOLOCK:
            return self._pool.instance.submit(
                _no_lock_impl, self.instance, fn, args, kw
            )
        elif locktype == READLOCK:
            return self._pool.instance.submit(
                _read_lock_impl, self.instance, fn, args, kw
            )
        elif locktype == WRITELOCK:
            return self._pool.instance.submit(
                _write_lock_impl, self.instance, fn, args, kw
            )
        else:
            return ValueError(f"Invalid locktype {locktype}")

    @property
    def executor_key(self):
        return self._executor_key

    def __enter__(self):
        return self

    def __exit__(self, evalue, etype, etraceback):
        pass


for method, locktype in METHOD_LOCKS:
    setattr(TableProxy, method, proxied_method_factory(method, locktype))


@normalize_token.register(TableProxy)
def _normalize_table_proxy(tp):
    return hash(tp)
