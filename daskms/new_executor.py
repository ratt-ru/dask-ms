from __future__ import annotations

from typing import Callable, ClassVar, Mapping, Any, Tuple

from cacheout import LRUCache

from daskms.multiton import FactoryFunctionT, FrozenKey, normalise_args


def on_get_keep_alive(key, value, exists):
    """Re-insert on get to update the TTL"""
    if exists:
        # Re-insert to update the TTL
        Executor._CACHE.set(key, value)


def on_delete(key, value, cause):
    """Invoke any instance close methods on deletion"""
    if hasattr(value, "close") and callable(value.close):
        value.close()


class Executor:
    """Hashable and pickleable factory class
    for creating and caching an object instance"""

    _CACHE: ClassVar[LRUCache] = LRUCache(
        maxsize=100, ttl=5 * 60, on_get=on_get_keep_alive, on_delete=on_delete
    )
    _factory: FactoryFunctionT
    _args: Tuple[Any, ...]
    _kw: Mapping[str, Any]
    _key: FrozenKey

    def __init__(
        self,
        factory: FactoryFunctionT,
        *args: Any,
        **kw: Any,
    ):
        self._factory = factory
        self._args, self._kw = normalise_args(factory, args, kw)
        self._key = FrozenKey(factory, *self._args, **self._kw)

    @staticmethod
    def from_reduce_args(
        factory: FactoryFunctionT, args: Tuple[Any, ...], kw: Mapping[str, Any]
    ) -> Executor:
        return Executor(factory, *args, **kw)

    def __reduce__(
        self,
    ) -> Tuple[Callable, Tuple[Callable, Tuple[Any, ...], Mapping[str, Any]]]:
        return (self.from_reduce_args, (self._factory, self._args, self._kw))

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Executor):
            return NotImplemented
        return self._key == other._key

    @staticmethod
    def _create_instance(self) -> Executor:
        return self._factory(*self._args, **self._kw)

    @property
    def instance(self) -> Executor:
        """Create the object instance represented by the Executor,
        or retrieved the cache instance"""
        return self._CACHE.get(self, self._create_instance)

    def release(self) -> bool:
        """Evict any cached instance associated with this Executor"""
        return self._CACHE.delete(self) > 0
