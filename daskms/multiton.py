from __future__ import annotations

from functools import partial
import inspect
from collections.abc import Callable
from typing import ClassVar, Hashable, Mapping, Sequence, Set, Type, Any, Dict, Tuple

from cacheout import Cache
from numpy import ndarray


FactoryFunctionT = Callable[..., Any]


FIVE_MINUTES = 5 * 60.0


def freeze(arg: Any) -> Any:
    """Recursively convert argument into an immutable representation"""
    if isinstance(arg, (bytes, str, FrozenKey)):
        # str and bytes are immutable sequences
        # return early to avoid tuplification
        # and freezing a FrozenKey is redundant
        return arg
    elif isinstance(arg, Sequence):
        return tuple(map(freeze, arg))
    elif isinstance(arg, Set):
        return frozenset(map(freeze, arg))
    elif isinstance(arg, Mapping):
        return frozenset((k, freeze(v)) for k, v in arg.items())
    elif isinstance(arg, ndarray):
        return (arg.data.tobytes(), arg.shape, arg.dtype.char)
    else:
        return arg


class FrozenKey(Hashable):
    """Converts args and kwargs into an immutable, hashable representation"""

    __slots__ = ("_frozen", "_hashvalue")
    _frozen: Tuple[Any, ...]
    _hashvalue: int

    def __init__(self, *args, **kw):
        self._frozen = freeze(args + (kw,))
        self._hashvalue = hash(self._frozen)

    @property
    def frozen(self) -> Tuple[Any, ...]:
        return self._frozen

    def __hash__(self) -> int:
        return self._hashvalue

    def __eq__(self, other) -> bool:
        if not isinstance(other, FrozenKey):
            return NotImplemented
        return self._hashvalue == other._hashvalue and self._frozen == other._frozen

    def __str__(self) -> str:
        return f"FrozenKey({self._hashvalue})"


def normalise_args(
    factory: Callable, args, kw
) -> Tuple[Tuple[Any, ...], Mapping[str, Any]]:
    """Move any arguments present in keywords into the arguments tuple.

    Args:
      factory: factory function
      args: positional arguments
      kw: keyword arguments

    Returns:
      tuple containing normalised positional arguments and keyword arguments
    """
    spec = inspect.getfullargspec(factory)
    args = list(args)

    for i, arg in enumerate(spec.args):
        if i < len(args):
            continue
        elif arg in kw:
            args.append(kw.pop(arg))
        elif spec.defaults and len(spec.args) - len(spec.defaults) <= i:
            default = spec.defaults[i - (len(spec.args) - len(spec.defaults))]
            args.append(default)

    return tuple(args), kw


def on_get_keep_alive(cls, key, value, exists):
    """Re-insert on get to update the TTL"""
    if exists:
        # Re-insert to update the TTL
        cls._CACHE.set(key, value)


def on_delete(key, value, cause):
    """Invoke any instance close methods on deletion"""
    if hasattr(value, "close") and callable(value.close):
        value.close()


class MultitonMetaclass(type):
    def __new__(
        meta_cls: Type["MultitonMetaclass"],
        name: str,
        bases: Tuple[type, ...],
        namespace: Dict[str, Any],
        cache_params: Dict[str, Any] | None = None,
    ) -> None:
        # Define instance methods for incorporation into the class namespace
        def __init__(
            self,
            factory: FactoryFunctionT,
            *args: Tuple[Any, ...],
            **kw: Dict[str, Any],
        ):
            self._factory = factory
            self._args, self._kw = normalise_args(factory, args, kw)
            self._key = FrozenKey(factory, *self._args, **self._kw)
            self.__post_init__()

        def __post_init__(self):
            pass

        def __reduce__(self) -> Tuple[Callable, Tuple]:
            return (type(self).reduce_from_args, (self._factory, self._args, self._kw))

        def __eq__(self, other: Any) -> bool:
            if not isinstance(other, type(self)):
                return NotImplemented
            return self._key == other._key

        def __hash__(self) -> int:
            return hash(self._key)

        @property
        def instance(self) -> Any:
            return self._CACHE.get(self, self._create_instance)

        def release(self) -> bool:
            return self._CACHE.delete(self) > 0

        namespace.setdefault("__init__", __init__)
        namespace.setdefault("__post_init__", __post_init__)
        namespace.setdefault("__reduce__", __reduce__)
        namespace.setdefault("__hash__", __hash__)
        namespace.setdefault("__eq__", __eq__)
        namespace.setdefault("instance", instance)
        namespace.setdefault("release", release)

        # Configure the class to be slotted
        REQUIRED_SLOTS = {"_factory", "_args", "_kw", "_key"}
        slots = set(namespace.get("__slots__", REQUIRED_SLOTS))
        namespace["__slots__"] = tuple(sorted(slots | REQUIRED_SLOTS))

        # Create the class
        cls = super().__new__(meta_cls, name, bases, namespace)

        # Create and assign the class cache
        cache_params = {} if cache_params is None else cache_params
        cache_cls = cache_params.pop("cls", Cache)
        assert issubclass(cache_cls, Cache)
        cache_params.setdefault("ttl", FIVE_MINUTES)
        cache_params.setdefault("maxsize", 100)
        cache_params.setdefault("on_get", partial(on_get_keep_alive, cls))
        cache_params.setdefault("on_delete", on_delete)

        cls._CACHE = cache_cls(**cache_params)

        # Define class and static methods for inclusion on the class
        @classmethod
        def reduce_from_args(cls, factory, args, kw):
            return cls(factory, *args, **kw)

        @staticmethod
        def _create_instance(obj) -> Any:
            return obj._factory(*obj._args, **obj._kw)

        cls.reduce_from_args = classmethod(reduce_from_args)
        cls._create_instance = staticmethod(_create_instance)

        # Add type annotations
        cls.__annotations__["_factory"] = FactoryFunctionT
        cls.__annotations__["_args"] = Tuple[Any, ...]
        cls.__annotations__["_kw"] = Mapping[str, Any]
        cls.__annotations__["_key"] = FrozenKey
        cls.__annotations__["_CACHE"] = ClassVar[cache_cls]

        return cls
