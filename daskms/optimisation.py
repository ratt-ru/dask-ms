# -*- coding: utf-8 -*-

from threading import Lock
import uuid
from weakref import WeakValueDictionary, WeakKeyDictionary

import dask.array as da
from dask.core import flatten, _execute_task
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import cull, inline


_key_cache = WeakValueDictionary()
_key_cache_lock = Lock()


class KeyMetaClass(type):
    """
    Ensures that Key identities are the same,
    given the same constructor arguments
    """
    def __call__(cls, key):
        with _key_cache_lock:
            try:
                return _key_cache[key]
            except KeyError:
                _key_cache[key] = instance = type.__call__(cls, key)
                return instance


class Key(metaclass=KeyMetaClass):
    """
    Suitable for storing a tuple
    (or other dask key type) in a WeakKeyDictionary.
    Uniques of key identity guaranteed by KeyMetaClass
    """
    __slots__ = ("key", "__weakref__")

    def __init__(self, key):
        self.key = key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        return "Key%s" % (self.key,)

    def __reduce__(self):
        return (Key, (self.key,))

    __str__ = __repr__


def cache_entry(cache, key, task):
    with cache.lock:
        try:
            return cache.cache[key]
        except KeyError:
            cache.cache[key] = value = _execute_task(task, {})
            return value


_array_cache_cache = WeakValueDictionary()
_array_cache_lock = Lock()


class ArrayCacheMetaClass(type):
    """
    Ensures that Array Cache identities are the same,
    given the same constructor arguments
    """
    def __call__(cls, token):
        key = (cls, token)

        with _array_cache_lock:
            try:
                return _array_cache_cache[key]
            except KeyError:
                instance = type.__call__(cls, token)
                _array_cache_cache[key] = instance
                return instance


class ArrayCache(metaclass=ArrayCacheMetaClass):
    """
    Thread-safe array data cache. token makes this picklable.

    Cached on a WeakKeyDictionary with ``Key`` objects.
    """
    def __init__(self, token):
        self.token = token
        self.cache = WeakKeyDictionary()
        self.lock = Lock()

    def __reduce__(self):
        return (ArrayCache, (self.token,))

    def __repr__(self):
        return "ArrayCache[%s]" % self.token


def cached_array(array, token=None):
    """
    Return a new array that functionally has the same values as array,
    but flattens the underlying graph and introduces a cache lookup
    when the individual array chunks are accessed.

    Useful for caching data that can fit in-memory for the duration
    of the graph's execution.

    Parameters
    ----------
    array : :class:`dask.array.Array`
        dask array to cache.
    token : optional, str
        A unique token for identifying the internal cache.
        If None, it will be automatically generated.
    """
    dsk = dict(array.__dask_graph__())
    keys = set(flatten(array.__dask_keys__()))

    if token is None:
        token = uuid.uuid4().hex

    # Inline + cull everything except the current array
    inline_keys = set(dsk.keys() - keys)
    dsk2 = inline(dsk, inline_keys, inline_constants=True)
    dsk3, _ = cull(dsk2, keys)

    # Create a cache used to store array values
    cache = ArrayCache(token)

    assert len(dsk3) == len(keys)

    for k in keys:
        dsk3[k] = (cache_entry, cache, Key(k), dsk3.pop(k))

    graph = HighLevelGraph.from_collections(array.name, dsk3, [])

    return da.Array(graph, array.name, array.chunks, array.dtype)


def inlined_array(a, inline_arrays=None):
    """ Flatten underlying graph """
    agraph = a.__dask_graph__()
    akeys = set(flatten(a.__dask_keys__()))

    # Inline everything except the output keys
    if inline_arrays is None:
        inline_keys = set(agraph.keys()) - akeys
        dsk2 = inline(agraph, keys=inline_keys, inline_constants=True)
        dsk3, _ = cull(dsk2, akeys)

        graph = HighLevelGraph.from_collections(a.name, dsk3, [])
        return da.Array(graph, a.name, a.chunks, dtype=a.dtype)

    # We're given specific arrays to inline, promote to list
    if isinstance(inline_arrays, da.Array):
        inline_arrays = [inline_arrays]
    elif isinstance(inline_arrays, tuple):
        inline_arrays = list(inline_arrays)

    if not isinstance(inline_arrays, list):
        raise TypeError("Invalid inline_arrays, must be "
                        "(None, list, tuple, dask.array.Array)")

    layers = agraph.layers.copy()
    deps = {k: v.copy() for k, v in agraph.dependencies.items()}
    inline_keys = set()
    dsk = dict(layers[a.name])

    # Inline specified arrays
    for array in inline_arrays:
        # Remove array from layers and dependencies
        try:
            dsk.update(layers.pop(array.name))
            del deps[array.name]
            deps[a.name].discard(array.name)
        except KeyError:
            raise ValueError(f"{array.name} is not a "
                             f"valid dependency of {a.name}")

        # Record keys to inline
        inline_keys.update(flatten(array.__dask_keys__()))

    dsk2 = inline(dsk, keys=inline_keys, inline_constants=True)
    dsk3, _ = cull(dsk2, akeys)

    layers[a.name] = dsk3
    graph = HighLevelGraph(layers, deps)

    return da.Array(graph, a.name, a.chunks, a.dtype)
