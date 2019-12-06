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
    def __call__(cls, key):
        with _key_cache_lock:
            try:
                return _key_cache[key]
            except KeyError:
                _key_cache[key] = instance = type.__call__(cls, key)
                return instance


class Key(metaclass=KeyMetaClass):
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


def get_cache_entry(cache, key, task):
    with cache.lock:
        try:
            return cache.cache[key]
        except KeyError:
            cache.cache[key] = value = _execute_task(task, {})
            return value


_array_cache_cache = WeakValueDictionary()
_array_cache_lock = Lock()


class ArrayCacheMetaClass(type):
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
    def __init__(self, token):
        self.token = token
        self.cache = WeakKeyDictionary()
        self.lock = Lock()

    def __reduce__(self):
        return (ArrayCache, (self.token,))

    def __repr__(self):
        return "ArrayCache[%s]" % self.token


def cached_array(array):
    dsk = dict(array.__dask_graph__())
    keys = set(flatten(array.__dask_keys__()))

    # Inline + cull everything except the current array name
    inline_keys = set(dsk.keys() - keys)
    dsk2 = inline(dsk, inline_keys, inline_constants=True)
    dsk3, _ = cull(dsk2, keys)

    cache = ArrayCache(uuid.uuid4().hex)

    for k in keys:
        dsk3[k] = (get_cache_entry, cache, Key(k), dsk3.pop(k))

    graph = HighLevelGraph.from_collections(array.name, dsk3, [])

    return da.Array(graph, array.name, array.chunks, array.dtype)


def inlined_array(a, inline_arrays=None):
    akeys = set(flatten(a.__dask_keys__()))

    # Inline arrays
    if inline_arrays is None:
        inline_keys = set(a.__dask_graph__().keys()) - akeys
    elif isinstance(inline_arrays, da.Array):
        inline_keys = set(flatten(inline_arrays.__dask_keys__()))
    elif isinstance(inline_arrays, (tuple, list)):
        inline_keys = set(flatten([a.__dask_keys__() for a in inline_arrays]))
    else:
        raise TypeError("Invalid inline_arrays")

    dsk2 = inline(a.__dask_graph__(), keys=inline_keys, inline_constants=True)

    # Remove everything except keys of A
    dsk3, _ = cull(dsk2, akeys)

    return da.Array(dsk3, a.name, a.chunks, a.dtype)
