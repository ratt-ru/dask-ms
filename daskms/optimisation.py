# -*- coding: utf-8 -*-

from itertools import product

import dask.array as da
from dask.base import tokenize
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
from dask.local import get_sync
from dask.optimization import cull, inline

from daskms.multiton import MultitonMetaclass


class _ArrayRef:
    """Wraps a dask Array with a stable identity token. Equality and hash
    are keyed solely on the token so that the array itself is opaque to
    the :class:`MultitonMetaclass` :class:`FrozenKey` machinery."""

    __slots__ = ("array", "token")

    def __init__(self, array, token):
        self.array = array
        self.token = token

    def __hash__(self):
        return hash(self.token)

    def __eq__(self, other):
        if not isinstance(other, _ArrayRef):
            return NotImplemented
        return self.token == other.token

    def __reduce__(self):
        return (type(self), (self.array, self.token))


def _materialise(ref):
    graph = dict(ref.array.__dask_graph__())
    keys = list(flatten(ref.array.__dask_keys__()))
    values = get_sync(graph, keys)
    return {key[1:]: val for key, val in zip(keys, values)}


class _CachedArray(metaclass=MultitonMetaclass):
    """Multiton holding the per-block results of a materialised dask Array,
    keyed on ``(_materialise, _ArrayRef.token)``."""


def _read_cached_block(holder, block_idx):
    return holder.instance[block_idx]


def cached_array(array, token=None):
    """
    Return a new array that functionally has the same values as ``array``.

    On first block access within a process, the entire input array is
    materialised (using the synchronous scheduler) and each block's
    computed value is stored in a process-local cache keyed on ``token``.
    Subsequent block accesses fetch pre-computed blocks from that cache.

    Cache entries are released once no dask graph references the cached
    array any longer. Call sites that pass the same ``token`` share a
    single cached result.

    Parameters
    ----------
    array : :class:`dask.array.Array`
        dask array to cache.
    token : optional, str
        Identity token for the cached result. If ``None``, derived from
        ``dask.base.tokenize(array)``.
    """
    if token is None:
        token = tokenize(array)

    holder = _CachedArray(_materialise, _ArrayRef(array, token))
    name = f"cached-array-{token}"

    dsk = {
        (name, *idx): (_read_cached_block, holder, idx)
        for idx in product(*(range(len(c)) for c in array.chunks))
    }

    graph = HighLevelGraph.from_collections(name, dsk, [])
    return da.Array(graph, name, chunks=array.chunks, dtype=array.dtype)


def inlined_array(a, inline_arrays=None):
    """Flatten underlying graph"""
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
        raise TypeError(
            "Invalid inline_arrays, must be " "(None, list, tuple, dask.array.Array)"
        )

    inline_names = set(a.name for a in inline_arrays)
    layers = agraph.layers.copy()
    deps = {k: v.copy() for k, v in agraph.dependencies.items()}
    # We want to inline layers that depend on the inlined arrays
    inline_layers = set(
        k for k, v in deps.items() if len(inline_names.intersection(v)) > 0
    )

    for layer_name in inline_layers:
        dsk = dict(layers[layer_name])
        layer_keys = set(dsk.keys())
        inline_keys = set()

        for array in inline_arrays:
            dsk.update(layers[array.name])
            deps.pop(array.name, None)
            deps[layer_name].discard(array.name)
            inline_keys.update(layers[array.name].keys())

        dsk2 = inline(dsk, keys=inline_keys, inline_constants=True)
        layers[layer_name], _ = cull(dsk2, layer_keys)

    # Remove layers containing the inlined arrays
    for inline_name in inline_names:
        layers.pop(inline_name)

    return da.Array(HighLevelGraph(layers, deps), a.name, a.chunks, a.dtype)
