# -*- coding: utf-8 -*-

from threading import Lock
from itertools import product

import dask
import dask.array as da
from dask.core import flatten, get
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import cull, inline
import numpy as np




class GraphCache:
    def __init__(self, collection):
        self.collection = collection
        self.lock = Lock()
        self.cache = {}

    def __reduce__(self):
        return (GraphCache, (self.collection,))

    def __call__(self, block_id):
        try:
            dsk = self.dsk
        except AttributeError:
            with self.lock:
                try:
                    dsk = self.dsk
                except AttributeError:
                    self.dsk = dsk = dict(self.collection.__dask_graph__())

        key = (self.collection.name,) + block_id

        with self.lock:
            try:
                return self.cache[key]
            except KeyError:
                return get(dsk, key, self.cache)


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
    """
    assert isinstance(array, da.Array)
    name = f"block-id-{array.name}"
    dsk = {(name,) + block_id: block_id
           for block_id in product(*(range(len(c)) for c in array.chunks))}
    assert all(all(isinstance(e, int) for e in bid) for bid in dsk.values())
    block_id_array = da.Array(dsk, name,
                              chunks=tuple((1,)*len(c) for c in array.chunks),
                              dtype=np.object_)

    assert array.ndim == block_id_array.ndim
    idx = list(range(array.ndim))
    adjust_chunks = dict(zip(idx, array.chunks))
    cache = GraphCache(array)
    token = f"GraphCache-{dask.base.tokenize(cache, block_id_array)}"

    return da.blockwise(cache, idx,
                        block_id_array, idx,
                        adjust_chunks=adjust_chunks,
                        meta=array._meta,
                        name=token)

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

    inline_names = set(a.name for a in inline_arrays)
    layers = agraph.layers.copy()
    deps = {k: v.copy() for k, v in agraph.dependencies.items()}
    # We want to inline layers that depend on the inlined arrays
    inline_layers = set(k for k, v in deps.items()
                        if len(inline_names.intersection(v)) > 0)

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
