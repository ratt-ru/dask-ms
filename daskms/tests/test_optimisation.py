# -*- coding: utf-8 -*-

import gc
import pickle

import dask.array as da
import numpy as np
import pytest
from dask.core import flatten
from numpy.testing import assert_array_almost_equal, assert_array_equal

from daskms import xds_from_ms
from daskms.optimisation import (
    _DASK_HAS_LEGACY_TASKS,
    ArrayCache,
    Key,
    _array_cache_cache,
    _CachedCompute,
    _key_cache,
    cached_array,
    inlined_array,
)


def test_optimisation_identity():
    # Test identity
    assert Key((0, 1, 2)) is Key((0, 1, 2))
    assert ArrayCache(1) is ArrayCache(1)

    # Test pickling
    assert pickle.loads(pickle.dumps(Key((0, 1, 2)))) is Key((0, 1, 2))
    assert pickle.loads(pickle.dumps(ArrayCache(1))) is ArrayCache(1)


def test_inlined_array():
    A = da.ones((10, 10), chunks=(2, 2), dtype=np.float64)
    B = da.full((10, 10), np.float64(2), chunks=(2, 2))
    C = A + B
    E = C + 1

    D = inlined_array(C)
    assert len(C.__dask_graph__().layers) == 3
    assert D.name == C.name
    assert D.name in D.__dask_graph__().layers
    assert A.name not in D.__dask_graph__().layers
    assert B.name not in D.__dask_graph__().layers
    graph_keys = set(flatten(D.__dask_graph__().keys()))
    assert graph_keys == set(flatten(D.__dask_keys__()))
    assert_array_equal(D, C)

    D = inlined_array(C, [A, B])
    assert len(D.__dask_graph__().layers) == 1
    assert D.name == C.name
    assert D.name in D.__dask_graph__().layers
    assert A.name not in D.__dask_graph__().layers
    assert B.name not in D.__dask_graph__().layers
    graph_keys = set(flatten(D.__dask_graph__().keys()))
    assert graph_keys == set(flatten(D.__dask_keys__()))
    assert_array_equal(D, C)

    D = inlined_array(C, [A])
    assert len(D.__dask_graph__().layers) == 2
    assert D.name == C.name
    assert D.name in D.__dask_graph__().layers
    assert A.name not in D.__dask_graph__().layers
    assert B.name in D.__dask_graph__().layers
    graph_keys = set(flatten(D.__dask_graph__().keys()))
    assert graph_keys == set(flatten([a.__dask_keys__() for a in [D, B]]))
    assert_array_equal(D, C)

    D = inlined_array(C, [B])
    assert len(D.__dask_graph__().layers) == 2
    assert D.name == C.name
    assert D.name in D.__dask_graph__().layers
    assert A.name in D.__dask_graph__().layers
    assert B.name not in D.__dask_graph__().layers
    graph_keys = set(flatten(D.__dask_graph__().keys()))
    assert graph_keys == set(flatten([a.__dask_keys__() for a in [D, A]]))
    assert_array_equal(D, C)

    D = inlined_array(E, [A])
    assert len(D.__dask_graph__().layers) == 3
    assert D.name == E.name
    assert D.name in D.__dask_graph__().layers
    assert B.name in D.__dask_graph__().layers
    assert A.name not in D.__dask_graph__().layers
    assert C.name in D.__dask_graph__().layers
    graph_keys = set(flatten(D.__dask_graph__().keys()))
    assert graph_keys == set(flatten([a.__dask_keys__() for a in [D, B, C]]))
    assert_array_equal(D, E)


def test_cached_array(ms):
    ds = xds_from_ms(ms, group_cols=[], chunks={"row": 1, "chan": 4})[0]

    data = ds.DATA.data
    cached_data = cached_array(data)
    assert_array_almost_equal(cached_data, data)

    # row blocks + row x chan x corr blocks
    assert len(_key_cache) == data.numblocks[0] + data.npartitions
    # rows and data array cache's
    assert len(_array_cache_cache) == 2

    # Pickling works
    pickled_data = pickle.loads(pickle.dumps(cached_data))
    assert_array_almost_equal(pickled_data, data)

    # Same underlying caching is re-used
    # row blocks + row x chan x corr blocks
    assert len(_key_cache) == data.numblocks[0] + data.npartitions
    # rows and data array cache's
    assert len(_array_cache_cache) == 2

    del pickled_data, cached_data, data, ds
    gc.collect()

    assert len(_key_cache) == 0
    assert len(_array_cache_cache) == 0


@pytest.mark.parametrize("token", ["0xdeadbeaf", None])
def test_cached_data_token(token):
    zeros = da.zeros(1000, chunks=100)
    carray = cached_array(zeros, token)

    dsk = dict(carray.__dask_graph__())
    k, v = dsk.popitem()
    # Old dask: v = (cache_entry, cache, Key(k), ...)  → v[1] is cache
    # New dask: v = (_CachedCompute(cache, k, task),)  → v[0]._cache is cache
    if _DASK_HAS_LEGACY_TASKS:
        cache = v[1]
    else:
        cache = v[0]._cache

    if token is None:
        assert cache.token is not None
    else:
        assert cache.token == token
