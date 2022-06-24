# -*- coding: utf-8 -*-

import pickle
import gc

import dask.array as da
from dask.core import flatten
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from daskms import xds_from_ms
from daskms.optimisation import (inlined_array,
                                 cached_array)



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


def test_blah_cached():
    A = da.arange(5, chunks=1)

    def f(a):
        print(f"Got {a}")
        return a

    B = da.blockwise(f, 'a', A, 'a', meta=A._meta)
    B = cached_array(inlined_array(B))

    C = da.ones(20, chunks=5)[None, :] * B[:, None]
    C.compute()


def test_cached_array(ms):
    ds = xds_from_ms(ms, group_cols=[], chunks={'row': 1, 'chan': 4})[0]

    data = ds.DATA.data
    cached_data = cached_array(data)
    assert_array_almost_equal(cached_data, data)

    # Pickling works
    pickled_data = pickle.loads(pickle.dumps(cached_data))
    assert_array_almost_equal(pickled_data, data)

    del pickled_data, cached_data, data, ds
    gc.collect()
