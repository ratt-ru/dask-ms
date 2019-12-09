# -*- coding: utf-8 -*-

from functools import reduce
from operator import mul
import pickle
import gc

import dask.array as da
import numpy as np
from numpy.testing import assert_array_almost_equal

from daskms import xds_from_ms
from daskms.optimisation import (inlined_array,
                                 cached_array,
                                 ArrayCache,
                                 Key,
                                 _key_cache,
                                 _array_cache_cache)


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

    assert len(C.__dask_graph__().layers) == 3

    D = inlined_array(C)

    assert len(D.__dask_graph__().layers) == 1
    assert len(dict(D.__dask_graph__())) == reduce(mul, D.numblocks, 1)


def test_cached_array(ms):
    ds = xds_from_ms(ms, group_cols=[], chunks={'row': 1, 'chan': 4})[0]

    data = ds.DATA.data
    cached_data = cached_array(data)
    assert_array_almost_equal(cached_data, data)

    # 2 x row blocks + row x chan x corr blocks
    assert len(_key_cache) == data.numblocks[0] * 2 + data.npartitions
    # rows, row runs and data array cache's
    assert len(_array_cache_cache) == 3

    # Pickling works
    pickled_data = pickle.loads(pickle.dumps(cached_data))
    assert_array_almost_equal(pickled_data, data)

    # Same underlying caching is re-used
    # 2 x row blocks + row x chan x corr blocks
    assert len(_key_cache) == data.numblocks[0] * 2 + data.npartitions
    # rows, row runs and data array cache's
    assert len(_array_cache_cache) == 3

    del pickled_data, cached_data, data, ds
    gc.collect()

    assert len(_key_cache) == 0
    assert len(_array_cache_cache) == 0
