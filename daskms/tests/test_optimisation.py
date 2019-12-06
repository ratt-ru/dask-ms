# -*- coding: utf-8 -*-

import gc

from numpy.testing import assert_array_almost_equal

from daskms import xds_from_ms
from daskms.optimisation import cached_array, _key_cache, _array_cache_cache


def test_cached_array(ms):
    ds = xds_from_ms(ms, group_cols=[], chunks={'row': 1, 'chan': 4})[0]

    data = ds.DATA.data
    assert_array_almost_equal(cached_array(data), data)

    assert len(_key_cache) > 0
    assert len(_array_cache_cache) > 0

    del data, ds
    gc.collect()

    assert len(_key_cache) == 0
    assert len(_array_cache_cache) == 0
