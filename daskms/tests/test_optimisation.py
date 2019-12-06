# -*- coding: utf-8 -*-

from numpy.testing import assert_array_almost_equal

from daskms import xds_from_ms
from daskms.optimisation import cached_array


def test_cached_array(ms):
    ds = xds_from_ms(ms, group_cols=[], chunks={'row': 1})[0]

    from pprint import pprint
    from daskms.optimisation import _key_cache, _array_cache_cache

    data = ds.DATA.data
    print('='*80)
    pprint(dict(data.__dask_graph__()))

    cached_data = cached_array(data)

    print('='*80)
    pprint(dict(cached_data.__dask_graph__()))

    print(len(_key_cache), len(_array_cache_cache))

    assert_array_almost_equal(cached_data, data)

    pprint(dict(_key_cache))
    pprint(dict(_array_cache_cache))

    del cached_data, data
    import gc
    gc.collect()

    pprint(dict(_key_cache))
    pprint(dict(_array_cache_cache))
