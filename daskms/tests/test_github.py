import os

import dask
import dask.array as da
from daskms import xds_from_ms

import pytest


def test_github_98():
    ms = "/home/sperkins/data/AF0236_spw01.ms/"

    if not os.path.exists(ms):
        pytest.skip("AF0236_spw01.ms on which this "
                    "test depends is not present")

    datasets = xds_from_ms(ms, columns=['DATA', 'ANTENNA1', 'ANTENNA2'],
                           group_cols=['DATA_DESC_ID'],
                           taql_where='ANTENNA1 == 5 || ANTENNA2 == 5')

    assert len(datasets) == 2
    assert datasets[0].DATA_DESC_ID == 0
    assert datasets[1].DATA_DESC_ID == 1

    for ds in datasets:
        expr = da.logical_or(ds.ANTENNA1.data == 5, ds.ANTENNA2.data == 5)
        expr, equal = dask.compute(expr, da.all(expr))
        assert equal.item() is True
        assert len(expr) > 0
