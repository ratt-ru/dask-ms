# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask
from dask.array.core import normalize_chunks
import pytest

from xarrayms.dataset import dataset
from xarrayms.utils import (select_cols_str,
                            group_cols_str,
                            index_cols_str,
                            assert_liveness)


@pytest.mark.parametrize("group_cols", [
    ["FIELD_ID", "SCAN_NUMBER"]],
    ids=group_cols_str)
@pytest.mark.parametrize("index_cols", [
    ["TIME", "ANTENNA1", "ANTENNA2"]],
    ids=index_cols_str)
@pytest.mark.parametrize("select_cols", [
    ["TIME", "DATA"]],
    ids=select_cols_str)
@pytest.mark.parametrize("shapes", [
    {"row": 10, "chan": 16, "corr": 4}],
    ids=lambda s: "shapes=%s" % s)
@pytest.mark.parametrize("chunks", [
    {"row": 2},
    {"row": 3, "chan": 4, "corr": 1},
    {"row": 3, "chan": (4, 4, 4, 4), "corr": (2, 2)}],
    ids=lambda c: "chunks=%s" % c)
def test_dataset(ms, select_cols, group_cols, index_cols, shapes, chunks):
    datasets = dataset(ms, select_cols, group_cols, index_cols, chunks)
    # (1) Read-only TableProxy
    # (2) Read-only TAQL TableProxy
    assert_liveness(2, 1)

    rows = shapes['row']
    chans = shapes['chan']
    corrs = shapes['corr']

    # Expected output chunks
    echunks = {'row': normalize_chunks(chunks['row'],
                                       shape=(rows,))[0],
               'chan': normalize_chunks(chunks.get('chan', chans),
                                        shape=(chans,))[0],
               'corr': normalize_chunks(chunks.get('corr', corrs),
                                        shape=(corrs,))[0]}

    for ds in datasets:
        res = dask.compute(dict(ds.variables))[0]
        assert res['DATA'].shape[1:] == (chans, corrs)
        assert 'TIME' in res

        chunks = dict(ds.chunks)
        assert chunks["chan"] == echunks['chan']
        assert chunks["corr"] == echunks['corr']

        assert dict(ds.dims) == {
            'ROWID': ("row",),
            'TIME': ("row",),
            'DATA': ("row", "chan", "corr"),
        }

    del ds, datasets
    assert_liveness(0, 0)
