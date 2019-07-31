# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask
from dask.array.core import normalize_chunks
from numpy.testing import assert_array_equal
import pyrap.tables as pt
import pytest

from xarrayms.dataset import dataset, write_columns
from xarrayms.utils import (select_cols_str,
                            group_cols_str,
                            index_cols_str,
                            assert_liveness)


@pytest.mark.parametrize("group_cols", [
    ["FIELD_ID", "SCAN_NUMBER"],
    []],
    ids=group_cols_str)
@pytest.mark.parametrize("index_cols", [
    ["TIME", "ANTENNA1", "ANTENNA2"]],
    ids=index_cols_str)
@pytest.mark.parametrize("select_cols", [
    ["STATE_ID", "TIME", "DATA"]],
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

    chans = shapes['chan']
    corrs = shapes['corr']

    # Expected output chunks
    echunks = {'chan': normalize_chunks(chunks.get('chan', chans),
                                        shape=(chans,))[0],
               'corr': normalize_chunks(chunks.get('corr', corrs),
                                        shape=(corrs,))[0]}

    for ds in datasets:
        res = dask.compute(dict(ds.variables))[0]
        assert res['DATA'].shape[1:] == (chans, corrs)
        assert 'STATE_ID' in res
        assert 'TIME' in res

        chunks = dict(ds.chunks)
        assert chunks["chan"] == echunks['chan']
        assert chunks["corr"] == echunks['corr']

        assert dict(ds.dims) == {
            'STATE_ID': ("row",),
            'ROWID': ("row",),
            'TIME': ("row",),
            'DATA': ("row", "chan", "corr"),
        }

    del ds, datasets
    assert_liveness(0, 0)


@pytest.mark.parametrize("group_cols", [
    ["FIELD_ID", "SCAN_NUMBER"],
    []],
    ids=group_cols_str)
@pytest.mark.parametrize("index_cols", [
    ["TIME", "ANTENNA1", "ANTENNA2"]],
    ids=index_cols_str)
@pytest.mark.parametrize("select_cols", [
    ["STATE_ID", "TIME", "DATA"]],
    ids=select_cols_str)
@pytest.mark.parametrize("shapes", [
    {"row": 10, "chan": 16, "corr": 4}],
    ids=lambda s: "shapes=%s" % s)
@pytest.mark.parametrize("chunks", [
    {"row": 2},
    {"row": 3, "chan": 4, "corr": 1},
    {"row": 3, "chan": (4, 4, 4, 4), "corr": (2, 2)}],
    ids=lambda c: "chunks=%s" % c)
def test_dataset_writes(ms, select_cols,
                        group_cols, index_cols,
                        shapes, chunks):
    datasets = dataset(ms, select_cols, group_cols, index_cols, chunks)
    assert_liveness(2, 1)

    # Test writes
    writes = []

    # Obtain original  STATE_ID
    with pt.table(ms, ack=False, readonly=True) as T:
        state_id = T.getcol("STATE_ID")

    # Create write operations and execute them
    for i, ds in enumerate(datasets):
        new_ds = ds.assign(STATE_ID=(ds.STATE_ID + 1, ("row",)))
        writes.append(write_columns(ms, new_ds, ["STATE_ID"]))

    dask.compute(writes)

    # NOTE(sjperkins)
    # Interesting behaviour here. If these objects are not
    # cleared up at this point, attempts to re-open the table below
    # can fail, reproducing https://github.com/ska-sa/xarray-ms/issues/26
    # Adding auto-locking to the table opening command seems to fix
    # this somehow
    del ds, new_ds, datasets, writes
    assert_liveness(0, 0)

    # Compare against expected result and restore STATE_ID
    with pt.table(ms, ack=False, readonly=False) as T:
        assert_array_equal(state_id + 1, T.getcol("STATE_ID"))
        T.putcol("STATE_ID", state_id)


# Even though we ask for two rows, we get single rows out
# due to the "__row__" in group_col
@pytest.mark.parametrize("chunks", [{"row": 2}], ids=lambda c: str(c))
def test_row_grouping(spw_table, spw_chan_freqs, chunks):
    datasets = dataset(spw_table, [], ["__row__"], [], chunks)

    assert_liveness(2, 1)

    assert len(datasets) == len(spw_chan_freqs)

    for i, chan_freq in enumerate(spw_chan_freqs):
        assert_array_equal(datasets[i].CHAN_FREQ, chan_freq)

    del datasets
    assert_liveness(0, 0)
