# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import dask
import dask.array as da
from dask.array.core import normalize_chunks
import numpy as np
from numpy.testing import assert_array_equal
import pyrap.tables as pt
import pytest

from xarrayms.dataset import dataset, write_datasets, Dataset
from xarrayms.table_proxy import TableProxy
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
    """ Test dataset creation """
    datasets = dataset(ms, select_cols, group_cols, index_cols, chunks=chunks)
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
        compute_dict = {}

        for k, (dims, var, attrs) in ds.variables.items():
            compute_dict[k] = var

            if k in select_cols:
                assert "__coldesc__" in attrs

        res = dask.compute(compute_dict)[0]

        assert res['DATA'].shape[1:] == (chans, corrs)
        assert 'STATE_ID' in res
        assert 'TIME' in res

        chunks = ds.chunks
        assert chunks["chan"] == echunks['chan']
        assert chunks["corr"] == echunks['corr']

        dims = ds.dims
        dims.pop('row')  # row changes
        assert dims == {"chan": shapes['chan'],
                        "corr": shapes['corr']}

    del ds, datasets, compute_dict, var
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
    """ Test dataset writes """

    # Get original STATE_ID and DATA
    with pt.table(ms, ack=False, readonly=True, lockoptions='auto') as T:
        original_state_id = T.getcol("STATE_ID")
        original_data = T.getcol("DATA")

    try:
        datasets = dataset(ms, select_cols, group_cols,
                           index_cols, chunks=chunks)
        assert_liveness(2, 1)

        # Test writes
        writes = []

        # Create write operations and execute them
        for i, ds in enumerate(datasets):
            new_ds = ds.assign(STATE_ID=ds.STATE_ID + 1, DATA=ds.DATA + 1)
            writes.append(write_datasets(ms, new_ds, ["STATE_ID", "DATA"]))

        dask.compute(writes)

        # NOTE(sjperkins)
        # Interesting behaviour here. If these objects are not
        # cleared up at this point, attempts to re-open the table below
        # can fail, reproducing https://github.com/ska-sa/xarray-ms/issues/26
        # Adding auto-locking to the table opening command seems to fix
        # this somehow
        del ds, new_ds, datasets, writes
        assert_liveness(0, 0)

    finally:
        # Restore original STATE_ID
        with pt.table(ms, ack=False, readonly=False, lockoptions='auto') as T:
            state_id = T.getcol("STATE_ID")
            data = T.getcol("DATA")
            T.putcol("STATE_ID", original_state_id)
            T.putcol("DATA", original_data)

    # Compare against expected result
    assert_array_equal(original_state_id + 1, state_id)
    assert_array_equal(original_data + 1, data)


# Even though we ask for two rows, we get single rows out
# due to the "__row__" in group_col
@pytest.mark.parametrize("chunks", [{"row": 2}], ids=lambda c: str(c))
def test_row_grouping(spw_table, spw_chan_freqs, chunks):
    """ Test grouping on single rows """
    datasets = dataset(spw_table, [], ["__row__"], [], chunks=chunks)

    assert_liveness(2, 1)

    assert len(datasets) == len(spw_chan_freqs)

    for i, chan_freq in enumerate(spw_chan_freqs):
        assert_array_equal(datasets[i].CHAN_FREQ, chan_freq)
        assert_array_equal(datasets[i].NUM_CHAN, chan_freq.shape[0])

    del datasets
    assert_liveness(0, 0)


def test_antenna_table_string_names(ant_table, wsrt_antenna_positions):
    ds = dataset(ant_table, [], [], None)
    assert len(ds) == 1
    ds = ds[0]

    names = ["ANTENNA-%d" % i for i in range(wsrt_antenna_positions.shape[0])]

    assert_array_equal(ds.POSITION, wsrt_antenna_positions)
    assert_array_equal(ds.NAME, names)

    names = ds.NAME.compute()

    # Test that writing back string ndarrays work as
    # they must be converted from ndarrays to lists
    # of strings internally
    write_cols = set(ds.variables.keys()) - set(["ROWID"])
    writes = write_datasets(ant_table, ds, write_cols)

    dask.compute(writes)


def test_dataset_assign(ms):
    """ Test dataset assignment """
    datasets = dataset(ms, [], [], [])

    assert len(datasets) == 1
    ds = datasets[0]

    # Assign on an existing column is easier because we can
    # infer the dimension schema from it
    nds = ds.assign(TIME=ds.TIME + 1)
    assert ds.DATA is nds.DATA
    assert ds.TIME is not nds.TIME
    assert_array_equal(nds.TIME, ds.TIME + 1)

    # This doesn't work for new columns
    with pytest.raises(ValueError, match="Couldn't find existing dimension"):
        ds.assign(ANTENNA3=ds.ANTENNA1 + 3)

    # We have to explicitly supply a dimension schema
    nds = ds.assign(ANTENNA3=(("row",), ds.ANTENNA1 + 3))
    assert_array_equal(ds.ANTENNA1 + 3, nds.ANTENNA3)

    dims = ds.dims
    chunks = ds.chunks

    with pytest.raises(ValueError, match="size 9 for dimension 'row'"):
        array = da.zeros(dims['row'] - 1, chunks['row'])
        nds = ds.assign(ANTENNA4=(("row",),  array))
        nds.dims

    assert chunks['row'] == (10,)

    with pytest.raises(ValueError, match=r"chunking \(4, 4, 2\) for dim"):
        array = da.zeros(dims['row'], chunks=4)
        nds = ds.assign(ANTENNA4=(("row",),  array))
        nds.chunks

    del datasets, ds, nds
    assert_liveness(0, 0)


def test_dataset_table_schemas(ms):
    """ Test that we can pass table schemas """
    data_dims = ("mychan", "mycorr")
    table_schema = ["MS", {"DATA": {'dask': {"dims": data_dims}}}]
    datasets = dataset(ms, [], [], [], table_schema=table_schema)
    assert datasets[0].variables["DATA"].dims == ("row", ) + data_dims


@pytest.mark.parametrize("dtype", [
    np.complex64,
    np.complex128,
    np.float32,
    np.float64,
    np.int16,
    np.int32,
    np.uint32,
    np.bool,
    pytest.param(np.object,
                 marks=pytest.mark.xfail(reason="putcol can't handle "
                                                "lists of ints")),
    pytest.param(np.uint16,
                 marks=pytest.mark.xfail(reason="RuntimeError: RecordRep::"
                                                "createDataField: unknown data"
                                                " type 17")),
    pytest.param(np.uint8,
                 marks=pytest.mark.xfail(reason="Creates uint16 column")),
])
def test_dataset_add_column(ms, dtype):
    datasets = dataset(ms, [], [], [])
    assert len(datasets) == 1
    ds = datasets[0]

    bitflag = da.zeros_like(ds.DATA, dtype=dtype)
    nds = ds.assign(BITFLAG=(("row", "chan", "corr"), bitflag))
    writes = write_datasets(ms, nds, ["BITFLAG"])

    dask.compute(writes)

    del datasets, ds, writes, nds
    assert_liveness(0, 0)

    with pt.table(ms, readonly=False, ack=False, lockoptions='auto') as T:
        bf = T.getcol("BITFLAG")
        assert bf.dtype == dtype


def test_dataset_add_string_column(ms):
    datasets = dataset(ms, [], [], [])
    assert len(datasets) == 1
    ds = datasets[0]
    dims = ds.dims

    name_list = ["BOB"] * dims['row']
    names = np.asarray(name_list, dtype=np.object)
    names = da.from_array(names, chunks=ds.TIME.chunks)

    nds = ds.assign(NAMES=(("row",), names))

    writes = write_datasets(ms, nds, ["NAMES"])
    dask.compute(writes)

    del datasets, ds, writes, nds
    assert_liveness(0, 0)

    with pt.table(ms, readonly=False, ack=False, lockoptions='auto') as T:
        assert name_list == T.getcol("NAMES")
