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

from daskms.dataset import Dataset, Variable
from daskms.reads import read_datasets
from daskms.writes import write_datasets
from daskms.utils import (select_cols_str, group_cols_str,
                          index_cols_str, assert_liveness)


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
    datasets = read_datasets(ms, select_cols, group_cols,
                             index_cols, chunks=chunks)
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

        for k, v in ds.data_vars.items():
            compute_dict[k] = v.data

            if k in select_cols:
                assert "__coldesc__" in v.attrs

            assert v.dtype == v.data.dtype

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

    del ds, datasets, compute_dict, v
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
        datasets = read_datasets(ms, select_cols, group_cols,
                                 index_cols, chunks=chunks)
        assert_liveness(2, 1)

        # Test writes
        writes = []

        # Create write operations and execute them
        for i, ds in enumerate(datasets):
            new_ds = ds.assign(STATE_ID=ds.STATE_ID.data + 1,
                               DATA=ds.DATA.data + 1)
            writes.append(write_datasets(ms, new_ds, ["STATE_ID", "DATA"]))

        dask.compute(writes)

        # NOTE(sjperkins)
        # Interesting behaviour here. If these objects are not
        # cleared up at this point, attempts to re-open the table below
        # can fail, reproducing https://github.com/ska-sa/dask-ms/issues/26
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
    datasets = read_datasets(spw_table, [], ["__row__"], [], chunks=chunks)

    assert_liveness(2, 1)

    assert len(datasets) == len(spw_chan_freqs)

    for i, chan_freq in enumerate(spw_chan_freqs):
        assert_array_equal(datasets[i].CHAN_FREQ.data, chan_freq)
        assert_array_equal(datasets[i].NUM_CHAN.data, chan_freq.shape[0])

    del datasets
    assert_liveness(0, 0)


def test_antenna_table_string_names(ant_table, wsrt_antenna_positions):
    ds = read_datasets(ant_table, [], [], None)
    assert len(ds) == 1
    ds = ds[0]

    names = ["ANTENNA-%d" % i for i in range(wsrt_antenna_positions.shape[0])]

    assert_array_equal(ds.POSITION.data, wsrt_antenna_positions)
    assert_array_equal(ds.NAME.data, names)

    names = ds.NAME.data.compute()

    # Test that writing back string ndarrays work as
    # they must be converted from ndarrays to lists
    # of strings internally
    write_cols = set(ds.data_vars.keys()) - set(["ROWID"])
    writes = write_datasets(ant_table, ds, write_cols)

    dask.compute(writes)


def test_dataset_assign(ms):
    """ Test dataset assignment """
    datasets = read_datasets(ms, [], [], [])

    assert len(datasets) == 1
    ds = datasets[0]

    # Assign on an existing column is easier because we can
    # infer the dimension schema from it
    nds = ds.assign(TIME=ds.TIME.data + 1)
    assert ds.DATA.data is nds.DATA.data
    assert ds.TIME.data is not nds.TIME.data
    assert_array_equal(nds.TIME.data, ds.TIME.data + 1)

    # This doesn't work for new columns
    with pytest.raises(ValueError, match="Couldn't find existing dimension"):
        ds.assign(ANTENNA3=ds.ANTENNA1.data + 3)

    # We have to explicitly supply a dimension schema
    nds = ds.assign(ANTENNA3=(("row",), ds.ANTENNA1.data + 3))
    assert_array_equal(ds.ANTENNA1.data + 3, nds.ANTENNA3.data)

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
    datasets = read_datasets(ms, [], [], [], table_schema=table_schema)
    assert datasets[0].data_vars["DATA"].dims == ("row", ) + data_dims


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
    import daskms.descriptors.ratt_ms  # noqa. needed for descriptor to work

    datasets = read_datasets(ms, [], [], [])
    assert len(datasets) == 1
    ds = datasets[0]

    bitflag = da.zeros_like(ds.DATA.data, dtype=dtype)
    nds = ds.assign(BITFLAG=(("row", "chan", "corr"), bitflag))
    writes = write_datasets(ms, nds, ["BITFLAG"], descriptor='ratt_ms')

    dask.compute(writes)

    del datasets, ds, writes, nds
    assert_liveness(0, 0)

    with pt.table(ms, readonly=False, ack=False, lockoptions='auto') as T:
        bf = T.getcol("BITFLAG")
        assert bf.dtype == dtype


def test_dataset_add_string_column(ms):
    datasets = read_datasets(ms, [], [], [])
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


@pytest.mark.parametrize("dataset_chunks", [
    [{'row': (5, 3, 2), 'chan': (4, 4, 4, 4), 'corr': (4,)},
     {'row': (4, 3, 3), 'chan': (5, 5, 3, 3), 'corr': (2, 2)}],
])
@pytest.mark.parametrize("dtype", [np.complex128, np.float32])
def test_dataset_create_table(tmp_path, dataset_chunks, dtype):
    datasets = []
    names = []
    datas = []
    row_sum = 0

    for chunks in dataset_chunks:
        shapes = {k: sum(c) for k, c in chunks.items()}
        row_sum += shapes['row']

        # Make some visibilities
        dims = ("row", "chan", "corr")
        shape = tuple(shapes[d] for d in dims)
        data_chunks = tuple(chunks[d] for d in dims)
        data = da.random.random(shape, chunks=data_chunks).astype(dtype)
        data_var = Variable(dims, data, {})

        # Make some string names
        dims = ("row",)
        shape = tuple(shapes[d] for d in dims)
        str_chunks = tuple(chunks[d] for d in dims)
        np_str_array = np.asarray(["BOB"] * shape[0], dtype=np.object)
        da_str_array = da.from_array(np_str_array, chunks=str_chunks)
        str_array_var = Variable(dims, da_str_array, {})

        datasets.append(Dataset({"DATA": data_var, "NAMES": str_array_var}))
        datas.append(data)
        names.extend(np_str_array.tolist())

    # Write the data to a new table
    table_name = os.path.join(str(tmp_path), 'test.table')
    writes = write_datasets(table_name, datasets, ["DATA", "NAMES"])
    dask.compute(writes)

    # Check written data
    with pt.table(table_name, readonly=True,
                  lockoptions='auto', ack=False) as T:
        assert row_sum == T.nrows()
        assert_array_equal(T.getcol("DATA"), np.concatenate(datas))
        assert_array_equal(T.getcol("NAMES"), names)


def test_dataset_computes_and_values(ms):
    datasets = read_datasets(ms, [], [], [])
    assert len(datasets) == 1
    ds = datasets[0]

    # All dask arrays
    for k, v in ds.data_vars.items():
        assert isinstance(v.data, da.Array)

    nds = ds.compute()

    # Now we have numpy arrays that match original data
    for k, v in nds.data_vars.items():
        assert isinstance(v.data, np.ndarray)
        assert_array_equal(v.data, ds.data_vars[k].data)
        assert_array_equal(v.values, ds.data_vars[k].data)
