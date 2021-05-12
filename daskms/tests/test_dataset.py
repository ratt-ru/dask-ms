# -*- coding: utf-8 -*-

from itertools import product
import os
import uuid

import dask
import dask.array as da
from dask.array.core import normalize_chunks
from dask.highlevelgraph import HighLevelGraph
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pyrap.tables as pt
import pytest

from daskms import xds_from_ms
from daskms.dataset import Dataset, Variable
from daskms.reads import read_datasets
from daskms.writes import write_datasets
from daskms.utils import (select_cols_str, group_cols_str,
                          index_cols_str, assert_liveness)

try:
    import xarray as xr
except ImportError:
    have_xarray = False
else:
    have_xarray = True


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
    ids=lambda s: f"shapes={s}")
@pytest.mark.parametrize("chunks", [
    {"row": 2},
    {"row": 3, "chan": 4, "corr": 1},
    {"row": 3, "chan": (4, 4, 4, 4), "corr": (2, 2)}],
    ids=lambda c: f"chunks={c}")
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
            assert v.dtype == v.data.dtype

        res = dask.compute(compute_dict)[0]

        assert res['DATA'].shape[1:] == (chans, corrs)
        assert 'STATE_ID' in res
        assert 'TIME' in res

        chunks = ds.chunks
        assert chunks["chan"] == echunks['chan']
        assert chunks["corr"] == echunks['corr']

        dims = dict(ds.dims)
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
    ids=lambda s: f"shapes={s}")
@pytest.mark.parametrize("chunks", [
    {"row": 2},
    {"row": 3, "chan": 4, "corr": 1},
    {"row": 3, "chan": (4, 4, 4, 4), "corr": (2, 2)}],
    ids=lambda c: f"chunks={c}")
def test_dataset_updates(ms, select_cols,
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
        states = []
        datas = []

        # Create write operations and execute them
        for i, ds in enumerate(datasets):
            state_var = (("row",), ds.STATE_ID.data + 1)
            data_var = (("row", "chan", "corr"), ds.DATA.data + 1, {})
            states.append(state_var[1])
            datas.append(data_var[1])
            new_ds = ds.assign(STATE_ID=state_var, DATA=data_var)
            writes.append(write_datasets(ms, new_ds, ["STATE_ID", "DATA"]))

        _, states, datas = dask.compute(writes, states, datas)

        # NOTE(sjperkins)
        # Interesting behaviour here. If these objects are not
        # cleared up at this point, attempts to re-open the table below
        # can fail, reproducing https://github.com/ska-sa/dask-ms/issues/26
        # Adding auto-locking to the table opening command seems to fix
        # this somehow
        del ds, new_ds, datasets, writes, state_var, data_var
        assert_liveness(0, 0)

        datasets = read_datasets(ms, select_cols, group_cols,
                                 index_cols, chunks=chunks)

        for i, (ds, state, data) in enumerate(zip(datasets, states, datas)):
            assert_array_equal(ds.STATE_ID.data, state)
            assert_array_equal(ds.DATA.data, data)

        del ds, datasets
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
        assert_array_equal(datasets[i].CHAN_FREQ.data[0], chan_freq)
        assert_array_equal(datasets[i].NUM_CHAN.data[0], chan_freq.shape[0])

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
    nds = ds.assign(TIME=(ds.TIME.dims, ds.TIME.data + 1))
    assert ds.DATA.data is nds.DATA.data
    assert ds.TIME.data is not nds.TIME.data
    assert_array_equal(nds.TIME.data, ds.TIME.data + 1)

    # We have to explicitly supply a dimension schema
    nds = ds.assign(ANTENNA3=(("row",), ds.ANTENNA1.data + 3))
    assert_array_equal(ds.ANTENNA1.data + 3, nds.ANTENNA3.data)

    dims = ds.dims
    chunks = ds.chunks

    if have_xarray:
        match = "'row': length 9 on 'ANTENNA4'"
    else:
        match = ("Existing dimension size 9 for dimension 'row' "
                 "is inconsistent with same dimension 10 of array ANTENNA4")

    with pytest.raises(ValueError, match=match):
        array = da.zeros(dims['row'] - 1, chunks['row'])
        nds = ds.assign(ANTENNA4=(("row",),  array))
        nds.dims

    assert chunks['row'] == (10,)

    if have_xarray:
        match = "Object has inconsistent chunks along dimension row."
    else:
        match = r"chunking \(4, 4, 2\) for dim"

    with pytest.raises(ValueError, match=match):
        array = da.zeros(dims['row'], chunks=4)
        nds = ds.assign(ANTENNA4=(("row",),  array))
        nds.chunks

    del datasets, ds, nds
    assert_liveness(0, 0)


def test_dataset_table_schemas(ms):
    """ Test that we can pass table schemas """
    data_dims = ("mychan", "mycorr")
    table_schema = ["MS", {"DATA": {"dims": data_dims}}]
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
    bool,
    pytest.param(object,
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
    datasets = read_datasets(ms, [], [], [])
    assert len(datasets) == 1
    ds = datasets[0]

    # Create the dask array
    bitflag = da.zeros_like(ds.DATA.data, dtype=dtype)
    # Assign keyword attribute
    col_kw = {"BITFLAG": {'FLAGSETS': 'legacy,cubical',
                          'FLAGSET_legacy': 1,
                          'FLAGSET_cubical': 2}}
    # Assign variable onto the dataset
    nds = ds.assign(BITFLAG=(("row", "chan", "corr"), bitflag))
    writes = write_datasets(ms, nds, ["BITFLAG"], descriptor='ratt_ms',
                            column_keywords=col_kw)

    dask.compute(writes)

    del datasets, ds, writes, nds
    assert_liveness(0, 0)

    with pt.table(ms, readonly=False, ack=False, lockoptions='auto') as T:
        bf = T.getcol("BITFLAG")
        assert T.getcoldesc("BITFLAG")['keywords'] == col_kw['BITFLAG']
        assert bf.dtype == dtype


def test_dataset_add_string_column(ms):
    datasets = read_datasets(ms, [], [], [])
    assert len(datasets) == 1
    ds = datasets[0]
    dims = ds.dims

    name_list = ["BOB"] * dims['row']
    names = np.asarray(name_list, dtype=object)
    names = da.from_array(names, chunks=ds.TIME.chunks)

    nds = ds.assign(NAMES=(("row",), names))

    writes = write_datasets(ms, nds, ["NAMES"])
    dask.compute(writes)

    del datasets, ds, writes, nds
    assert_liveness(0, 0)

    with pt.table(ms, readonly=False, ack=False, lockoptions='auto') as T:
        assert name_list == T.getcol("NAMES")


@pytest.mark.parametrize("chunks", [
    {"row": (36,)},
    {"row": (18, 18)}])
def test_dataset_multidim_string_column(tmp_path, chunks):
    row = sum(chunks['row'])

    name_list = [["X-%d" % i, "Y-%d" % i, "Z-%d" % i] for i in range(row)]
    np_names = np.array(name_list, dtype=object)
    names = da.from_array(np_names, chunks=(chunks['row'], np_names.shape[1]))

    ds = Dataset({"POLARIZATION_TYPE": (("row", "xyz"), names)})
    table_name = str(tmp_path / "test.table")
    writes = write_datasets(table_name, ds, ["POLARIZATION_TYPE"])
    dask.compute(writes)

    del writes
    assert_liveness(0, 0)

    datasets = read_datasets(table_name, [], [], [],
                             chunks={'row': chunks['row']})
    assert len(datasets) == 1
    assert_array_equal(datasets[0].POLARIZATION_TYPE.data, np_names)
    del datasets
    assert_liveness(0, 0)


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
        np_str_array = np.asarray(["BOB"] * shape[0], dtype=object)
        da_str_array = da.from_array(np_str_array, chunks=str_chunks)
        str_array_var = Variable(dims, da_str_array, {})

        datasets.append(Dataset({"DATA": data_var, "NAMES": str_array_var}))
        datas.append(data)
        names.extend(np_str_array.tolist())

    freq = da.linspace(.856e9, 2*.856e9, 64, chunks=16)
    sub_datasets = [Dataset({"FREQ": (("row", "chan"), freq[None, :])})]

    # Write the data to new tables
    table_name = os.path.join(str(tmp_path), 'test.table')
    writes = write_datasets(table_name, datasets, ["DATA", "NAMES"])
    subt_writes = write_datasets(table_name + "::SPW",
                                 sub_datasets, ["FREQ"])
    dask.compute(writes, subt_writes)

    # Check written data
    with pt.table(table_name, readonly=True,
                  lockoptions='auto', ack=False) as T:
        assert row_sum == T.nrows()
        assert_array_equal(T.getcol("DATA"), np.concatenate(datas))
        assert_array_equal(T.getcol("NAMES"), names)

    # Sub-table correctly linked and populated
    with pt.table(table_name + "::SPW", readonly=True,
                  lockoptions='auto', ack=False) as T:
        assert T.nrows() == 1
        assert_array_equal(T.getcol("FREQ")[0], freq)


@pytest.mark.parametrize("chunks", [
    {'row': (5, 3, 2), 'chan': (16,), 'corr': (4,)},
])
@pytest.mark.parametrize("dtype", [np.complex128, np.float32])
def test_write_dict_data(tmp_path, chunks, dtype):
    rs = np.random.RandomState(42)
    row_sum = 0

    def _vis_factory(chan, corr):
        # Variably sized-channels per row, as in BDA data
        nchan = rs.randint(chan)
        return (rs.normal(size=(1, nchan, corr)) +
                rs.normal(size=(1, nchan, corr))*1j)

    shapes = {k: sum(c) for k, c in chunks.items()}
    row_sum += shapes['row']

    # assert len(chunks['chan']) == 1
    assert len(chunks['corr']) == 1

    # Make some visibilities
    dims = ("row", "chan", "corr")
    row, chan, corr = (shapes[d] for d in dims)
    name = "vis-data-" + uuid.uuid4().hex

    nchunks = (len(chunks[d]) for d in dims)
    keys = product((name,), *(range(c) for c in nchunks))
    chunk_sizes = product(*(chunks[d] for d in dims))

    layer = {k: {'r%d' % (i + 1): _vis_factory(chan, corr)
                 for i in range(r)}
             for k, (r, _, _) in zip(keys, chunk_sizes)}

    hlg = HighLevelGraph.from_collections(name, layer, [])
    chunks = tuple(chunks[d] for d in dims)
    meta = np.empty((0,)*len(chunks), dtype=np.complex128)
    vis = da.Array(hlg, name, chunks, meta=meta)
    ds = Dataset({"DATA": (dims, vis)})

    table_name = os.path.join(str(tmp_path), 'test.table')
    writes, table_proxy = write_datasets(table_name, ds, ["DATA"],
                                         table_proxy=True,
                                         # No fixed shape columns
                                         descriptor="ms(False)")

    dask.compute(writes)

    data = table_proxy.getvarcol("DATA").result()

    # First row chunk
    assert_array_almost_equal(layer[(name, 0, 0, 0)]['r1'], data['r1'])
    assert_array_almost_equal(layer[(name, 0, 0, 0)]['r2'], data['r2'])
    assert_array_almost_equal(layer[(name, 0, 0, 0)]['r3'], data['r3'])
    assert_array_almost_equal(layer[(name, 0, 0, 0)]['r4'], data['r4'])
    assert_array_almost_equal(layer[(name, 0, 0, 0)]['r5'], data['r5'])

    # Second row chunk
    assert_array_almost_equal(layer[(name, 1, 0, 0)]['r1'], data['r6'])
    assert_array_almost_equal(layer[(name, 1, 0, 0)]['r2'], data['r7'])
    assert_array_almost_equal(layer[(name, 1, 0, 0)]['r3'], data['r8'])

    # Third row chunk
    assert_array_almost_equal(layer[(name, 2, 0, 0)]['r1'], data['r9'])
    assert_array_almost_equal(layer[(name, 2, 0, 0)]['r2'], data['r10'])


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


@pytest.mark.xfail(reason="https://github.com/pydata/xarray/issues/4882")
def test_dataset_xarray(ms):
    datasets = xds_from_ms(ms)
    datasets = dask.persist(datasets)


@pytest.mark.skipif(
    have_xarray,
    reason="https://github.com/pydata/xarray/issues/4860")
def test_dataset_dask(ms):
    datasets = read_datasets(ms, [], [], [])
    assert len(datasets) == 1
    ds = datasets[0]

    # All dask arrays
    for k, v in ds.data_vars.items():
        assert isinstance(v.data, da.Array)

        # Test variable compute
        v2 = dask.compute(v)[0]
        assert isinstance(v2, xr.DataArray if have_xarray else Variable)
        assert isinstance(v2.data, np.ndarray)

        # Test variable persists
        v3 = dask.persist(v)[0]
        assert isinstance(v3, xr.DataArray if have_xarray else Variable)

        # Now have numpy array in the graph
        assert len(v3.data.__dask_keys__()) == 1
        data = next(iter(v3.__dask_graph__().values()))
        assert isinstance(data, np.ndarray)
        assert_array_equal(v2.data, v3.data)

    # Test compute
    nds = dask.compute(ds)[0]

    for k, v in nds.data_vars.items():
        assert isinstance(v.data, np.ndarray)
        cdata = getattr(ds, k).data
        assert_array_equal(cdata, v.data)

    # Test persist
    nds = dask.persist(ds)[0]

    for k, v in nds.data_vars.items():
        assert isinstance(v.data, da.Array)

        # Now have numpy array iin the graph
        assert len(v.data.__dask_keys__()) == 1
        data = next(iter(v.data.__dask_graph__().values()))
        assert isinstance(data, np.ndarray)

        cdata = getattr(ds, k).data
        assert_array_equal(cdata, v.data)


def test_dataset_numpy(ms):
    datasets = read_datasets(ms, [], [], [])
    assert len(datasets) == 1
    ds = datasets[0]

    row, chan, corr = (ds.dims[d] for d in ("row", "chan", "corr"))

    cdata = np.random.random((row, chan, corr)).astype(np.complex64)
    row_coord = np.arange(row)
    chan_coord = np.arange(chan)
    corr_coord = np.arange(corr)

    ds = ds.assign(**{"CORRECTED_DATA": (("row", "chan", "corr"), cdata)})

    ds = ds.assign_coords(**{
        "row": ("row", row_coord),
        "chan": ("chan", chan_coord),
        "corr": ("corr", corr_coord),
    })

    assert isinstance(ds.CORRECTED_DATA.data, np.ndarray)
    assert_array_equal(ds.CORRECTED_DATA.values, cdata)

    assert isinstance(ds.row.data, np.ndarray)
    assert_array_equal(ds.row.values, row_coord)
    assert isinstance(ds.chan.data, np.ndarray)
    assert_array_equal(ds.chan.values, chan_coord)
    assert isinstance(ds.corr.data, np.ndarray)
    assert_array_equal(ds.corr.values, corr_coord)

    nds = ds.compute()

    for k, v in nds.data_vars.items():
        assert_array_equal(v.data, getattr(ds, k).data)

    for k, v in nds.coords.items():
        assert_array_equal(v.data, getattr(ds, k).data)

    nds, = dask.compute(ds)

    for k, v in nds.data_vars.items():
        assert_array_equal(v.data, getattr(ds, k).data)

    for k, v in nds.coords.items():
        assert_array_equal(v.data, getattr(ds, k).data)

    nds, = dask.persist(ds)

    for k, v in nds.data_vars.items():
        assert_array_equal(v.data, getattr(ds, k).data)

    for k, v in nds.coords.items():
        assert_array_equal(v.data, getattr(ds, k).data)
