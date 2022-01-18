# -*- coding: utf-8 -*-

import sys

import dask
import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal
import pyrap.tables as pt
import pytest

try:
    from dask.optimization import key_split
except ImportError:
    from dask.utils import key_split

from daskms.constants import DASKMS_PARTITION_KEY
from daskms.dask_ms import (xds_from_ms,
                            xds_from_table,
                            xds_to_table)
from daskms.query import orderby_clause, where_clause
from daskms.table_proxy import TableProxy, taql_factory
from daskms.utils import (group_cols_str, index_cols_str,
                          select_cols_str,
                          table_path_split)


PY_37_GTE = sys.version_info[:2] >= (3, 7)


@pytest.mark.parametrize('group_cols', [
    [],
    ["FIELD_ID", "DATA_DESC_ID"],
    ["DATA_DESC_ID"],
    ["DATA_DESC_ID", "SCAN_NUMBER"]],
    ids=group_cols_str)
@pytest.mark.parametrize('index_cols', [
    ["ANTENNA2", "ANTENNA1", "TIME"],
    ["TIME", "ANTENNA1", "ANTENNA2"],
    ["ANTENNA1", "ANTENNA2", "TIME"]],
    ids=index_cols_str)
@pytest.mark.parametrize('select_cols', [
    ['TIME', 'ANTENNA1', 'DATA']],
    ids=select_cols_str)
def test_ms_read(ms, group_cols, index_cols, select_cols):
    xds = xds_from_ms(ms, columns=select_cols,
                      group_cols=group_cols,
                      index_cols=index_cols,
                      chunks={"row": 2})

    order = orderby_clause(index_cols)
    np_column_data = []

    with TableProxy(pt.table, ms, lockoptions='auto', ack=False) as T:
        for ds in xds:
            assert "ROWID" in ds.coords
            group_col_values = [ds.attrs[a] for a in group_cols]
            where = where_clause(group_cols, group_col_values)
            query = f"SELECT * FROM $1 {where} {order}"

            with TableProxy(taql_factory, query, tables=[T]) as Q:
                column_data = {c: Q.getcol(c).result() for c in select_cols}
                np_column_data.append(column_data)

    del T

    for d, (ds, column_data) in enumerate(zip(xds, np_column_data)):
        for c in select_cols:
            dask_data = ds.data_vars[c].data.compute()
            assert_array_equal(column_data[c], dask_data)


@pytest.mark.parametrize('group_cols', [
    [],
    ["FIELD_ID", "DATA_DESC_ID"],
    ["DATA_DESC_ID"],
    ["DATA_DESC_ID", "SCAN_NUMBER"]],
    ids=group_cols_str)
@pytest.mark.parametrize('index_cols', [
    ["ANTENNA2", "ANTENNA1", "TIME"],
    ["TIME", "ANTENNA1", "ANTENNA2"],
    ["ANTENNA1", "ANTENNA2", "TIME"]],
    ids=index_cols_str)
@pytest.mark.parametrize('select_cols', [
    ['DATA', 'STATE_ID']])
def test_ms_update(ms, group_cols, index_cols, select_cols):
    # Zero everything to be sure
    with TableProxy(pt.table, ms, readonly=False,
                    lockoptions='auto', ack=False) as T:
        nrows = T.nrows().result()
        T.putcol("STATE_ID", np.full(nrows, 0, dtype=np.int32)).result()
        data = np.zeros_like(T.getcol("DATA").result())
        data_dtype = data.dtype
        T.putcol("DATA", data).result()

    xds = xds_from_ms(ms, columns=select_cols,
                      group_cols=group_cols,
                      index_cols=index_cols,
                      chunks={"row": 2})

    written_states = []
    written_data = []
    writes = []

    # Write out STATE_ID and DATA
    for i, ds in enumerate(xds):
        dims = ds.dims
        chunks = ds.chunks
        state = da.arange(i, i + dims["row"], chunks=chunks["row"])
        state = state.astype(np.int32)
        written_states.append(state)

        data = da.arange(i, i + dims["row"]*dims["chan"]*dims["corr"])
        data = data.reshape(dims["row"], dims["chan"], dims["corr"])
        data = data.rechunk((chunks["row"], chunks["chan"], chunks["corr"]))
        data = data.astype(data_dtype)
        written_data.append(data)

        nds = ds.assign(STATE_ID=(("row",), state),
                        DATA=(("row", "chan", "corr"), data))

        write, = xds_to_table(nds, ms, ["STATE_ID", "DATA"])

        for k, _ in nds.attrs[DASKMS_PARTITION_KEY]:
            assert getattr(write, k) == getattr(nds, k)

        writes.append(write)

    # Do all writes in parallel
    dask.compute(writes)

    xds = xds_from_ms(ms, columns=select_cols,
                      group_cols=group_cols,
                      index_cols=index_cols,
                      chunks={"row": 2})

    # Check that state and data have been correctly written
    it = enumerate(zip(xds, written_states, written_data))
    for i, (ds, state, data) in it:
        assert_array_equal(ds.STATE_ID.data, state)
        assert_array_equal(ds.DATA.data, data)


@pytest.mark.parametrize('index_cols', [
    ["ANTENNA2", "ANTENNA1", "TIME"],
    ["TIME", "ANTENNA1", "ANTENNA2"],
    ["ANTENNA1", "ANTENNA2", "TIME"]],
    ids=index_cols_str)
def test_row_query(ms, index_cols):
    T = TableProxy(pt.table, ms, readonly=True, lockoptions='auto', ack=False)

    # Get the expected row ordering by lexically
    # sorting the indexing columns
    cols = [(name, T.getcol(name).result()) for name in index_cols]
    expected_rows = np.lexsort(tuple(c for n, c in reversed(cols)))

    del T

    xds = xds_from_ms(ms, columns=index_cols,
                      group_cols="__row__",
                      index_cols=index_cols,
                      chunks={"row": 2})

    actual_rows = da.concatenate([ds.ROWID.data for ds in xds])
    assert_array_equal(actual_rows, expected_rows)


@pytest.mark.parametrize('index_cols', [
    ["TIME", "ANTENNA1", "ANTENNA2"]],
    ids=index_cols_str)
def test_taql_where(ms, index_cols):
    # three cases test here, corresponding to the
    # if-elif-else ladder in xds_from_table

    # No group_cols case
    xds = xds_from_table(ms, taql_where="FIELD_ID >= 0 AND FIELD_ID < 2",
                         columns=["FIELD_ID"])

    assert len(xds) == 1
    assert_array_equal(xds[0].FIELD_ID.data, [0, 0, 0, 1, 1, 1, 1])

    # Group columns case
    xds = xds_from_table(ms, taql_where="FIELD_ID >= 0 AND FIELD_ID < 2",
                         group_cols=["DATA_DESC_ID", "SCAN_NUMBER"],
                         columns=["FIELD_ID"])

    assert len(xds) == 2

    # Check group id's
    assert xds[0].DATA_DESC_ID == 0 and xds[0].SCAN_NUMBER == 0
    assert xds[1].DATA_DESC_ID == 0 and xds[1].SCAN_NUMBER == 1

    # Check field id's in each group
    fields = da.concatenate([ds.FIELD_ID.data for ds in xds])
    assert_array_equal(fields, [0, 0, 1, 1, 0, 1, 1])

    # Group columns case
    xds = xds_from_table(ms, taql_where="FIELD_ID >= 0 AND FIELD_ID < 2",
                         group_cols=["DATA_DESC_ID", "FIELD_ID"],
                         columns=["FIELD_ID"])

    assert len(xds) == 2

    # Check group id's, no DATA_DESC_ID == 1 because it only
    # contains FIELD_ID == 2
    assert xds[0].DATA_DESC_ID == 0 and xds[0].FIELD_ID == 0
    assert xds[1].DATA_DESC_ID == 0 and xds[1].FIELD_ID == 1

    # Group on each row
    xds = xds_from_table(ms, taql_where="FIELD_ID >= 0 AND FIELD_ID < 2",
                         group_cols=["__row__"],
                         columns=["FIELD_ID"])

    assert len(xds) == 7

    fields = da.concatenate([ds.FIELD_ID.data for ds in xds])
    assert_array_equal(fields, [0, 0, 0, 1, 1, 1, 1])


def _proc_map_fn(args):
    import dask.threaded as dt

    # No dask pools are spun up
    with dt.pools_lock:
        assert dt.default_pool is None
        assert len(dt.pools) == 0

    try:
        ms, i = args
        xds = xds_from_ms(ms, columns=["STATE_ID"], group_cols=["FIELD_ID"])
        xds[i] = xds[i].assign(STATE_ID=(("row",), xds[i].STATE_ID.data + i))
        write = xds_to_table(xds[i], ms, ["STATE_ID"])
        dask.compute(write)
    except Exception as e:
        print(str(e))

    return True


@pytest.mark.parametrize("nprocs", [3])
def test_multiprocess_table(ms, nprocs):
    from multiprocessing import get_context
    pool = get_context("spawn").Pool(nprocs)

    try:
        args = [tuple((ms, i)) for i in range(nprocs)]
        assert all(pool.map(_proc_map_fn, args))
    finally:
        pool.close()


@pytest.mark.parametrize('group_cols', [
    ["FIELD_ID", "DATA_DESC_ID"],
    ["DATA_DESC_ID", "SCAN_NUMBER"]],
    ids=group_cols_str)
@pytest.mark.parametrize('index_cols', [
    ["TIME", "ANTENNA1", "ANTENNA2"]],
    ids=index_cols_str)
def test_multireadwrite(ms, group_cols, index_cols):
    xds = xds_from_ms(ms, group_cols=group_cols, index_cols=index_cols)

    nds = [ds.copy() for ds in xds]
    writes = [xds_to_table(sds, ms,
                           [k for k in sds.data_vars.keys() if k != "ROWID"])
              for sds in nds]

    da.compute(writes)


def test_column_promotion(ms):
    """ Test singleton columns promoted to lists """
    xds = xds_from_ms(ms, group_cols="SCAN_NUMBER", columns=("DATA",))

    for ds in xds:
        assert "DATA" in ds.data_vars
        assert "SCAN_NUMBER" in ds.attrs
        assert ds.attrs[DASKMS_PARTITION_KEY] == (("SCAN_NUMBER", "int32"),)


def test_read_array_names(ms):
    _, short_name, _ = table_path_split(ms)
    datasets = xds_from_ms(ms)

    for ds in datasets:
        for k, v in ds.data_vars.items():
            product = ("~[" + str(ds.FIELD_ID) +
                       "," + str(ds.DATA_DESC_ID) + "]")
            prefix = "".join(("read~", k, product))
            assert key_split(v.data.name) == prefix


def test_write_array_names(ms, tmp_path):
    _, short_name, _ = table_path_split(ms)
    datasets = xds_from_ms(ms)

    out_table = str(tmp_path / short_name)

    writes = xds_to_table(datasets, out_table, "ALL")

    for ds in writes:
        for k, v in ds.data_vars.items():
            prefix = "".join(("write~", k))
            assert key_split(v.data.name) == prefix
