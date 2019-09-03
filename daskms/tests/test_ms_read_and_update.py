# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask
import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal
import pyrap.tables as pt
import pytest


from daskms.query import orderby_clause, where_clause
from daskms.utils import (group_cols_str, index_cols_str,
                          select_cols_str, assert_liveness)
from daskms.table_proxy import TableProxy, taql_factory
from daskms.dask_ms import (xds_from_ms,
                            xds_from_table,
                            xds_to_table)


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
            query = "SELECT * FROM $1 %s %s" % (where, order)

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

        write = xds_to_table(nds, ms, ["STATE_ID", "DATA"])
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

    # Group on each row
    xds = xds_from_table(ms, taql_where="FIELD_ID >= 0 AND FIELD_ID < 2",
                         group_cols=["__row__"],
                         columns=["FIELD_ID"])

    assert len(xds) == 7

    fields = da.concatenate([ds.FIELD_ID.data for ds in xds])
    assert_array_equal(fields, [0, 0, 0, 1, 1, 1, 1])


def _proc_map_fn(args):
    try:
        ms, i = args
        xds = xds_from_ms(ms, columns=["STATE_ID"], group_cols=["FIELD_ID"])
        xds[i] = xds[i].assign(STATE_ID=(("row",), xds[i].STATE_ID.data + i))
        write = xds_to_table(xds[i], ms, ["STATE_ID"])
        write.compute(scheduler='sync')
    except Exception as e:
        print(str(e))

    return True


@pytest.mark.parametrize("nprocs", [3])
def test_multiprocess_table(ms, nprocs):
    # Check here so that we don't fork threads
    # https://rachelbythebay.com/w/2011/06/07/forked/
    assert_liveness(0, 0)

    from multiprocessing import Pool
    pool = Pool(nprocs)

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
        assert list(ds.attrs.keys()) == ["SCAN_NUMBER"]
