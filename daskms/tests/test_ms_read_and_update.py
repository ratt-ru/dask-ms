# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask
import dask.array as da
from mock import patch
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

    for ds, expected_row in zip(xds, expected_rows):
        assert ds.ROWID == expected_row

    del xds, ds


@pytest.mark.xfail(reason="Fragmentation not handled in rework, yet")
@pytest.mark.parametrize('group_cols', [
    [],
    ["DATA_DESC_ID"]],
    ids=group_cols_str)
@pytest.mark.parametrize('index_cols', [
    ["TIME"]],
    ids=index_cols_str)
def test_fragmented_ms(ms, group_cols, index_cols):
    select_cols = index_cols + ["STATE_ID"]

    # Zero everything to be sure
    with pt.table(ms, readonly=False, lockoptions='auto', ack=False) as table:
        table.putcol("STATE_ID", np.full(table.nrows(), 0, dtype=np.int32))

    # Patch the get_row_runs function to check that it is called
    # and resorting is invoked
    # Unfragmented is 1.00, induce
    # fragmentation handling
    min_frag_level = 0.9999
    from daskms.dask_ms import get_row_runs
    patch_target = "daskms.dask_ms.get_row_runs"

    def mock_row_runs(*args, **kwargs):
        """ Calls get_row_runs and does some testing """
        row_runs, row_resorts = get_row_runs(*args, **kwargs)
        # Do some checks to ensure that fragmentation was handled
        assert kwargs['min_frag_level'] == min_frag_level
        assert isinstance(row_resorts.compute(), np.ndarray)
        return row_runs, row_resorts

    with patch(patch_target, side_effect=mock_row_runs) as patch_fn:
        xds = xds_from_ms(ms, columns=select_cols,
                          group_cols=group_cols,
                          index_cols=index_cols,
                          min_frag_level=min_frag_level,
                          chunks={"row": 1e9})

    # Check that mock_row_runs was called
    assert patch_fn.called_once_with(min_frag_level=min_frag_level,
                                     sort_dir="read")

    order = orderby_clause(index_cols)
    written_states = []

    with pt.table(ms, readonly=True, lockoptions='auto', ack=False) as table:
        for i, ds in enumerate(xds):
            group_col_values = [getattr(ds, c) for c in group_cols]
            where = where_clause(group_cols, group_col_values)
            query = "SELECT * FROM $table %s %s" % (where, order)

            # Check that each column is correctly read
            with pt.taql(query) as Q:
                for c in select_cols:
                    np_data = Q.getcol(c)
                    dask_data = getattr(ds, c).data.compute()
                    assert np.all(np_data == dask_data)

            # Now write some data to the STATE_ID column
            state = da.arange(i, i + ds.dims['row'], chunks=ds.chunks['row'])
            written_states.append(state)

            nds = ds.assign(STATE_ID=(("row",), state))

            with patch(patch_target, side_effect=mock_row_runs) as patch_fn:
                xds_to_table(nds, ms, "STATE_ID",
                             min_frag_level=min_frag_level).compute()

            assert patch_fn.called_once_with(min_frag_level=min_frag_level,
                                             sort_dir="write")

    # Check that state has been correctly written
    xds = list(xds_from_ms(ms, columns=select_cols,
                           group_cols=group_cols,
                           index_cols=index_cols,
                           min_frag_level=min_frag_level,
                           chunks={"row": 1e9}))

    for i, (ds, expected) in enumerate(zip(xds, written_states)):
        assert np.all(ds.STATE_ID.data.compute() == expected)


@pytest.mark.xfail(reason="Fragmentation not handled in rework, yet")
@pytest.mark.parametrize('group_cols', [
    [],
    ["DATA_DESC_ID"]],
    ids=group_cols_str)
@pytest.mark.parametrize('index_cols', [
    ["TIME"]],
    ids=index_cols_str)
def test_unfragmented_ms(ms, group_cols, index_cols):
    from daskms.dask_ms import get_row_runs
    patch_target = "daskms.dask_ms.get_row_runs"

    def mock_row_runs(*args, **kwargs):
        """ Calls get_row_runs and does some testing """
        # import pdb; pdb.set_trace()
        row_runs, row_resorts = get_row_runs(*args, **kwargs)
        # Do some checks to ensure that fragmentation was handled
        assert kwargs['min_frag_level'] is False
        assert row_resorts.compute() is None
        return row_runs, row_resorts

    with patch(patch_target, side_effect=mock_row_runs) as patch_fn:
        xds = xds_from_ms(ms, columns=index_cols,  # noqa
                          group_cols=group_cols,
                          index_cols=index_cols,
                          min_frag_level=False,
                          chunks={"row": 1e9})

        assert patch_fn.called_once_with(min_frag_level=False, sort_dir="read")


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
    assert_array_equal(xds[0].FIELD_ID.data, [0, 0, 1, 1])
    assert_array_equal(xds[1].FIELD_ID.data, [0, 1, 1])

    # Group on each row
    xds = xds_from_table(ms, taql_where="FIELD_ID >= 0 AND FIELD_ID < 2",
                         group_cols=["__row__"],
                         columns=["FIELD_ID"])

    assert len(xds) == 7

    assert np.all(xds[0].FIELD_ID.data.compute() == 0)
    assert np.all(xds[1].FIELD_ID.data.compute() == 0)
    assert np.all(xds[2].FIELD_ID.data.compute() == 0)
    assert np.all(xds[3].FIELD_ID.data.compute() == 1)
    assert np.all(xds[4].FIELD_ID.data.compute() == 1)
    assert np.all(xds[5].FIELD_ID.data.compute() == 1)
    assert np.all(xds[6].FIELD_ID.data.compute() == 1)


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
    xds = xds_from_ms(ms, group_cols="SCAN_NUMBER", columns=("DATA"))

    for ds in xds:
        assert "DATA" in ds.data_vars
        assert list(ds.attrs.keys()) == ["SCAN_NUMBER"]
