# -*- coding: utf-8 -*-
from pprint import pprint

import dask
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal
import pyrap.tables as pt
import pytest

from daskms.table_proxy import TableProxy, taql_factory, _function_runs
from daskms.query import orderby_clause, where_clause
from daskms.utils import (group_cols_str, index_cols_str, select_cols_str,
                          assert_liveness, table_path_split)
from daskms.dask_ms import (xds_from_ms, xds_from_table, xds_to_table)


@pytest.mark.parametrize('group_cols', [
    ["DATA_DESC_ID", "SCAN_NUMBER"]], ids=group_cols_str)
@pytest.mark.parametrize('index_cols', [
    ["TIME", "ANTENNA1", "ANTENNA2"]], ids=index_cols_str)
@pytest.mark.parametrize('select_cols', [
    ['TIME', 'DATA']], ids=select_cols_str)
def test_readlock_profiling(ms, group_cols, index_cols, select_cols):

    xds = xds_from_ms(ms, columns=select_cols, group_cols=group_cols,
                          index_cols=index_cols)

    assert ('getcol' and 'getcoldesc') in _function_runs.keys()
    # 2 - __tablerows__ , __firstrow__
    assert _function_runs['getcol'][2] == 2 + len(group_cols)
    assert _function_runs['getcoldesc'][2] == len(select_cols) * len(xds)

    pprint(_function_runs)
    getcol_runs = _function_runs['getcol'][2]
    getcoldesc_runs = _function_runs['getcoldesc'][2]

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

        assert _function_runs['getcol'][2] == getcol_runs + len(select_cols)*len(xds)
        assert _function_runs['getcoldesc'][2] == getcoldesc_runs
        del T

    for ds, column_data in zip(xds, np_column_data):
        for c in select_cols:
            dask_data = ds.data_vars[c].data.compute(scheduler='single-threaded')
            assert_array_equal(column_data[c], dask_data)

    # 1 - __tablerow__
    assert _function_runs['getcellslice'][2] == (1 + len(index_cols)) * (len(xds) * len(select_cols))
    # we are reading DATA column ???


@pytest.mark.parametrize('group_cols', [
    ["SCAN_NUMBER"]], ids=group_cols_str)
@pytest.mark.parametrize('index_cols', [
    ["TIME", "ANTENNA1", "ANTENNA2"]], ids=index_cols_str)
@pytest.mark.parametrize('select_cols', [
    ['DATA', 'STATE_ID']])
def test_writelock_profiling(ms, group_cols, index_cols, select_cols):
    # Zero everything to be sure
    with TableProxy(pt.table, ms, readonly=False,
                    lockoptions='auto', ack=False) as T:
        nrows = T.nrows().result()
        assert 'nrows' in _function_runs.keys()
        assert _function_runs['nrows'][2] == 1
    #   put a new column with zeros
        T.putcol("STATE_ID", np.full(nrows, 0, dtype=np.int32)).result()
        assert 'putcol' in _function_runs.keys()
        assert _function_runs['putcol'][2] == 1
    #   get the DATA column and create 'data' variable
        data = np.zeros_like(T.getcol("DATA").result())
        data_dtype = data.dtype
        assert 'getcol' in _function_runs.keys()
        assert _function_runs['getcol'][2] == 1
        getcol_runs = _function_runs['getcol'][2]
    #   put new data into DATA column
        T.putcol("DATA", data).result()
        assert _function_runs['putcol'][2] == 2

    xds = xds_from_ms(ms, columns=select_cols,
                      group_cols=group_cols,
                      index_cols=index_cols,
                      chunks={"row": 2})
    assert 'getcoldesc' in _function_runs.keys()
    # 2 - __tablerows__ , __firstrow__
    assert _function_runs['getcol'][2] == getcol_runs + 2 + len(group_cols)
    assert _function_runs['getcoldesc'][2] == len(select_cols) * len(xds)

    written_states = []
    written_data = []
    writes = []

    # # Write out STATE_ID and DATA
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

    assert ('colnames' and '_put_keywords') in _function_runs.keys()
    assert _function_runs['colnames'][2] == len(xds)
    assert _function_runs['_put_keywords'][2] == len(xds)
    # Do all writes in parallel
    dask.compute(writes)
    
    assert 'getcellslice' in _function_runs.keys()
    print(xds[0].dims, xds[1].dims)
    print(xds[0].chunks, xds[1].chunks)

    # 0 : [ 5 rows, 1 __tablerow__ , 3 index_columns ] : 9 
    # 1 : [ 5 rows, 1 __tablerow__ , 3 index_columns ] : 9
    # or
    # 0 : [ 5 rows, 4 corr, 3 index_columns ] : 12
    # 1 : [ 5 rows, 4 corr, 3 index_columns ] : 12
    # read CASA getcellslice
    
    # assert _function_runs['getcellslice'][2] == len(index_col) len(selecols) len(xds)

    xds = xds_from_ms(ms, columns=select_cols,
                      group_cols=group_cols,
                      index_cols=index_cols,
                      chunks={"row": 2})

    # Check that state and data have been correctly written
    it = enumerate(zip(xds, written_states, written_data))
    for i, (ds, state, data) in it:
        assert_array_equal(ds.STATE_ID.data, state)
        assert_array_equal(ds.DATA.data, data)

# @ToDo or @ToThinkAbout
# Assert statements on all the tests file for _function_runs
# Because no new function was added (just wrappers) on existing functions. 
# 