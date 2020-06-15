# -*- coding: utf-8 -*-

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
from daskms.utils import (group_cols_str, index_cols_str,select_cols_str,
                          assert_liveness, table_path_split)
from daskms.dask_ms import (xds_from_ms, xds_from_table, xds_to_table)


   # Look at the rest of the test cases that use xds_from_ms
   # Set this one's arguments up similarly
   # Use xds_from_ms to produce datasets
   # then call compute on the datasets or the dataset arrays
   # that should invoke your profiling code
   # Whose results you can inspect in _function_runs

# columns
# index_cols
# group_cols
# @pytest.mark.parametrize('group_cols', [
#     [],
#     ["FIELD_ID", "DATA_DESC_ID"],
#     ["DATA_DESC_ID"],
#     ["DATA_DESC_ID", "SCAN_NUMBER"]],
#     ids=group_cols_str)
# @pytest.mark.parametrize('index_cols', [
#     ["ANTENNA2", "ANTENNA1", "TIME"],
#     ["TIME", "ANTENNA1", "ANTENNA2"],
#     ["ANTENNA1", "ANTENNA2", "TIME"]],
#     ids=index_cols_str)
# @pytest.mark.parametrize('select_cols', [
#     ['TIME', 'ANTENNA1', 'DATA']],
#     ids=select_cols_str)
###########################################
# nrows=1, getcoldesc=3, getcol=6
# @pytest.mark.parametrize('group_cols', [
#     []],
#     ids=group_cols_str)
# @pytest.mark.parametrize('index_cols', [
#     ["ANTENNA2", "ANTENNA1", "TIME"]
#     ],
#     ids=index_cols_str)
# @pytest.mark.parametrize('select_cols', [
#     ['TIME', 'ANTENNA1', 'DATA']],
#     ids=select_cols_str)
#############################################
# getcol=12, getcoldesc=9, getcellslice=36
@pytest.mark.parametrize('group_cols', [
    ["FIELD_ID", "DATA_DESC_ID"]],
    ids=group_cols_str)
# @pytest.mark.parametrize('group_cols', [
#     ["FIELD_ID"]],
#     ids=group_cols_str)
@pytest.mark.parametrize('index_cols', [
    ["TIME", "ANTENNA1", "ANTENNA2"]],
    ids=index_cols_str)
@pytest.mark.parametrize('select_cols', [
    ['TIME', 'ANTENNA1', 'DATA']],
    ids=select_cols_str)
def test_readlock_profiling(ms, group_cols, index_cols, select_cols):
    xds = xds_from_ms(ms, columns=select_cols,
                        index_cols=index_cols,
                        group_cols=group_cols)

    # print(type(xds))
    # print(xds[0])
    order = orderby_clause(index_cols)
    np_column_data = []

    # based on the Parameters, get the function names being called
    # based on the parameters, get the number of times (expected) functions
    # will be called
    this_function_runs = {}
    function_name = []
    function_calls = 0

    with TableProxy(pt.table, ms, lockoptions='auto', ack=False) as T:
        for ds in xds:
            assert "ROWID" in ds.coords
            group_col_values = [ds.attrs[a] for a in group_cols]
            # print(group_col_values)
            # [print(ds.attrs[a]) for a in group_cols]
            print(ds.attrs)
            # print(ds.coords)
            print(ds.data_vars)
            where = where_clause(group_cols, group_col_values)
            query = "SELECT * FROM $1 %s %s" % (where, order)
            with TableProxy(taql_factory, query, tables=[T]) as Q:
                column_data = {c: Q.getcol(c).result() for c in select_cols}
                # [print(c) for c in select_cols]
                np_column_data.append(column_data)

        del T

    count = 0
    for d, (ds, column_data) in enumerate(zip(xds, np_column_data)):
        for c in select_cols:
            count+=1
            dask_data = ds.data_vars[c].data.compute(scheduler='single-threaded')
            assert_array_equal(column_data[c], dask_data)

    # print(count)
    # print(ds.data_vars)
    # Read _function_runs (display)
    # print(_function_runs['getcol'])
    from pprint import pprint
    pprint(_function_runs['getcol'])

    # print("group", len(group_cols))
    # print("index", len(index_cols))
    # print("select", len(select_cols))

    attributes = xds[0].attrs
    # print(attributes)
    attrs_list = []
    # Read the Xarray Dataset
    # They are sorted
    for ds in xds:
        # print(ds.attrs)
        # [print(ds.attrs[a]) for a in group_cols]
        for a in group_cols:
            val = ds.attrs[a]
            # print(val, a)
            # attributes[key] = a
            # attributes.update(key = a)

        # print(ds.data_vars)
        # print(ds.coords)
    # print(attributes)
    # Get the number of FIELD_ID's
    # print("xds", len(xds))

    # getcol = len(xds) * (len(select_cols) + len(group_cols))
    # getcoldesc = len(xds) * len(select_cols)
    # getcellslice = None

    # print("getcol", getcol)
    # print("getcoldesc", getcoldesc)



# @pytest.mark.parametrize('group_cols', [
#     [],
#     ["FIELD_ID", "DATA_DESC_ID"],
#     ["DATA_DESC_ID"],
#     ["DATA_DESC_ID", "SCAN_NUMBER"]],
#     ids=group_cols_str)
# @pytest.mark.parametrize('index_cols', [
#     ["ANTENNA2", "ANTENNA1", "TIME"],
#     ["TIME", "ANTENNA1", "ANTENNA2"],
#     ["ANTENNA1", "ANTENNA2", "TIME"]],
#     ids=index_cols_str)
# @pytest.mark.parametrize('select_cols', [
#     ['DATA', 'STATE_ID']])
# def test_writelock_profiling(ms, group_cols, index_cols, select_cols):
#     # Zero everything to be sure
#     with TableProxy(pt.table, ms, readonly=False,
#                     lockoptions='auto', ack=False) as T:
#         nrows = T.nrows().result()
#         # print("nrows", nrows)
#         # print("T.putcol - 1")
#         T.putcol("STATE_ID", np.full(nrows, 0, dtype=np.int32)).result()
#         # print("T.getcol")
#         data = np.zeros_like(T.getcol("DATA").result())
#         data_dtype = data.dtype
#         # print("T.putcol - 2")
#         T.putcol("DATA", data).result()
#
#     xds = xds_from_ms(ms, columns=select_cols,
#                       group_cols=group_cols,
#                       index_cols=index_cols,
#                       chunks={"row": 2})
#
#     written_states = []
#     written_data = []
#     writes = []
#
#     # Write out STATE_ID and DATA
#     for i, ds in enumerate(xds):
#         dims = ds.dims
#         chunks = ds.chunks
#         state = da.arange(i, i + dims["row"], chunks=chunks["row"])
#         state = state.astype(np.int32)
#         written_states.append(state)
#
#         data = da.arange(i, i + dims["row"]*dims["chan"]*dims["corr"])
#         data = data.reshape(dims["row"], dims["chan"], dims["corr"])
#         data = data.rechunk((chunks["row"], chunks["chan"], chunks["corr"]))
#         data = data.astype(data_dtype)
#         written_data.append(data)
#
#         nds = ds.assign(STATE_ID=(("row",), state),
#                         DATA=(("row", "chan", "corr"), data))
#
#         write = xds_to_table(nds, ms, ["STATE_ID", "DATA"])
#         writes.append(write)
#
#     # Do all writes in parallel
#     dask.compute(writes)
#
#     xds = xds_from_ms(ms, columns=select_cols,
#                       group_cols=group_cols,
#                       index_cols=index_cols,
#                       chunks={"row": 2})
#
#     # Check that state and data have been correctly written
#     it = enumerate(zip(xds, written_states, written_data))
#     for i, (ds, state, data) in it:
#         assert_array_equal(ds.STATE_ID.data, state)
#         assert_array_equal(ds.DATA.data, data)
#
#     # Read _function_runs (display)
#     print(_function_runs)
