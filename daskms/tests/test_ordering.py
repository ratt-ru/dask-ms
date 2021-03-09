# -*- coding: utf-8 -*-

import dask
import dask.array as da
from numpy.testing import assert_array_equal
import pyrap.tables as pt
import pytest

from daskms.table_proxy import TableProxy
from daskms.ordering import (ordering_taql,
                             row_ordering,
                             group_ordering_taql,
                             group_row_ordering)
from daskms.utils import group_cols_str, index_cols_str, assert_liveness


def table_proxy(ms):
    return TableProxy(pt.table, ms, ack=False,
                      lockoptions='user', readonly=True)


@pytest.mark.parametrize("group_cols", [
    ["FIELD_ID", "SCAN_NUMBER"]],
    ids=group_cols_str)
@pytest.mark.parametrize("index_cols", [
    ["TIME", "ANTENNA1", "ANTENNA2"]],
    ids=index_cols_str)
def test_ordering_query_taql_where_strings(ms, group_cols, index_cols):
    taql = group_ordering_taql(table_proxy(ms), group_cols, index_cols,
                               taql_where="ANTENNA1 != ANTENNA2")
    assert taql._args[0].replace("\t", " "*4) == (
        "SELECT\n"
        "    FIELD_ID,\n"
        "    SCAN_NUMBER,\n"
        "    GAGGR(TIME) as GROUP_TIME,\n"
        "    GAGGR(ANTENNA1) as GROUP_ANTENNA1,\n"
        "    GAGGR(ANTENNA2) as GROUP_ANTENNA2,\n"
        "    GROWID() AS __tablerow__,\n"
        "    GCOUNT() as __tablerows__,\n"
        "    GROWID()[0] as __firstrow__\n"
        "FROM\n"
        "    $1\n"
        "WHERE\n"
        "    ANTENNA1 != ANTENNA2\n"
        "GROUPBY\n"
        "    FIELD_ID,\n"
        "    SCAN_NUMBER")

    taql = group_ordering_taql(table_proxy(ms), group_cols, index_cols)
    assert taql._args[0].replace("\t", " "*4) == (
        "SELECT\n"
        "    FIELD_ID,\n"
        "    SCAN_NUMBER,\n"
        "    GAGGR(TIME) as GROUP_TIME,\n"
        "    GAGGR(ANTENNA1) as GROUP_ANTENNA1,\n"
        "    GAGGR(ANTENNA2) as GROUP_ANTENNA2,\n"
        "    GROWID() AS __tablerow__,\n"
        "    GCOUNT() as __tablerows__,\n"
        "    GROWID()[0] as __firstrow__\n"
        "FROM\n"
        "    $1\n"
        "GROUPBY\n"
        "    FIELD_ID,\n"
        "    SCAN_NUMBER")

    taql = group_ordering_taql(table_proxy(ms), group_cols, [])
    assert taql._args[0].replace("\t", " "*4) == (
        "SELECT\n"
        "    FIELD_ID,\n"
        "    SCAN_NUMBER,\n"
        "    GROWID() AS __tablerow__,\n"
        "    GCOUNT() as __tablerows__,\n"
        "    GROWID()[0] as __firstrow__\n"
        "FROM\n"
        "    $1\n"
        "GROUPBY\n"
        "    FIELD_ID,\n"
        "    SCAN_NUMBER")

    taql = ordering_taql(table_proxy(ms), index_cols,
                         taql_where="ANTENNA1 != ANTENNA2")
    assert taql._args[0].replace("\t", " "*4) == (
        "SELECT\n"
        "    ROWID() as __tablerow__\n"
        "FROM\n"
        "    $1\n"
        "WHERE\n"
        "    ANTENNA1 != ANTENNA2\n"
        "ORDERBY\n"
        "    TIME,\n"
        "    ANTENNA1,\n"
        "    ANTENNA2")

    taql = ordering_taql(table_proxy(ms), index_cols)
    assert taql._args[0].replace("\t", " "*4) == (
        "SELECT\n"
        "    ROWID() as __tablerow__\n"
        "FROM\n"
        "    $1\n"
        "ORDERBY\n"
        "    TIME,\n"
        "    ANTENNA1,\n"
        "    ANTENNA2")

    taql = ordering_taql(table_proxy(ms), [])
    assert taql._args[0].replace("\t", " "*4) == (
        "SELECT\n"
        "    ROWID() as __tablerow__\n"
        "FROM\n"
        "    $1\n")


@pytest.mark.parametrize("group_cols", [
    ["FIELD_ID", "SCAN_NUMBER"]],
    ids=group_cols_str)
@pytest.mark.parametrize("index_cols", [
    ["TIME", "ANTENNA1", "ANTENNA2"]],
    ids=index_cols_str)
def test_ordering_multiple_groups(ms, group_cols, index_cols):
    group_taql = group_ordering_taql(table_proxy(ms), group_cols, index_cols)
    assert_liveness(2, 1)
    orders = group_row_ordering(group_taql, group_cols,
                                index_cols, [{'row': 2}])
    assert_liveness(2, 1)
    first_rows = group_taql.getcol("__firstrow__").result()
    assert_liveness(2, 1)

    assert len(first_rows) == len(orders) == 6

    assert_array_equal(first_rows, [0, 1, 3, 4, 7, 8])

    rowid_arrays = tuple(o[0] for o in orders)
    rowids = dask.compute(rowid_arrays)[0]

    assert_array_equal(rowids[0], [2, 0])
    assert_array_equal(rowids[1], [1])
    assert_array_equal(rowids[2], [5, 3])
    assert_array_equal(rowids[3], [6, 4])
    assert_array_equal(rowids[4], [9, 7])
    assert_array_equal(rowids[5], [8])

    del first_rows, orders, rowid_arrays, group_taql
    assert_liveness(0, 0)


@pytest.mark.parametrize("index_cols", [
    ["TIME", "ANTENNA1", "ANTENNA2"]],
    ids=index_cols_str)
@pytest.mark.parametrize("chunks", [
    {'row': 2},
    {'row': (2, 3, 4, 1)},
    {'row': (5, 3, 2)}],
    ids=lambda c: f'chunks={c}')
def test_row_ordering_no_group(ms, index_cols, chunks):
    order_taql = ordering_taql(table_proxy(ms), index_cols)
    assert_liveness(2, 1)
    orders = row_ordering(order_taql, index_cols, chunks)
    assert_liveness(2, 1)

    # Normalise chunks to match that of the output array
    expected_chunks = da.core.normalize_chunks(chunks['row'], (10,))

    assert orders[0].chunks == expected_chunks

    rowids = dask.compute(orders[0])[0]
    assert_array_equal(rowids, [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    del orders, order_taql
    assert_liveness(0, 0)


# Grouping on DATA_DESC_ID gives us two groups
# one with 7 rows and the other with 3
@pytest.mark.parametrize("group_cols", [
    ["DATA_DESC_ID"]],
    ids=group_cols_str)
@pytest.mark.parametrize("index_cols", [
    ["TIME", "ANTENNA1", "ANTENNA2"]],
    ids=index_cols_str)
@pytest.mark.parametrize("chunks", [
    [{'row': 2}, {'row': 2}],
    [{'row': (3, 4)}, {'row': 3}],
    [{'row': (2, 3, 2)}, {'row': (2, 1)}],
    [{'row': 2}]],
    ids=lambda c: f'chunks={c}')
def test_row_ordering_multiple_groups(ms, group_cols,
                                      index_cols, chunks):
    group_taql = group_ordering_taql(table_proxy(ms), group_cols, index_cols)
    assert_liveness(2, 1)
    orders = group_row_ordering(group_taql, group_cols, index_cols, chunks)
    assert_liveness(2, 1)
    first_rows = group_taql.getcol("__firstrow__").result()
    assert_liveness(2, 1)

    # We get two groups out
    assert len(orders) == len(first_rows) == 2
    assert_array_equal(first_rows, [0, 7])

    rowid_arrays = tuple(o[0] for o in orders)
    rowids = dask.compute(rowid_arrays)[0]

    # Check the two resulting groups

    # Normalise chunks to match that of the output array
    row_chunks = chunks[0]['row']
    expected_chunks = da.core.normalize_chunks(row_chunks, (7,))
    assert_array_equal(rowids[0], [6, 5, 4, 3, 2, 1, 0])
    assert rowid_arrays[0].chunks == expected_chunks

    # If chunks only supplied for the first group, re-use it's chunking
    row_chunks = chunks[0]['row'] if len(chunks) == 1 else chunks[1]['row']
    expected_chunks = da.core.normalize_chunks(row_chunks, (3,))
    assert_array_equal(rowids[1], [9, 8, 7])
    assert rowid_arrays[1].chunks == expected_chunks

    del first_rows, orders, rowid_arrays, group_taql
    assert_liveness(0, 0)
