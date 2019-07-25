# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask
from numpy.testing import assert_array_equal
import pytest

from xarrayms.ordering import group_ordering_taql, row_ordering
from xarrayms.utils import group_cols_str, index_cols_str, assert_liveness


@pytest.mark.parametrize("group_cols", [
    ["FIELD_ID", "SCAN_NUMBER"]],
    ids=group_cols_str)
@pytest.mark.parametrize("index_cols", [
    ["TIME", "ANTENNA1", "ANTENNA2"]],
    ids=index_cols_str)
def test_ordering(ms, group_cols, index_cols):
    group_taql = group_ordering_taql(ms, group_cols, index_cols)
    assert_liveness(1, 1)
    orders = row_ordering(group_taql, group_cols, index_cols, 2)
    assert_liveness(1, 1)
    first_rows = group_taql.getcol("__firstrow__").result()
    assert_liveness(1, 1)

    assert len(first_rows) == len(orders) == 6

    assert_array_equal(first_rows, [0, 1, 3, 4, 7, 8])

    rowids = dask.compute([o[0] for o in orders])[0]

    assert_array_equal(rowids[0], [2, 0])
    assert_array_equal(rowids[1], [1])
    assert_array_equal(rowids[2], [5, 3])
    assert_array_equal(rowids[3], [6, 4])
    assert_array_equal(rowids[4], [9, 7])
    assert_array_equal(rowids[5], [8])

    del group_taql, orders, first_rows
    assert_liveness(0, 0)
