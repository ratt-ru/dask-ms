# -*- coding: utf-8 -*-

import pickle

import dask
import pytest
from numpy.testing import assert_array_equal

from daskms.casa_table import CasaTable
from daskms.ordering import (
    group_ordering_taql,
    group_row_ordering,
    ordering_taql,
    row_ordering,
)
from daskms.structure import ROW_GROUP, TableStructure, structure_factory

INDEX_COLS = [
    [],
    ["TIME"],
    ["TIME", "ANTENNA1", "ANTENNA2"],
    ["ANTENNA1", "TIME"],
    # Has ties, which are broken by table row
    ["ANTENNA1"],
]
GROUP_COLS = [["FIELD_ID"], ["DATA_DESC_ID"], ["FIELD_ID", "SCAN_NUMBER"]]
TAQL_WHERE = ["", "ANTENNA1 != 0", "FIELD_ID != 1"]


def _table(ms):
    return CasaTable.from_table(ms, readonly=True)


def _taql_groups(table, group_cols, index_cols, taql_where):
    """The partitions of the TaQL GROUPBY ordering, as
    (key, rows, exemplar_row) tuples in dataset order"""
    order_taql = group_ordering_taql(table, group_cols, index_cols, taql_where)
    orders = group_row_ordering(order_taql, group_cols, index_cols, [{"row": -1}])
    (rows,) = dask.compute(orders)
    values = [order_taql.instance.getcol(c) for c in group_cols]
    exemplars = order_taql.instance.getcol("__firstrow__")

    return [
        (tuple(zip(group_cols, (v.item() for v in key))), r, int(e))
        for key, r, e in zip(zip(*values), rows, exemplars)
    ]


@pytest.mark.parametrize("taql_where", TAQL_WHERE)
@pytest.mark.parametrize("index_cols", INDEX_COLS)
@pytest.mark.parametrize("group_cols", GROUP_COLS)
def test_structure_matches_group_ordering(ms, group_cols, index_cols, taql_where):
    table = _table(ms)
    expected = _taql_groups(table, group_cols, index_cols, taql_where)
    structure = TableStructure(table, group_cols, index_cols, taql_where)

    assert list(structure) == [key for key, _, _ in expected]

    for key, rows, exemplar_row in expected:
        partition = structure[key]
        assert partition.key == key
        assert_array_equal(partition.rows, rows)
        assert partition.exemplar_row == exemplar_row


@pytest.mark.parametrize("taql_where", TAQL_WHERE)
@pytest.mark.parametrize("index_cols", INDEX_COLS)
def test_structure_matches_row_ordering(ms, index_cols, taql_where):
    table = _table(ms)
    order_taql = ordering_taql(table, index_cols, taql_where)
    (expected,) = dask.compute(row_ordering(order_taql, index_cols, {"row": -1}))
    structure = TableStructure(table, [], index_cols, taql_where)

    assert list(structure) == [()]
    assert_array_equal(structure[()].rows, expected)
    assert structure[()].exemplar_row == min(expected)


def test_structure_first_appearance_order(ms):
    """Partitions keep TaQL GROUPBY's first appearance order,
    rather than being sorted by key"""
    structure = TableStructure(_table(ms), ["FIELD_ID", "SCAN_NUMBER"], [])
    assert [p.exemplar_row for p in structure.values()] == [0, 1, 3, 4, 7, 8]
    assert [dict(k)["SCAN_NUMBER"] for k in structure] == [0, 1, 1, 0, 1, 0]


@pytest.mark.parametrize("index_cols", INDEX_COLS)
def test_structure_row_grouping(ms, index_cols):
    table = _table(ms)
    order_taql = ordering_taql(table, index_cols)
    (expected,) = dask.compute(row_ordering(order_taql, index_cols, {"row": 1}))
    structure = TableStructure(table, [ROW_GROUP], index_cols)

    assert list(structure) == [((ROW_GROUP, int(r)),) for r in expected]

    for row, partition in zip(expected, structure.values()):
        assert_array_equal(partition.rows, [row])
        assert partition.exemplar_row == row


def test_structure_row_grouping_exclusive(ms):
    with pytest.raises(ValueError, match="only grouping column"):
        TableStructure(_table(ms), [ROW_GROUP, "FIELD_ID"], [])


@pytest.mark.parametrize("group_cols", [[], ["FIELD_ID"], [ROW_GROUP]])
def test_structure_empty_selection(ms, group_cols):
    structure = TableStructure(_table(ms), group_cols, ["TIME"], "ROWID() < 0")

    if group_cols:
        # No rows, so no groups
        assert len(structure) == 0
    else:
        # An ungrouped table always has a partition, whose exemplar is row 0
        assert list(structure) == [()]
        assert structure[()].nrow == 0
        assert structure[()].exemplar_row == 0


def test_structure_rows_read_only(ms):
    structure = TableStructure(_table(ms), [], ["TIME"])

    with pytest.raises(ValueError, match="read-only"):
        structure[()].rows[0] = 5


def test_structure_factory(ms):
    table = _table(ms)
    factory = structure_factory(table, ["FIELD_ID"], ["TIME"], epoch="a")

    # Equal arguments share a structure
    same = structure_factory(table, ["FIELD_ID"], ["TIME"], epoch="a")
    assert same == factory
    assert same.instance is factory.instance

    # A different epoch builds a new one
    other = structure_factory(table, ["FIELD_ID"], ["TIME"], epoch="b")
    assert other != factory
    assert other.instance is not factory.instance

    # The factory pickles by its arguments
    unpickled = pickle.loads(pickle.dumps(factory))
    assert unpickled == factory
    assert unpickled.instance is factory.instance
