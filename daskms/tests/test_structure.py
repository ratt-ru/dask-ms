# -*- coding: utf-8 -*-

import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from daskms.casa_table import CasaTable, taql_table
from daskms.query import groupby_clause, orderby_clause, select_clause
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


def _where(taql_where):
    return f"\nWHERE\n\t{taql_where}" if taql_where else ""


def _taql_rows(table, index_cols, taql_where=""):
    """Rows sorted by a TaQL ORDERBY, which is how dask-ms
    ordered rows before TableStructure"""
    select = select_clause(["ROWID() AS __tablerow__"])
    orderby = orderby_clause(index_cols)
    query = f"{select}\nFROM\n\t$1{_where(taql_where)}\n{orderby}"

    with taql_table(query, (table,)) as result:
        if result.nrow() == 0:
            return np.empty(0, dtype=np.int64)

        return result.getcol("__tablerow__")


def _taql_groups(table, group_cols, index_cols, taql_where):
    """The partitions of a TaQL GROUPBY, which is how dask-ms grouped
    rows before TableStructure, as (key, rows, exemplar_row) tuples
    in dataset order"""
    select = select_clause(
        group_cols
        + [f"GAGGR({c}) AS GROUP_{c}" for c in index_cols]
        + ["GROWID() AS __tablerow__", "GROWID()[0] AS __firstrow__"]
    )
    groupby = groupby_clause(group_cols)
    query = f"{select}\nFROM\n\t$1{_where(taql_where)}\n{groupby}"
    groups = []

    with taql_table(query, (table,)) as result:
        for g in range(result.nrow()):
            # The aggregated columns are ragged, so read one group at a time
            index = (slice(g, g + 1),)
            rows = result.getcol("__tablerow__", index=index)[0]

            if index_cols:
                sort = [
                    result.getcol(f"GROUP_{c}", index=index)[0]
                    for c in reversed(index_cols)
                ]
                rows = rows[np.lexsort(sort)]

            key = tuple(
                (c, result.getcol(c, index=index)[0].item()) for c in group_cols
            )
            exemplar_row = int(result.getcol("__firstrow__", index=index)[0])
            groups.append((key, rows, exemplar_row))

    return groups


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
    expected = _taql_rows(table, index_cols, taql_where)
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
    expected = _taql_rows(table, index_cols)
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


def test_read_graph(ms):
    """A read is the row ids, looked up in the structure, and one getcol
    per block. Nothing else is cached or inlined in the graph"""
    from daskms import xds_from_table

    (ds,) = xds_from_table(ms, columns=["DATA"], chunks={"row": 3, "chan": 4})
    data = ds.DATA.data
    layers = data.__dask_graph__().layers

    assert sorted(name.split("~")[0] for name in layers) == ["read", "rowid"]
    assert data.numblocks == (4, 4, 1)
    assert len(dict(data.__dask_graph__())) == 4 + 4 * 4


def test_read_epoch(ms):
    from daskms import xds_from_table

    def rowid(**kwargs):
        (ds,) = xds_from_table(ms, columns=["TIME"], **kwargs)
        return ds.ROWID.data

    # Each call builds its own structure by default
    assert rowid().name != rowid().name
    # An epoch shares one
    assert rowid(epoch="a").name == rowid(epoch="a").name
    assert rowid(epoch="a").name != rowid(epoch="b").name
