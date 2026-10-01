# -*- coding: utf-8 -*-
"""Reads of variably shaped columns, whose cells are padded to a maximal shape"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from daskms import xds_from_table
from daskms.casa_table import create_table

# Cell shapes of the VAR and VSTR columns in each row. Row 4 is undefined
NCHAN = [4, 2, 3, 1, None, 2]
GROUP = [0, 0, 0, 1, 1, 1]
NCORR = 2


@pytest.fixture
def var_table(tmp_path):
    path = str(tmp_path / "var.table")
    desc = {
        "GROUP": {"valueType": "int", "option": 0},
        "TIME": {"valueType": "double", "option": 0},
        "VAR": {"valueType": "float", "ndim": 2, "option": 0, "_c_order": True},
        "VSTR": {"valueType": "string", "ndim": 1, "option": 0, "_c_order": True},
    }

    with create_table(path, table_desc=desc, nrow=len(NCHAN)) as t:
        t.putcol("GROUP", np.array(GROUP, np.int32))
        # Descending TIME so that ordering on it reverses the rows
        t.putcol("TIME", np.arange(len(NCHAN), 0, -1, dtype=np.float64))

        for row, nchan in enumerate(NCHAN):
            if nchan is None:
                continue

            index = (slice(row, row + 1),)
            cell = _var_cell(row, nchan)
            t.putcol("VAR", cell[None], index=index)
            t.putcol("VSTR", np.array([_vstr_cell(row, nchan)], object), index=index)

    return path


def _var_cell(row, nchan):
    return np.arange(nchan * NCORR, dtype=np.float32).reshape(nchan, NCORR) + row


def _vstr_cell(row, nchan):
    return [f"r{row}c{c}" for c in range(nchan)]


def _expected_var(rows, nchan):
    expected = np.full((len(rows), nchan, NCORR), np.nan, np.float32)

    for i, row in enumerate(rows):
        if NCHAN[row] is not None:
            expected[i, : NCHAN[row]] = _var_cell(row, NCHAN[row])

    return expected


def _expected_vstr(rows, nchan):
    expected = np.full((len(rows), nchan), "", object)

    for i, row in enumerate(rows):
        if NCHAN[row] is not None:
            expected[i, : NCHAN[row]] = _vstr_cell(row, NCHAN[row])

    return expected


@pytest.mark.parametrize(
    "chunks", [{"row": 2}, {"row": 4}], ids=lambda c: f"chunks={c}"
)
@pytest.mark.parametrize("index_cols", [[], ["TIME"]])
def test_variable_column_padded(var_table, chunks, index_cols):
    (ds,) = xds_from_table(
        var_table, columns=["VAR", "VSTR"], index_cols=index_cols, chunks=chunks
    )
    rows = ds.ROWID.values

    if index_cols:
        assert_array_equal(rows, np.arange(len(NCHAN))[::-1])

    # Cells are padded to the largest cell
    assert ds.VAR.shape == (len(NCHAN), 4, NCORR)
    assert ds.VSTR.shape == (len(NCHAN), 4)
    assert_array_equal(ds.VAR.values, _expected_var(rows, 4))
    assert_array_equal(ds.VSTR.values, _expected_vstr(rows, 4))


def test_variable_column_group_shapes(var_table):
    """Each group is padded to its own largest cell, not the column's"""
    datasets = xds_from_table(
        var_table, columns=["VAR", "VSTR"], group_cols=["GROUP"], chunks={"row": 2}
    )

    assert len(datasets) == 2

    for ds, nchan in zip(datasets, [4, 2]):
        rows = ds.ROWID.values
        assert ds.VAR.shape == (len(rows), nchan, NCORR)
        assert_array_equal(ds.VAR.values, _expected_var(rows, nchan))
        assert_array_equal(ds.VSTR.values, _expected_vstr(rows, nchan))


def test_variable_column_empty_selection(var_table):
    (ds,) = xds_from_table(var_table, columns=["VAR"], taql_where="ROWID() < 0")
    # The exemplar row supplies the shape of an empty dataset
    assert ds.VAR.shape == (0, 4, NCORR)
    assert ds.VAR.values.shape == (0, 4, NCORR)


def test_variable_column_chunked_regular_cells(var_table):
    """Non-row dimensions can be chunked if a dataset's cells share a shape"""
    # Rows 1 and 5 both have 2 channels
    (ds,) = xds_from_table(
        var_table,
        columns=["VAR", "VSTR"],
        taql_where="ROWID() IN [1, 5]",
        chunks={"row": 2, "VAR-1": 1, "VSTR-1": 1},
    )
    rows = ds.ROWID.values
    assert_array_equal(rows, [1, 5])
    assert ds.VAR.data.chunks[1] == (1, 1)
    assert_array_equal(ds.VAR.values, _expected_var(rows, 2))
    assert_array_equal(ds.VSTR.values, _expected_vstr(rows, 2))


def test_variable_column_chunked_ragged_cells(var_table):
    """Chunking the non-row dimensions of numeric cells
    with different shapes fails"""
    (ds,) = xds_from_table(var_table, columns=["VAR"], chunks={"row": 6, "VAR-1": 3})

    with pytest.raises(IndexError, match="Group the table"):
        ds.VAR.values


def test_variable_string_column_chunked_ragged_cells(var_table):
    """Strings are read as whole cells, so ragged chunks are padded"""
    (ds,) = xds_from_table(var_table, columns=["VSTR"], chunks={"row": 6, "VSTR-1": 3})
    assert ds.VSTR.data.chunks[1] == (3, 1)
    assert_array_equal(ds.VSTR.values, _expected_vstr(ds.ROWID.values, 4))
