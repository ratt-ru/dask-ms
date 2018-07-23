from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask.array as da
from mock import patch
import numpy as np
import pyrap.tables as pt
import pytest
import xarray as xr

from xarrayms.xarray_ms import (xds_from_ms,
                                xds_to_table,
                                orderby_clause,
                                where_clause)


@pytest.mark.parametrize('group_cols', [
    [],
    ["FIELD_ID", "DATA_DESC_ID"],
    ["DATA_DESC_ID"],
    ["DATA_DESC_ID", "SCAN_NUMBER"]])
@pytest.mark.parametrize('index_cols', [
    ["ANTENNA2", "ANTENNA1", "TIME"],
    ["TIME", "ANTENNA1", "ANTENNA2"],
    ["ANTENNA1", "ANTENNA2", "TIME"]])
def test_ms_read(ms, group_cols, index_cols):
    select_cols = index_cols

    xds = list(xds_from_ms(ms, columns=select_cols,
                           group_cols=group_cols,
                           index_cols=index_cols,
                           chunks={"row": 2}))

    order = orderby_clause(index_cols)

    with pt.table(ms, lockoptions='auto') as T:  # noqa
        for ds in xds:
            group_col_values = [getattr(ds, c) for c in group_cols]
            where = where_clause(group_cols, group_col_values)
            query = "SELECT * FROM $T %s %s" % (where, order)

            with pt.taql(query) as Q:
                for c in select_cols:
                    np_data = Q.getcol(c)
                    dask_data = getattr(ds, c).data.compute()
                    assert np.all(np_data == dask_data)


@pytest.mark.parametrize('group_cols', [
    [],
    ["FIELD_ID", "DATA_DESC_ID"],
    ["DATA_DESC_ID"],
    ["DATA_DESC_ID", "SCAN_NUMBER"]])
@pytest.mark.parametrize('index_cols', [
    ["ANTENNA2", "ANTENNA1", "TIME"],
    ["TIME", "ANTENNA1", "ANTENNA2"],
    ["ANTENNA1", "ANTENNA2", "TIME"]])
def test_ms_write(ms, group_cols, index_cols):
    select_cols = ["STATE_ID"]

    order = orderby_clause(index_cols)

    # Zero everything to be sure
    with pt.table(ms, readonly=False, lockoptions='auto') as table:
        table.putcol("STATE_ID", np.full(table.nrows(), 0, dtype=np.int32))

    xds = list(xds_from_ms(ms, columns=select_cols,
                           group_cols=group_cols,
                           index_cols=index_cols,
                           chunks={"row": 2}))

    written_states = []

    # Write out STATE_ID
    for i, ds in enumerate(xds):
        state = da.arange(i, i + ds.dims['row'], chunks=ds.chunks['row'])
        written_states.append(state)
        state = xr.DataArray(state, dims=['row'])
        nds = ds.assign(STATE_ID=state)
        xds_to_table(nds, ms, "STATE_ID").compute()

    xds = list(xds_from_ms(ms, columns=select_cols,
                           group_cols=group_cols,
                           index_cols=index_cols,
                           chunks={"row": 2}))

    # Check that state has been correctly written
    for i, (ds, expected) in enumerate(zip(xds, written_states)):
        assert np.all(ds.STATE_ID.data.compute() == expected)


@pytest.mark.parametrize('index_cols', [
    ["ANTENNA2", "ANTENNA1", "TIME"],
    ["TIME", "ANTENNA1", "ANTENNA2"],
    ["ANTENNA1", "ANTENNA2", "TIME"]])
def test_row_query(ms, index_cols):
    xds = list(xds_from_ms(ms, columns=index_cols,
                           group_cols="__row__",
                           index_cols=index_cols,
                           chunks={"row": 2}))

    with pt.table(ms, readonly=False) as table:
        # Get the expected row ordering by lexically
        # sorting the indexing columns
        cols = [(name, table.getcol(name)) for name in index_cols]
        expected_rows = np.lexsort(tuple(c for n, c in reversed(cols)))

        assert len(xds) == table.nrows()

        for ds, expected_row in zip(xds, expected_rows):
            assert ds.table_row == expected_row


@pytest.mark.parametrize('group_cols', [
    [],
    ["DATA_DESC_ID"]])
@pytest.mark.parametrize('index_cols', [
    ["TIME"]])
def test_fragmented_ms(ms, group_cols, index_cols):
    select_cols = index_cols + ["STATE_ID"]

    # Zero everything to be sure
    with pt.table(ms, readonly=False, lockoptions='auto') as table:
        table.putcol("STATE_ID", np.full(table.nrows(), 0, dtype=np.int32))

    # Patch the get_row_runs function to check that it is called
    # and resorting is invoked
    # Unfragmented is 1.00, induce
    # fragmentation handling
    min_frag_level = 0.9999
    from xarrayms.xarray_ms import get_row_runs
    patch_target = "xarrayms.xarray_ms.get_row_runs"

    def mock_row_runs(*args, **kwargs):
        """ Calls get_row_runs and does some testing """
        row_runs, row_resorts = get_row_runs(*args, **kwargs)
        # Do some checks to ensure that fragmentation was handled
        assert kwargs['min_frag_level'] == min_frag_level
        assert all(isinstance(resort, np.ndarray) for resort in row_resorts)
        return row_runs, row_resorts

    with patch(patch_target, side_effect=mock_row_runs) as patch_fn:
        xds = list(xds_from_ms(ms, columns=select_cols,
                               group_cols=group_cols,
                               index_cols=index_cols,
                               min_frag_level=min_frag_level,
                               chunks={"row": 1e9}))

    # Check that mock_row_runs was called
    assert patch_fn.called_once_with(min_frag_level=min_frag_level,
                                     sort_dir="read")

    order = orderby_clause(index_cols)
    written_states = []

    with pt.table(ms, readonly=True, lockoptions='auto') as table:
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
            state = xr.DataArray(state, dims=['row'])
            nds = ds.assign(STATE_ID=state)

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


@pytest.mark.parametrize('group_cols', [
    [],
    ["DATA_DESC_ID"]])
@pytest.mark.parametrize('index_cols', [
    ["TIME"]])
def test_unfragmented_ms(ms, group_cols, index_cols):
    from xarrayms.xarray_ms import get_row_runs
    patch_target = "xarrayms.xarray_ms.get_row_runs"

    def mock_row_runs(*args, **kwargs):
        """ Calls get_row_runs and does some testing """
        row_runs, row_resorts = get_row_runs(*args, **kwargs)
        # Do some checks to ensure that fragmentation was handled
        assert kwargs['min_frag_level'] is False
        assert all(resort is None for resort in row_resorts)
        return row_runs, row_resorts

    with patch(patch_target, side_effect=mock_row_runs) as patch_fn:
        xds = list(xds_from_ms(ms, columns=index_cols,
                               group_cols=group_cols,
                               index_cols=index_cols,
                               min_frag_level=False,
                               chunks={"row": 1e9}))

        assert patch_fn.called_once_with(min_frag_level=False, sort_dir="read")
