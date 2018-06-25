from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import os

import dask.array as da
from mock import patch
import numpy as np
import pyrap.tables as pt
import pytest
import xarray as xr

from xarrayms.xarray_ms import (_DEFAULT_GROUP_COLUMNS,
                                _DEFAULT_INDEX_COLUMNS,
                                xds_from_ms,
                                xds_to_table,
                                select_clause,
                                orderby_clause,
                                groupby_clause,
                                where_clause)


@pytest.fixture(scope="session")
def ms(tmpdir_factory):
    msdir = tmpdir_factory.mktemp("msdir", numbered=False)
    fn = os.path.join(str(msdir), "test.ms")

    create_table_query = """
    CREATE TABLE %s
    [FIELD_ID I4,
    ANTENNA1 I4,
    ANTENNA2 I4,
    DATA_DESC_ID I4,
    SCAN_NUMBER I4,
    STATE_ID I4,
    TIME R8]
    LIMIT 10
    """ % fn

    # Common grouping columns
    field = [0,   0,   0,   1,   1,   1,   1,   2,   2,   2]
    ddid = [0,   0,   0,   0,   0,   0,   0,   1,   1,   1]
    scan = [0,   1,   0,   1,   0,   1,   0,   1,   0,   1]

    # Common indexing columns
    time = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    ant1 = [0,   0,   1,   1,   1,   2,   1,   0,   0,   1]
    ant2 = [1,   2,   2,   3,   2,   1,   0,   1,   1,   2]

    # Column we'll write to
    state = [0,   0,   0,   0,   0,   0,   0,   0,   0,   0]

    # Create the table
    with pt.taql(create_table_query) as ms:
        ms.putcol("TIME", time)
        ms.putcol("FIELD_ID", field)
        ms.putcol("DATA_DESC_ID", ddid)
        ms.putcol("ANTENNA1", ant1)
        ms.putcol("ANTENNA2", ant2)
        ms.putcol("SCAN_NUMBER", scan)
        ms.putcol("STATE_ID", state)

    yield fn

    # Remove the temporary directory
    shutil.rmtree(str(msdir))


@pytest.mark.parametrize('group_cols', [
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

    for ds in xds:
        where = where_clause(group_cols, [getattr(ds, c) for c in group_cols])
        query = "SELECT * FROM $ms %s %s" % (where, order)

        with pt.taql(query) as Q:
            for c in select_cols:
                np_data = Q.getcol(c)
                dask_data = getattr(ds, c).data.compute()
                assert np.all(np_data == dask_data)


@pytest.mark.parametrize('group_cols', [
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
    with pt.table(ms, readonly=False) as table:
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


@pytest.mark.parametrize('group_cols', [
    ["DATA_DESC_ID"]])
@pytest.mark.parametrize('index_cols', [
    ["TIME"]])
def test_fragmented_ms(ms, group_cols, index_cols):
    select_cols = index_cols + ["STATE_ID"]

    # Zero everything to be sure
    with pt.table(ms, readonly=False) as table:
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

    for i, ds in enumerate(xds):
        where = where_clause(group_cols, [getattr(ds, c) for c in group_cols])
        query = "SELECT * FROM $ms %s %s" % (where, order)

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
