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
                                xds_from_table,
                                xds_to_table,
                                orderby_clause,
                                where_clause)

from xarrayms.known_table_schemas import MS_SCHEMA, ColumnSchema


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
        xds = list(xds_from_ms(ms, columns=index_cols,  # noqa
                               group_cols=group_cols,
                               index_cols=index_cols,
                               min_frag_level=False,
                               chunks={"row": 1e9}))

        assert patch_fn.called_once_with(min_frag_level=False, sort_dir="read")


@pytest.mark.parametrize('group_cols', [
    ["DATA_DESC_ID"]])
@pytest.mark.parametrize('index_cols', [
    ["TIME"]])
def test_table_schema(ms, group_cols, index_cols):
    # Test default MS Schema
    xds = list(xds_from_ms(ms, columns=["DATA"],
                           group_cols=group_cols,
                           index_cols=index_cols,
                           chunks={"row": 1e9}))

    assert xds[0].DATA.dims == ("row", "chan", "corr")

    # Test custom column schema specified by ColumnSchema objet
    table_schema = MS_SCHEMA.copy()
    table_schema['DATA'] = ColumnSchema(("my-chan", "my-corr"))

    xds = list(xds_from_ms(ms, columns=["DATA"],
                           group_cols=group_cols,
                           index_cols=index_cols,
                           table_schema=table_schema,
                           chunks={"row": 1e9}))

    assert xds[0].DATA.dims == ("row", "my-chan", "my-corr")

    # Test custom column schema specified by tuple object
    table_schema['DATA'] = ("my-chan", "my-corr")

    xds = list(xds_from_ms(ms, columns=["DATA"],
                           group_cols=group_cols,
                           index_cols=index_cols,
                           table_schema=table_schema,
                           chunks={"row": 1e9}))

    assert xds[0].DATA.dims == ("row", "my-chan", "my-corr")

    table_schema = {"DATA": ("my-chan", "my-corr")}

    xds = list(xds_from_ms(ms, columns=["DATA"],
                           group_cols=group_cols,
                           index_cols=index_cols,
                           table_schema=["MS", table_schema],
                           chunks={"row": 1e9}))

    assert xds[0].DATA.dims == ("row", "my-chan", "my-corr")


@pytest.mark.parametrize('group_cols', [
    ["DATA_DESC_ID"]])
@pytest.mark.parametrize('index_cols', [
    ["TIME"]])
def test_table_kwargs(ms, group_cols, index_cols):
    # TODO(sjperkins)
    # This really just test failure of table_kwargs,
    # come up with some testing modification of the table
    # object in the graph itself. lockoptions is not currently
    # possible because we override it.

    # Fail if we pass readonly=False table_kwargs
    with pytest.raises(ValueError) as ex:
        xds = list(xds_from_table(ms, table_kwargs={'readonly': False,
                                                    'ack': False}))

    expected = "'readonly=False' in xds_from_table table_kwargs"

    assert expected in str(ex.value)

    # Succeeds on normal
    xds = list(xds_from_table(ms, table_kwargs={'ack': False}))

    # Fail if we pass readonly=True table_kwargs
    with pytest.raises(ValueError) as ex:
        xds_to_table(xds[0], ms, "TIME",  table_kwargs={'readonly': True,
                                                        'ack': False})

    expected = "'readonly=True' in xds_to_table table_kwargs"

    assert expected in str(ex.value)


@pytest.mark.parametrize('index_cols', [
    ["TIME", "ANTENNA1", "ANTENNA2"]])
def test_taql_where(ms, index_cols):
    # three cases test here, corresponding to the
    # if-elif-else ladder in xds_from_table

    # No group_cols case
    xds = list(xds_from_table(ms, taql_where="FIELD_ID >= 0 AND FIELD_ID < 2",
                              columns=["FIELD_ID"],
                              table_kwargs={'ack': False}))

    assert len(xds) == 1
    assert (xds[0].FIELD_ID.data.compute() == [0, 0, 0, 1, 1, 1, 1]).all()

    # Group columns case
    xds = list(xds_from_table(ms, taql_where="FIELD_ID >= 0 AND FIELD_ID < 2",
                              group_cols=["DATA_DESC_ID", "SCAN_NUMBER"],
                              columns=["FIELD_ID"],
                              table_kwargs={'ack': False}))

    assert len(xds) == 2

    # Check group id's
    assert xds[0].DATA_DESC_ID == 0 and xds[0].SCAN_NUMBER == 0
    assert xds[1].DATA_DESC_ID == 0 and xds[1].SCAN_NUMBER == 1

    # Check field id's in each group
    assert np.all(xds[0].FIELD_ID.data.compute() == [0, 0, 1, 1])
    assert np.all(xds[1].FIELD_ID.data.compute() == [0, 1, 1])

    # Group on each row
    xds = list(xds_from_table(ms, taql_where="FIELD_ID >= 0 AND FIELD_ID < 2",
                              group_cols=["__row__"],
                              columns=["FIELD_ID"],
                              table_kwargs={'ack': False}))

    assert len(xds) == 7

    xds = [ds.compute() for ds in xds]

    assert np.all(xds[0].FIELD_ID == 0)
    assert np.all(xds[1].FIELD_ID == 0)
    assert np.all(xds[2].FIELD_ID == 0)
    assert np.all(xds[3].FIELD_ID == 1)
    assert np.all(xds[4].FIELD_ID == 1)
    assert np.all(xds[5].FIELD_ID == 1)
    assert np.all(xds[6].FIELD_ID == 1)


def _proc_map_fn(args):
    ms, i = args
    xds = list(xds_from_ms(ms, columns=["STATE_ID"],
                           group_cols=["FIELD_ID"],
                           table_kwargs={'ack': False}))
    xds[i] = xds[i].assign(STATE_ID=xds[i].STATE_ID + i)
    write = xds_to_table(xds[i], ms, ["STATE_ID"], table_kwargs={'ack': False})
    write.compute(scheduler='sync')
    return True


@pytest.mark.parametrize("nprocs", [3])
def test_multiprocess_table(ms, nprocs):
    from multiprocessing import Pool

    pool = Pool(nprocs)
    assert all(pool.map(_proc_map_fn, [tuple((ms, i)) for i in range(nprocs)]))
