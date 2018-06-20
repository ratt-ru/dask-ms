from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import os

import dask.array as da
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
    msdir = tmpdir_factory.mktemp("msdir")
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

    return fn


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
                           index_cols=index_cols))

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
                           index_cols=index_cols))

    for i, ds in enumerate(xds):
        state = np.full(ds.dims['row'], i, dtype=np.int32)
        state = da.from_array(state, chunks=ds.chunks['row'])
        state = xr.DataArray(state, dims=['row'])
        ds = ds.assign(STATE_ID=state)
        xds_to_table(ds, ms, "STATE_ID").compute()

    xds = list(xds_from_ms(ms, columns=select_cols,
                           group_cols=group_cols,
                           index_cols=index_cols))

    # Check that state has been correctly written
    for i, ds in enumerate(xds):
        assert np.all(ds.STATE_ID.data.compute() == i)
