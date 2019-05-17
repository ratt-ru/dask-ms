from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask
import numpy as np
import pytest
import pyrap.tables as pt
from xarrayms.xarray_ms import (xds_from_table,
                                orderby_clause,
                                where_clause)


def group_cols_str(group_cols):
    return "group_cols=%s" % group_cols


def index_cols_str(index_cols):
    return "index_cols=%s" % index_cols


@pytest.mark.parametrize('group_cols', [
    [],
    ["FIELD_ID", "DATA_DESC_ID"],
    ["DATA_DESC_ID"],
    ["DATA_DESC_ID", "SCAN_NUMBER"]],
    ids=group_cols_str)
@pytest.mark.parametrize('index_cols', [
    ["ANTENNA2", "ANTENNA1", "TIME"],
    ["TIME", "ANTENNA1", "ANTENNA2"],
    ["ANTENNA1", "ANTENNA2", "TIME"]],
    ids=index_cols_str)
def test_distributed(ms, group_cols, index_cols):
    distributed = pytest.importorskip("distributed")

    select_cols = index_cols
    order = orderby_clause(index_cols)

    with distributed.LocalCluster(processes=False) as cluster:
        with distributed.Client(cluster) as client:
            xds = xds_from_table(ms, index_cols=index_cols,
                                 group_cols=group_cols)

            with pt.table(ms, lockoptions='auto', ack=False) as T:  # noqa
                for ds in xds:
                    group_col_values = [getattr(ds, c) for c in group_cols]
                    where = where_clause(group_cols, group_col_values)
                    query = "SELECT * FROM $T %s %s" % (where, order)

                    with pt.taql(query) as Q:
                        for c in select_cols:
                            np_data = Q.getcol(c)
                            dask_data = getattr(ds, c).data.compute()
                            assert np.all(np_data == dask_data)
