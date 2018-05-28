"""
Tests that data read from xarrayms for the default
grouping and indexing scheme matches a taql query
and getcol via pyrap.tables.

Currently needs a Measurement Set to run.
simms_ would be a good choice for generating one.


.. _simms: https://github.com/radio-astro/simms
"""

import argparse
import logging

import numpy as np
import pyrap.tables as pt

from xarrayms import xds_from_ms
from xarrayms.xarray_ms import (_DEFAULT_GROUP_COLUMNS,
                                _DEFAULT_INDEX_COLUMNS,
                                select_clause,
                                orderby_clause,
                                groupby_clause,
                                where_clause)

logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.WARN)


def _split_column_str(col_str):
    cols = [c.strip().upper() for c in col_str.split(",")]
    return [c for c in cols if c]


def create_parser():
    DEFAULT_SELECT = ["TIME", "ANTENNA1", "ANTENNA2", "UVW"]

    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-gc", "--group-columns", type=_split_column_str,
                   default=",".join(_DEFAULT_GROUP_COLUMNS))
    p.add_argument("-sc", "--select-columns", type=_split_column_str,
                   default=",".join(DEFAULT_SELECT))
    p.add_argument("-ic", "--index-columns", type=_split_column_str,
                   default=",".join(_DEFAULT_INDEX_COLUMNS))

    return p


args = create_parser().parse_args()

with pt.table(args.ms) as table:
    index_cols = args.index_columns
    group_cols = args.group_columns

    columns = set(table.colnames() if args.select_columns == "all"
                  else args.select_columns)
    order = orderby_clause(index_cols)

    for ds in xds_from_ms(args.ms, columns=columns,
                          group_cols=group_cols,
                          index_cols=index_cols):

        ds_cols = set(ds.data_vars.keys())
        cmp_cols = columns.difference(group_cols)

        if not ds_cols == cmp_cols:
            missing = ds_cols.symmetric_difference(cmp_cols)
            logging.warn("The following columns were requested "
                         "but not present on the dataset. "
                         "It will not be compared: %s",
                         list(missing))

            cmp_cols = cmp_cols - missing

        where = where_clause(group_cols, [getattr(ds, c) for c in group_cols])

        # Select data from the relevant data from the MS
        with pt.taql("SELECT * FROM $table %s %s" % (where, order)) as Q:
            for c in cmp_cols:
                dask_data = getattr(ds, c).data.compute()
                np_data = Q.getcol(c)
                assert np.all(dask_data == np_data)
