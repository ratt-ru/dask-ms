"""
Tests that data written by xarrayms for the default
grouping and indexing scheme matches a taql query
and getcol via pyrap.tables.

Currently needs a Measurement Set to run.
simms_ would be a good choice for generating one.


.. _simms: https://github.com/radio-astro/simms
"""

import argparse
import logging

import dask.array as da
import numpy as np
import pyrap.tables as pt
import xarray as xr

from xarrayms import xds_from_ms, xds_to_table

from xarrayms.xarray_ms import (_DEFAULT_GROUP_COLUMNS,
                                _DEFAULT_INDEX_COLUMNS)

logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.WARN)


def _split_column_str(col_str):
    cols = [c.strip().upper() for c in col_str.split(",")]
    return [c for c in cols if c]


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-gc", "--group-columns", type=_split_column_str,
                   default=",".join(_DEFAULT_GROUP_COLUMNS))
    p.add_argument("-ic", "--index-columns", type=_split_column_str,
                   default=",".join(_DEFAULT_INDEX_COLUMNS))
    # STATE_ID is relatively innocuous
    p.add_argument("-c", "--column", default="STATE_ID")

    return p


args = create_parser().parse_args()

with pt.table(args.ms) as table:
    index_cols = args.index_columns
    group_cols = args.group_columns

    for ds in xds_from_ms(args.ms, columns=[args.column],
                          group_cols=group_cols,
                          index_cols=index_cols):

        xrcol = getattr(ds, args.column)

        # Persist original as in memory dask array
        original = xrcol.data.persist()

        try:
            # Write flipped arange to the table
            arange = da.arange(xrcol.size, chunks=np.product(original.chunks))
            arange = da.flip(arange, 0)
            arange = da.reshape(arange, xrcol.shape)

            new_xda = xr.DataArray(arange, dims=xrcol.dims)
            nds = ds.assign(**{args.column: new_xda})
            write = xds_to_table(nds, args.ms, args.column).compute()

            # Check that we get the right thing back.
            # Note that we can use the
            assert np.all(arange.compute() == xrcol.data.compute())
        finally:
            # Write original data back to the table
            orig_xda = xr.DataArray(original, dims=xrcol.dims)
            nds = ds.assign(**{args.column: orig_xda})
            write = xds_to_table(nds, args.ms, args.column).compute()

            assert np.all(original.compute() == xrcol.data.compute())
