"""
Tests that data written by xarrayms for the default
partioning and indexing scheme matches a taql query
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

logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.WARN)

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    # STATE_ID is relatively innocuous
    p.add_argument("-c", "--column", default="STATE_ID")

    return p

args = create_parser().parse_args()

with pt.table(args.ms) as table:
    for ds in xds_from_ms(args.ms, columns=[args.column]):
        data_desc_id = ds.attrs['DATA_DESC_ID']
        field_id = ds.attrs['FIELD_ID']
        row_chunks = ds.chunks["row"]

        xrcol = getattr(ds, args.column)

        # Compute original, then save it as a dask array
        original = da.from_array(xrcol.data.compute(), chunks=row_chunks)

        try:
            # Write flipped arange to the table
            arange = da.arange(xrcol.size, chunks=row_chunks)
            arange = da.flip(arange, 0)

            nds = ds.assign(**{args.column: xr.DataArray(arange, dims="row")})
            write = xds_to_table(nds, args.ms, args.column).compute()

            # Check that we get the right thing back.
            # Note that we can use the
            assert np.all(arange.compute() == xrcol.data.compute())
        finally:
            # Write original data back to the table
            nds = ds.assign(**{args.column: xr.DataArray(original, dims="row")})
            write = xds_to_table(nds, args.ms, args.column).compute()

            assert np.all(original.compute() == ds.STATE_ID.data.compute())
