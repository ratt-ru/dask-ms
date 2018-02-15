"""
Tests that data read from xarrayms for the default
partioning and indexing scheme matches a taql query
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

logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.WARN)

def create_parser():
    DEFAULT_COLS = ["TIME", "ANTENNA1", "ANTENNA2", "UVW"]

    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-c", "--columns", default=", ".join(DEFAULT_COLS))

    return p

args = create_parser().parse_args()

with pt.table(args.ms) as table:
    if args.columns == "all":
            columns = set(table.colnames())
    else:
        columns = set([c.strip().upper() for c in args.columns.split(',')])

    for ds in xds_from_ms(args.ms, columns=columns):
        data_desc_id = ds.attrs['DATA_DESC_ID']
        field_id = ds.attrs['FIELD_ID']

        ds_cols = set(ds.data_vars.keys())
        cmp_cols = columns.difference(['DATA_DESC_ID', 'FIELD_ID'])

        if not ds_cols == cmp_cols:
            missing = ds_cols.symmetric_difference(cmp_cols)
            logging.warn("The following columns were requested "
                             "but not present on the dataset. "
                             "It will not be compared: %s",
                             list(missing))

            cmp_cols = cmp_cols - missing


        # Select data from the relevant data from the MS
        query = ("SELECT * FROM $table WHERE DATA_DESC_ID=%d AND FIELD_ID=%d" %
                                                    (data_desc_id, field_id))

        # Compare
        with pt.taql(query) as Q:
            for c in cmp_cols:
                assert np.all(getattr(ds, c).data.compute() == Q.getcol(c))

