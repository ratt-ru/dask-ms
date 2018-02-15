"""
Tests that data read from xarrayms for the default
partioning and indexing scheme matches a taql query
and getcol via pyrap.tables
"""

import argparse

import numpy as np
import pyrap.tables as pt

from xarrayms import xds_from_ms

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-c", "--columns", default="UVW,DATA")
    return p

args = create_parser().parse_args()

columns = [c.strip().upper() for c in args.columns.split(',')]

for ds in xds_from_ms(args.ms):
    data_desc_id = ds.attrs['DATA_DESC_ID']
    field_id = ds.attrs['FIELD_ID']

    with pt.table(args.ms) as M:
        query = ("SELECT * FROM $M "
                "WHERE DATA_DESC_ID=%d "
                "AND FIELD_ID=%d" % (data_desc_id, field_id))

        with pt.taql(query) as Q:
            # Compre
            for c in columns:
                assert np.all(getattr(ds, c).data.compute() == Q.getcol(c))

