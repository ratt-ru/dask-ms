import argparse
from contextlib import contextmanager
import time

import dask.array as da
import xarray.ufuncs as xru
import xarray_ms

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("--dist", action="store_true", default=False)
    return p

@contextmanager
def scheduler_context(args):
    """
    Runs MS local distributed cluster if requested
    """
    try:
        if args.dist == True:
            log.info("Starting distributed dask")
            from distributed import Client, LocalCluster
            cluster = LocalCluster(processes=True)
            client = Client(cluster.scheduler_address)
        yield "OK"
    finally:
        if args.dist == True:
            log.info("Shutting down distributed dask")
            client.close()
            cluster.close()


args = create_parser().parse_args()

with scheduler_context(args):
    start = time.clock()

    # Create dataset from Measurement Set
    ds = xarray_ms.xds_from_ms(args.ms, chunks=100000)

    # Create dataset with flag inverted
    ds = ds.assign(flag=xru.logical_not(ds.flag))
    assert isinstance(ds.flag.data, da.Array)

    # Write the flag column to the Measurement Set
    xarray_ms.xds_to_table(ds, "FLAG").compute()

    print "Flag inversion time '%s'" % (time.clock() - start)

    #Load the ANTENNA table
    ant_ds = xarray_ms.xds_from_table('::'.join((args.ms, "ANTENNA")),
                                    chunks=100000, table_schema="ANTENNA")

    spw_ds = xarray_ms.xds_from_table('::'.join((args.ms, "SPECTRAL_WINDOW")),
                                    chunks=100000, table_schema="SPECTRAL_WINDOW")

    print ant_ds
    print spw_ds
    print ds

