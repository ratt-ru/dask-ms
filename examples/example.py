import argparse
from contextlib import contextmanager
import time

import dask.array as da
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
    ds = xarray_ms.xds_from_table(args.ms, chunks=100000, table_schema="MS")

    start = time.clock()
    inv_flag = da.logical_not(ds.flag.data)

    ds['flag'] = (ds.flag.dims, inv_flag)

    xarray_ms.xds_to_table(ds, "FLAG").compute()

    print "Time handling '%s'" % (time.clock() - start)

    print ds


