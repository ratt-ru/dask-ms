import argparse
import logging
from contextlib import contextmanager

import dask

from xarray_ms import xds_from_table, xds_from_ms, xds_to_table, TableProxy

@contextmanager
def scheduler_context(args):
    """ Set the scheduler to use, based on the script arguments """

    import dask

    sched_type = None

    try:
        if args.scheduler in ("mt", "thread", "threaded", "threading"):
            import dask.threaded
            logging.info("Using multithreaded scheduler")
            dask.set_options(get=dask.threaded.get)
            sched_type = ("threaded",)
        elif args.scheduler in ("mp", "multiprocessing"):
            import dask.multiprocessing
            logging.info("Using multiprocessing scheduler")
            dask.set_options(get=dask.multiprocessing.get)
            sched_type = ("multiprocessing",)
        else:
            import distributed
            local_cluster = None

            if args.scheduler == "local":
                local_cluster = distributed.LocalCluster(processes=False)
                address = local_cluster.scheduler_address
            elif args.scheduler.startswith('tcp'):
                address = args.scheduler
            else:
                import json

                with open(args.scheduler, 'r') as f:
                    address = json.load(f)['address']

            logging.info("Using distributed scheduler "
                         "with address '{}'".format(address))
            client = distributed.Client(address)
            dask.set_options(get=client.get)
            client.restart()
            sched_type = ("distributed", client, local_cluster)

        yield
    except Exception:
        logging.exception("Error setting up scheduler", exc_info=True)

    finally:
        if sched_type[0] == "distributed":
            client, cluster = sched_type[1:3]

            if client:
                client.close()

            if cluster:
                cluster.close()


if __name__ == "__main__":

    from dask.diagnostics import Profiler, ProgressBar

    def create_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("ms")
        parser.add_argument("-c", "--chunks", default=10000, type=int)
        parser.add_argument("-s", "--scheduler", default="threaded")
        return parser

    args = create_parser().parse_args()

    with scheduler_context(args):


        # Create a dataset representing the entire antenna table
        ant_table =  '::'.join((args.ms, 'ANTENNA'))

        for ant_ds in xds_from_table(ant_table):
            print(ant_ds)
            print(dask.compute(ant_ds.NAME.data, ant_ds.POSITION.data, ant_ds.DISH_DIAMETER.data))


        # Create datasets representing each row of the spw table
        spw_table =  '::'.join((args.ms, 'SPECTRAL_WINDOW'))

        for spw_ds in xds_from_table(spw_table, part_cols="__row__"):
            print(spw_ds)
            print(spw_ds.NUM_CHAN.values)
            print(spw_ds.CHAN_FREQ.values)


        # Create datasets from a partioning of the MS
        datasets = list(xds_from_ms(args.ms, chunks={'row':args.chunks}))

        for ds in datasets:
            print(ds)

            # Try write the STATE_ID column back
            write = xds_to_table(ds, args.ms, 'STATE_ID')
            with ProgressBar(), Profiler() as prof:
                write.compute()

            # Profile
            prof.visualize(file_path="chunked.html")



