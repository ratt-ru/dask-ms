import argparse
from contextlib import ExitStack
import dask
from daskms import xds_from_table, xds_from_ms, xds_to_table, TableProxy

try:
    import bokeh
except ImportError:
    bokeh = None

from sched_context import scheduler_context

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
        ant_table = "::".join((args.ms, "ANTENNA"))

        for ant_ds in xds_from_table(ant_table):
            print(
                dask.compute(
                    ant_ds.NAME.data, ant_ds.POSITION.data, ant_ds.DISH_DIAMETER.data
                )
            )

        # Create datasets representing each row of the spw table
        spw_table = "::".join((args.ms, "SPECTRAL_WINDOW"))

        for spw_ds in xds_from_table(spw_table, group_cols="__row__"):
            print(spw_ds)
            print(spw_ds.NUM_CHAN.values)
            print(spw_ds.CHAN_FREQ.values)

        # Create datasets from a partioning of the MS
        datasets = list(xds_from_ms(args.ms, chunks={"row": args.chunks}))

        writes = []

        for ds in datasets:
            print(ds)

            # Try write the STATE_ID column back
            write = xds_to_table(ds, args.ms, "STATE_ID")
            writes.append(write)

        with ExitStack() as stack:
            stack.enter_context(ProgressBar())

            if bokeh is not None:
                prof = Profiler()
                stack.enter_context(prof)
            else:
                prof = None

            dask.compute(writes)

        # Profile
        if prof:
            prof.visualize(file_path="chunked.html")
