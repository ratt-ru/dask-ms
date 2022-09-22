import argparse

import dask
from daskms import xds_from_table, xds_from_ms, xds_to_table, TableProxy

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

    # Create short names mapped to the full table path
    table_name = {
        short: "::".join((args.ms, full))
        for short, full in [
            ("antenna", "ANTENNA"),
            ("ddid", "DATA_DESCRIPTION"),
            ("spw", "SPECTRAL_WINDOW"),
            ("pol", "POLARIZATION"),
            ("field", "FIELD"),
        ]
    }

    with scheduler_context(args):
        # Get datasets from the main MS
        # partition by FIELD_ID and DATA_DESC_ID
        # and sorted by TIME
        datasets = xds_from_ms(
            args.ms, group_cols=("FIELD_ID", "DATA_DESC_ID"), index_cols="TIME"
        )

        # Get the antenna dataset
        ant_ds = list(xds_from_table(table_name["antenna"]))
        assert len(ant_ds) == 1
        ant_ds = ant_ds[0].rename({"row": "antenna"})

        # Get datasets for DATA_DESCRIPTION, SPECTRAL_WINDOW
        # POLARIZATION and FIELD, partitioned by row
        ddid_ds = list(xds_from_table(table_name["ddid"], group_cols="__row__"))
        spwds = list(xds_from_table(table_name["spw"], group_cols="__row__"))
        pds = list(xds_from_table(table_name["pol"], group_cols="__row__"))
        field_ds = list(xds_from_table(table_name["field"], group_cols="__row__"))

        # For each partitioned dataset from the main MS,
        # assign additional arrays from the FIELD, SPECTRAL_WINDOW
        # and POLARISATION subtables
        for ms_ds in datasets:
            # Look up the Spectral Window and Polarization
            # datasets, given the Data Descriptor ID
            field = field_ds[ms_ds.attrs["FIELD_ID"]]
            ddid = ddid_ds[ms_ds.attrs["DATA_DESC_ID"]]
            spw = spwds[ddid.SPECTRAL_WINDOW_ID.values[0]]
            pol = pds[ddid.POLARIZATION_ID.values[0]]

            ms_ds = ms_ds.assign(
                ANTENNA_POSITION=ant_ds.POSITION,
                PHASE_CENTRE=field.PHASE_DIR[0],
                FREQUENCY=spw.CHAN_FREQ[0],
                CORRELATION_TYPE=pol.CORR_TYPE[0],
                CORRELATION_PRODUCT=pol.CORR_PRODUCT[0],
            )

            print(ms_ds)
