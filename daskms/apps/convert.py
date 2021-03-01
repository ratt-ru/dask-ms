from argparse import ArgumentTypeError
from pathlib import Path
import shutil

from daskms.apps.application import Application


def _check_input_path(input: str):
    input_path = Path(input)
    if not input_path.exists():
        raise ArgumentTypeError(f"{input} is an invalid path.")

    return input_path


class Convert(Application):
    def __init__(self, args, log):
        self.log = log
        self.args = args

    @classmethod
    def setup_parser(cls, parser):
        parser.add_argument("input", type=_check_input_path)
        parser.add_argument("-o", "--output", type=Path)
        parser.add_argument("-f", "--format",
                            choices=["casa", "zarr", "parquet"],
                            default="zarr",
                            help="Output format")
        parser.add_argument("--force",
                            action="store_true",
                            default=False,
                            help="Force overwrite of output")

    def execute(self):
        from daskms import xds_from_ms
        import dask

        if self.args.output.exists():
            if self.args.force:
                shutil.rmtree(self.args.output)
            else:
                raise ValueError(f"{self.args.output} exists. "
                                 f"Use --force to overwrite.")

        datasets = xds_from_ms(self.args.input)

        if self.args.format == "casa":
            from daskms import xds_to_table
            from functools import partial
            writer = partial(xds_to_table, descriptor="ms")
            datasets = [ds.drop_vars("ROWID", errors="ignore") for ds in datasets]
        elif self.args.format == "zarr":
            from daskms.experimental.zarr import xds_to_zarr
            writer = xds_to_zarr
        elif self.args.format == "parquet":
            from daskms.experimental.arrow.writes import xds_to_parquet
            writer = xds_to_parquet
        else:
            raise ValueError(f"Unknown format {self.args.format}")


        writes = writer(datasets, str(self.args.output))
        dask.compute(writes)
