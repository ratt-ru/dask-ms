from argparse import ArgumentTypeError
from pathlib import Path

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
                            default="zarr")

    def execute(self):
        from daskms import xds_from_ms
        import dask

        if self.args.format == "casa":
            from daskms import xds_to_table
            writer = xds_to_table
        elif self.args.format == "zarr":
            from daskms.experimental.zarr import xds_to_zarr
            writer = xds_to_zarr
        elif self.args.format == "parquet":
            from daskms.experimental.arrow.writes import xds_to_parquet
            writer = xds_to_parquet
        else:
            raise ValueError(f"Unknown format {self.args.format}")

        datasets = xds_from_ms(self.args.input)
        writes = writer(datasets, str(self.args.output))
        dask.compute(writes)
