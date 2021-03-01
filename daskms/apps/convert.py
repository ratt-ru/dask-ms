from argparse import ArgumentTypeError
from pathlib import Path
import shutil

from daskms.apps.application import Application
from daskms.utils import dataset_type


def _check_input_path(input: str):
    input_path = Path(input)
    if not input_path.exists():
        raise ArgumentTypeError(f"{input} is an invalid path.")

    return input_path


class Convert(Application):
    TABLE_KEYWORD_PREFIX = "Table: "

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
        import dask

        if self.args.output.exists():
            if self.args.force:
                shutil.rmtree(self.args.output)
            else:
                raise ValueError(f"{self.args.output} exists. "
                                 f"Use --force to overwrite.")

        input_format = dataset_type(self.args.input)

        if input_format == "casa":
            from daskms.table_proxy import TableProxy
            import pyrap.tables as pt

            table_proxy = TableProxy(pt.table, str(self.args.input))
            keywords = table_proxy.getkeywords().result()

            if "MS_VERSION" in keywords:
                from daskms import xds_from_ms
                reader = xds_from_ms
            else:
                from daskms import xds_from_table
                reader = xds_from_table

            n = len(self.TABLE_KEYWORD_PREFIX)
            subtables = {k: v[n:] for k, v in keywords.items() if
                         isinstance(v, str) and
                         v.startswith(self.TABLE_KEYWORD_PREFIX)}
            from pprint import pprint
            pprint(subtables)
        elif input_format == "zarr":
            from daskms.experimental.zarr import xds_from_zarr
            reader = xds_from_zarr
        elif input_format == "parquet":
            from daskms.experimental.arrow.reads import xds_from_parquet
            reader = xds_from_parquet
        else:
            raise RuntimeError(f"Invalid input format {input_format}")

        datasets = reader(self.args.input)

        if self.args.format == "casa":
            from daskms import xds_to_table
            writer = xds_to_table

            if "MS_VERSION" in keywords:
                from functools import partial
                writer = partial(writer, descriptor="ms")

            # Strip out ROWID's so that a new Table is created
            datasets = [ds.drop_vars("ROWID", errors="ignore")
                        for ds in datasets]
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
