import abc
from argparse import ArgumentTypeError
from functools import partial
import logging
from pathlib import Path
import shutil

from daskms.apps.application import Application
from daskms.utils import dataset_type

log = logging.getLogger(__name__)


class TableFormat(abc.ABC):
    @abc.abstractproperty
    def version(self):
        raise NotImplementedError

    @abc.abstractproperty
    def subtables(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def reader(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def writer(self):
        raise NotImplementedError

    @staticmethod
    def from_path(path):
        if not isinstance(path, Path):
            path = Path(path)

        typ = dataset_type(path)

        if typ == "casa":
            from daskms.table_proxy import TableProxy
            import pyrap.tables as pt
            table_proxy = TableProxy(pt.table, str(path),
                                     readonly=True, ack=False)
            keywords = table_proxy.getkeywords().result()

            try:
                version = str(keywords["MS_VERSION"])
            except KeyError:
                typ = "plain"
                version = "<unspecified>"
            else:
                typ = "measurementset"

            subtables = CasaFormat.find_subtables(keywords)
            return CasaFormat(version, subtables, typ)
        elif typ == "zarr":
            subtables = ZarrFormat.find_subtables(path)
            return ZarrFormat("0.1", subtables)
        elif typ == "parquet":
            subtables = ParquetFormat.find_subtables(path)
            return ParquetFormat("0.1", subtables)
        else:
            raise ValueError(f"Unexpected table type {typ}")

    @staticmethod
    def from_type(typ):
        if typ == "casa":
            return CasaFormat("<unspecified>", [], "plain")
        elif typ == "zarr":
            return ZarrFormat("0.1", [])
        elif typ == "parquet":
            return ParquetFormat("0.1", [])
        else:
            raise ValueError(f"Unexpected table type {typ}")


class BaseTableFormat(TableFormat):
    def __init__(self, version):
        self._version = version

    @property
    def version(self):
        return self._version


class CasaFormat(BaseTableFormat):
    TABLE_PREFIX = "Table: "
    TABLE_TYPES = set(["plain", "measurementset"])

    def __init__(self, version, subtables, type="plain"):
        super().__init__(version)

        if type not in self.TABLE_TYPES:
            raise ValueError(f"{type} is not in {self.TABLE_TYPES}")

        self._subtables = subtables
        self._type = type

    @classmethod
    def find_subtables(cls, keywords):
        return [k for k, v in keywords.items() if cls.is_subtable(v)]

    @classmethod
    def is_subtable(cls, keyword):
        if not isinstance(keyword, str):
            return False

        if not keyword.startswith(cls.TABLE_PREFIX):
            return False

        path = Path(keyword[len(cls.TABLE_PREFIX):])
        return (path.exists() and
                path.is_dir() and
                (path / "table.dat").exists())

    def is_measurement_set(self):
        return self._type == "measurementset"

    def reader(self):
        if self.is_measurement_set():
            from daskms import xds_from_ms
            return xds_from_ms
        else:
            from daskms import xds_from_table
            return xds_from_table

    def writer(self):
        from daskms import xds_to_table

        if self.is_measurement_set():
            return partial(xds_to_table, descriptor="ms")
        else:
            return xds_to_table

    @property
    def subtables(self):
        return self._subtables

    def __str__(self):
        return "casa" if not self.is_measurement_set() else self._type


class ZarrFormat(BaseTableFormat):
    def __init__(self, version, subtables):
        self._subtables = subtables

    @classmethod
    def find_subtables(cls, path):
        return [p.relative_to(path) for p in path.iterdir()
                if p.is_dir()
                and p.name != "MAIN"
                and (p / ".zgroup").exists()]

    @property
    def subtables(self):
        return self._subtables

    def reader(self):
        from daskms.experimental.zarr import xds_from_zarr
        return xds_from_zarr

    def writer(self):
        from daskms.experimental.zarr import xds_to_zarr
        return xds_to_zarr

    def __str__(self):
        return "zarr"


class ParquetFormat(BaseTableFormat):
    def __init__(self, version, subtables):
        super().__init__(version)
        self._subtables = subtables

    @classmethod
    def find_subtables(cls, path):
        return [p.relative_to(path) for p in path.iterdir()
                if p.is_dir() and p.name != "MAIN"]

    @property
    def subtables(self):
        return self._subtables

    def reader(self):
        from daskms.experimental.arrow.reads import xds_from_parquet
        return xds_from_parquet

    def writer(self):
        from daskms.experimental.arrow.writes import xds_to_parquet
        return xds_to_parquet

    def __str__(self):
        return "parquet"


def convert_table(input_path, output_path, output_format):
    in_fmt = TableFormat.from_path(input_path)
    out_fmt = TableFormat.from_type(output_format)

    reader = in_fmt.reader()
    writer = out_fmt.writer()

    datasets = reader(input_path)

    if isinstance(in_fmt, CasaFormat):
        # Drop any ROWID columns
        datasets = [ds.drop_vars("ROWID", errors="ignore")
                    for ds in datasets]

    log.info("Input: '%s' %s", in_fmt, str(input_path))
    log.info("Output: '%s' %s", out_fmt, str(output_path))

    writes = [writer(datasets, str(output_path))]

    # Now do the subtables
    for table in list(in_fmt.subtables):
        if table == "SORTED_TABLE":
            log.warning(f"Ignoring {table}")
            continue

        in_path = input_path.parent / "::".join((input_path.name, table))
        in_fmt = TableFormat.from_path(in_path)
        out_path = output_path.parent / "::".join((output_path.name, table))
        out_fmt = TableFormat.from_type(output_format)

        reader = in_fmt.reader()
        writer = out_fmt.writer()

        if isinstance(in_fmt, CasaFormat):
            # Drop any ROWID columns
            datasets = [ds.drop_vars("ROWID", errors="ignore")
                        for ds in reader(in_path, group_cols="__row__")]
        else:
            datasets = reader(in_path)

        writes.append(writer(datasets, str(out_path)))

    return writes


def _check_input_path(input: str):
    input_path = Path(input)

    parts = input_path.name.split("::", 1)

    if len(parts) == 1:
        check_path = input_path
    elif len(parts) == 2:
        check_path = input_path.parent / parts[0]
    else:
        raise RuntimeError("len(parts) not in (1, 2)")

    if not check_path.exists():
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

        writes = convert_table(self.args.input,
                               self.args.output,
                               self.args.format)

        dask.compute(writes)
