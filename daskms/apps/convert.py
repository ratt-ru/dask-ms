import abc
import ast
from argparse import ArgumentTypeError
from functools import partial
import logging
from pathlib import Path
import shutil

from daskms.apps.application import Application
from daskms.fsspec_store import DaskMSStore

log = logging.getLogger(__name__)


class ChunkTransformer(ast.NodeTransformer):
    def visit_Module(self, node):
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
            raise ValueError("Module must contain a single expression")

        expr = node.body[0]

        if not isinstance(expr.value, ast.Dict):
            raise ValueError("Expression must contain a dictionary")

        return self.visit(expr).value

    def visit_Dict(self, node):
        keys = [self.visit(k) for k in node.keys]
        values = [self.visit(v) for v in node.values]
        return {k: v for k, v in zip(keys, values)}

    def visit_Name(self, node):
        return node.id

    def visit_Tuple(self, node):
        return tuple(self.visit(v) for v in node.elts)

    def visit_Num(self, node):
        return node.n


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
        store = DaskMSStore(path)
        typ = store.type()

        if typ == "casa":
            from daskms.table_proxy import TableProxy
            import pyrap.tables as pt
            table_proxy = TableProxy(pt.table, str(store.casa_path()),
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
            subtables = ZarrFormat.find_subtables(store)
            return ZarrFormat("0.1", subtables)
        elif typ == "parquet":
            subtables = ParquetFormat.find_subtables(store)
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

    def check_unused_kwargs(self, fn_name, **kwargs):
        if kwargs:
            raise NotImplementedError(f"The following kwargs: "
                                      f"{list(kwargs.keys())} "
                                      f"were not consumed by "
                                      f"{self.__class__.__name__}."
                                      f"{fn_name}(**kw)")


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

    def reader(self, **kw):
        try:
            group_cols = kw.pop("group_columns", None)
            index_cols = kw.pop("index_columns", None)

            if self.is_measurement_set():
                from daskms import xds_from_ms
                return partial(
                    xds_from_ms,
                    group_cols=group_cols,
                    index_cols=index_cols
                )
            else:
                from daskms import xds_from_table
                return xds_from_table
        finally:
            self.check_unused_kwargs("reader", **kw)

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
    def find_subtables(cls, store):
        paths = (p.relative_to(store.path) for p
                 in map(Path, store.subdirectories()))

        return [str(p) for p in paths if p.stem != "MAIN" and
                store.exists(str(p / ".zgroup"))]

    @property
    def subtables(self):
        return self._subtables

    def reader(self, **kw):
        group_columns = kw.pop("group_columns", False)
        index_columns = kw.pop("index_columns", False)

        if group_columns:
            raise ValueError("\"group_columns\" is not supported "
                             "for zarr inputs")
        if index_columns:
            raise ValueError("\"index_columns\" is not supported "
                             "for zarr inputs")
        try:
            from daskms.experimental.zarr import xds_from_zarr
            return xds_from_zarr
        finally:
            self.check_unused_kwargs("reader", **kw)

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
    def find_subtables(cls, store):
        paths = (p.relative_to(store.path) for p
                 in map(Path, store.subdirectories()))

        return [str(p) for p in paths if p.stem != "MAIN" and
                store.exists(str(p / ".zgroup"))]

    @property
    def subtables(self):
        return self._subtables

    def reader(self, **kw):
        group_columns = kw.pop("group_columns", False)
        index_columns = kw.pop("index_columns", False)

        if group_columns:
            raise ValueError("\"group_column\" is not supported "
                             "for parquet inputs")
        if index_columns:
            raise ValueError("\"index_columns\" is not supported "
                             "for parquet inputs")

        try:
            from daskms.experimental.arrow.reads import xds_from_parquet
            return xds_from_parquet
        finally:
            self.check_unused_kwargs("reader", **kw)

    def writer(self):
        from daskms.experimental.arrow.writes import xds_to_parquet
        return xds_to_parquet

    def __str__(self):
        return "parquet"


NONUNIFORM_SUBTABLES = ["SPECTRAL_WINDOW", "POLARIZATION", "FEED", "SOURCE"]


def convert_table(args):
    in_fmt = TableFormat.from_path(args.input)
    out_fmt = TableFormat.from_type(args.format)

    reader = in_fmt.reader(
        group_columns=args.group_columns,
        index_columns=args.index_columns
    )
    writer = out_fmt.writer()

    datasets = reader(args.input, chunks=args.chunks)

    if isinstance(in_fmt, CasaFormat):
        # Drop any ROWID columns
        datasets = [ds.drop_vars("ROWID", errors="ignore")
                    for ds in datasets]

    log.info("Input: '%s' %s", in_fmt, str(args.input))
    log.info("Output: '%s' %s", out_fmt, str(args.output))

    writes = [writer(datasets, str(args.output))]

    # Now do the subtables
    for table in list(in_fmt.subtables):
        if table == "SORTED_TABLE":
            log.warning(f"Ignoring {table}")
            continue

        in_path = args.input.parent / "::".join((args.input.name, table))
        in_fmt = TableFormat.from_path(in_path)
        out_path = args.output.parent / "::".join((args.output.name, table))
        out_fmt = TableFormat.from_type(args.format)

        reader = in_fmt.reader()
        writer = out_fmt.writer()

        if isinstance(in_fmt, CasaFormat) and table in NONUNIFORM_SUBTABLES:
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


def parse_chunks(chunks: str):
    return ChunkTransformer().visit(ast.parse(chunks))


class Convert(Application):
    TABLE_KEYWORD_PREFIX = "Table: "

    def __init__(self, args, log):
        self.log = log
        self.args = args

    @staticmethod
    def col_converter(columns):
        if not columns:
            return None

        return [c.strip() for c in columns.split(",")]

    @classmethod
    def setup_parser(cls, parser):
        parser.add_argument("input", type=_check_input_path)
        parser.add_argument("-o", "--output", type=Path)
        parser.add_argument("-g", "--group-columns",
                            type=Convert.col_converter,
                            default="",
                            help="Columns to group or partition "
                                 "the input dataset by. "
                                 "This defaults to the default "
                                 "for the underlying storage mechanism."
                                 "This is only supported when converting "
                                 "from casa format.")
        parser.add_argument("-i", "--index-columns",
                            type=Convert.col_converter,
                            default="",
                            help="Columns to sort "
                                 "the input dataset by. "
                                 "This defaults to the default "
                                 "for the underlying storage mechanism."
                                 "This is only supported when converting "
                                 "from casa format.")
        parser.add_argument("-f", "--format",
                            choices=["casa", "zarr", "parquet"],
                            default="zarr",
                            help="Output format")
        parser.add_argument("--force",
                            action="store_true",
                            default=False,
                            help="Force overwrite of output")
        parser.add_argument("-c", "--chunks",
                            default="{row: 10000}",
                            help=("chunking schema applied to each dataset "
                                  "e.g. {row: 1000, chan: 16, corr: 1}"),
                            type=parse_chunks)

    def execute(self):
        import dask

        if self.args.output.exists():
            if self.args.force:
                shutil.rmtree(self.args.output)
            else:
                raise ValueError(f"{self.args.output} exists. "
                                 f"Use --force to overwrite.")

        writes = convert_table(self.args)

        dask.compute(writes)
