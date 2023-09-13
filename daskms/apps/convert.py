import ast
from argparse import ArgumentTypeError
from collections import defaultdict
import logging

import dask.array as da

from daskms.apps.application import Application
from daskms.apps.formats import TableFormat, CasaFormat
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

    def visit_Constant(self, node):
        return node.n


NONUNIFORM_SUBTABLES = ["SPECTRAL_WINDOW", "POLARIZATION", "FEED", "SOURCE"]


def _check_input_path(input: str):
    input_path = DaskMSStore(input)

    if not input_path.exists():
        raise ArgumentTypeError(f"{input} is an invalid path.")

    return input_path


def _check_output_path(output: str):
    return DaskMSStore(output)


def _check_exclude_columns(columns: str):
    if not columns:
        return {}

    outputs = defaultdict(set)

    for column in (c.strip() for c in columns.split(",")):
        bits = column.split("::")

        if len(bits) == 2:
            table, column = bits
        elif len(bits) == 1:
            table, column = "MAIN", bits[0]
        else:
            raise ArgumentTypeError(
                f"Excluded columns must be of the form "
                f"COLUMN or SUBTABLE::COLUMN. "
                f"Received {column}"
            )

        outputs[table].add(column)

    outputs = {
        table: "*" if "*" in columns else columns for table, columns in outputs.items()
    }

    if outputs.get("MAIN", "") == "*":
        raise ValueError("Excluding all columns in the MAIN table is not supported")

    return outputs


def parse_chunks(chunks: str):
    return ChunkTransformer().visit(ast.parse(chunks))


class Convert(Application):
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
        parser.add_argument("-o", "--output", type=_check_output_path, required=True)
        parser.add_argument(
            "-x",
            "--exclude",
            type=_check_exclude_columns,
            default="",
            help="Comma-separated list of columns to exclude. "
            "For example 'CORRECTED_DATA,"
            "SPECTRAL_WINDOW::EFFECTIVE_BW' "
            "will exclude CORRECTED_DATA "
            "from the main table and "
            "EFFECTIVE_BW from the SPECTRAL_WINDOW "
            "subtable. SPECTRAL_WINDOW::* will exclude "
            "the entire SPECTRAL_WINDOW subtable",
        )
        parser.add_argument(
            "-g",
            "--group-columns",
            type=Convert.col_converter,
            default="",
            help="Comma-separatred list of columns to group "
            "or partition the input dataset by. "
            "This defaults to the default "
            "for the underlying storage mechanism."
            "This is only supported when converting "
            "from casa format.",
        )
        parser.add_argument(
            "-i",
            "--index-columns",
            type=Convert.col_converter,
            default="",
            help="Columns to sort "
            "the input dataset by. "
            "This defaults to the default "
            "for the underlying storage mechanism."
            "This is only supported when converting "
            "from casa format.",
        )
        parser.add_argument(
            "--taql-where",
            default="",
            help="TAQL where clause. "
            "Only useable with CASA inputs. "
            "For example, to exclude auto-correlations "
            '"ANTENNA1 != ANTENNA2"',
        )
        parser.add_argument(
            "-f",
            "--format",
            choices=["ms", "casa", "zarr", "parquet"],
            default="zarr",
            help="Output format",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            default=False,
            help="Force overwrite of output",
        )
        parser.add_argument(
            "-c",
            "--chunks",
            default="{row: 10000}",
            help=(
                "chunking schema applied to each dataset "
                "e.g. {row: 1000, chan: 16, corr: 1}"
            ),
            type=parse_chunks,
        )

    def execute(self):
        import dask

        if self.args.output.exists():
            if self.args.force:
                self.args.output.rm(recursive=True)
            else:
                raise ValueError(
                    f"{self.args.output} exists. " f"Use --force to overwrite."
                )

        writes = self.convert_table(self.args)

        dask.compute(writes)

    def _expand_group_columns(self, datasets, args):
        if not args.group_columns:
            return datasets

        new_datasets = []

        for ds in datasets:
            # Remove grouping attribute and recreate grouping columns
            new_group_vars = {}
            row_chunks = ds.chunks["row"]
            row_dims = ds.dims["row"]
            attrs = ds.attrs

            for column in args.group_columns:
                value = attrs.pop(column)
                group_column = da.full(row_dims, value, chunks=row_chunks)
                new_group_vars[column] = (("row",), group_column)

            new_ds = ds.assign_attrs(attrs).assign(**new_group_vars)
            new_datasets.append(new_ds)

        return new_datasets

    def convert_table(self, args):
        in_fmt = TableFormat.from_store(args.input)
        out_fmt = TableFormat.from_type(args.format)

        reader = in_fmt.reader(
            group_columns=args.group_columns,
            index_columns=args.index_columns,
            taql_where=args.taql_where,
        )
        writer = out_fmt.writer()

        datasets = reader(args.input, chunks=args.chunks)

        if exclude_columns := args.exclude.get("MAIN", False):
            datasets = [
                ds.drop_vars(exclude_columns, errors="ignore") for ds in datasets
            ]

        if isinstance(out_fmt, CasaFormat):
            # Reintroduce any grouping columns
            datasets = self._expand_group_columns(datasets, args)

        log.info("Input: '%s' %s", in_fmt, str(args.input))
        log.info("Output: '%s' %s", out_fmt, str(args.output))

        writes = [writer(datasets, args.output)]

        # Now do the subtables
        for table in list(in_fmt.subtables):
            if (
                table in {"SORTED_TABLE", "SOURCE"}
                or args.exclude.get(table, "") == "*"
            ):
                log.warning(f"Ignoring {table}")
                continue

            in_store = args.input.subtable_store(table)
            in_fmt = TableFormat.from_store(in_store)
            out_store = args.output.subtable_store(table)
            out_fmt = TableFormat.from_type(args.format, subtable=table)

            reader = in_fmt.reader()
            writer = out_fmt.writer()

            if isinstance(in_fmt, CasaFormat) and table in NONUNIFORM_SUBTABLES:
                datasets = reader(in_store, group_cols="__row__")
            else:
                datasets = reader(in_store)

            if exclude_columns := args.exclude.get(table, False):
                datasets = [
                    ds.drop_vars(exclude_columns, errors="ignore") for ds in datasets
                ]

            writes.append(writer(datasets, out_store))

        return writes
