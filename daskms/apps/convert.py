from argparse import ArgumentTypeError
from collections import defaultdict
import logging

import click
import dask.array as da

from daskms.apps.formats import TableFormat, CasaFormat
from daskms.fsspec_store import DaskMSStore
from daskms.utils import parse_chunks_dict

log = logging.getLogger(__name__)


NONUNIFORM_SUBTABLES = ["SPECTRAL_WINDOW", "POLARIZATION", "FEED", "SOURCE"]


def _check_input_path(ctx, param, value):
    input_path = DaskMSStore(value)

    if not input_path.exists():
        raise ArgumentTypeError(f"{value} is an invalid path.")

    return input_path


def _check_output_path(ctx, param, value):
    return DaskMSStore(value)


def _check_exclude_columns(ctx, param, value):
    if not value:
        return {}

    outputs = defaultdict(set)

    for column in (c.strip() for c in value.split(",")):
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


def parse_chunks(ctx, param, value):
    return parse_chunks_dict(value)


def col_converter(ctx, param, value):
    if not value:
        return None

    return [c.strip() for c in value.split(",")]


@click.command
@click.pass_context
@click.argument("input", required=True, callback=_check_input_path)
@click.option("-o", "--output", callback=_check_output_path, required=True)
@click.option(
    "-x",
    "--exclude",
    default="",
    callback=_check_exclude_columns,
    help="Comma-separated list of columns to exclude. "
    "For example 'CORRECTED_DATA,"
    "SPECTRAL_WINDOW::EFFECTIVE_BW' "
    "will exclude CORRECTED_DATA "
    "from the main table and "
    "EFFECTIVE_BW from the SPECTRAL_WINDOW "
    "subtable. SPECTRAL_WINDOW::* will exclude "
    "the entire SPECTRAL_WINDOW subtable",
)
@click.option(
    "-g",
    "--group-columns",
    default="",
    callback=col_converter,
    help="Comma-separatred list of columns to group "
    "or partition the input dataset by. "
    "This defaults to the default "
    "for the underlying storage mechanism."
    "This is only supported when converting "
    "from casa format.",
)
@click.option(
    "-i",
    "--index-columns",
    default="",
    callback=col_converter,
    help="Columns to sort "
    "the input dataset by. "
    "This defaults to the default "
    "for the underlying storage mechanism."
    "This is only supported when converting "
    "from casa format.",
)
@click.option(
    "--taql-where",
    default="",
    help="TAQL where clause. "
    "Only useable with CASA inputs. "
    "For example, to exclude auto-correlations "
    '"ANTENNA1 != ANTENNA2"',
)
@click.option(
    "-f",
    "--format",
    type=click.Choice(["ms", "casa", "zarr", "parquet"]),
    default="zarr",
)
@click.option("--force", is_flag=True)
@click.option(
    "-c",
    "--chunks",
    default="{row: 10000}",
    callback=parse_chunks,
    help="chunking schema applied to each dataset "
    "e.g. {row: 1000, chan: 16, corr: 1}",
)
def convert(
    ctx,
    input,
    output,
    exclude,
    group_columns,
    index_columns,
    taql_where,
    format,
    force,
    chunks,
):
    converter = Convert(
        input,
        output,
        exclude,
        group_columns,
        index_columns,
        taql_where,
        format,
        force,
        chunks,
    )
    converter.execute()


class Convert:
    def __init__(
        self,
        input,
        output,
        exclude,
        group_columns,
        index_columns,
        taql_where,
        format,
        force,
        chunks,
    ):
        self.input = input
        self.output = output
        self.exclude = exclude
        self.group_columns = group_columns
        self.index_columns = index_columns
        self.taql_where = taql_where
        self.format = format
        self.force = force
        self.chunks = chunks

    def execute(self):
        import dask

        if self.output.exists():
            if self.force:
                self.output.rm(recursive=True)
            else:
                raise ValueError(f"{self.output} exists. " f"Use --force to overwrite.")

        writes = self.convert_table()

        dask.compute(writes)

    def _expand_group_columns(self, datasets):
        if not self.group_columns:
            return datasets

        new_datasets = []

        for ds in datasets:
            # Remove grouping attribute and recreate grouping columns
            new_group_vars = {}
            row_chunks = ds.chunks["row"]
            row_dims = ds.sizes["row"]
            attrs = ds.attrs

            for column in self.group_columns:
                value = attrs.pop(column)
                group_column = da.full(row_dims, value, chunks=row_chunks)
                new_group_vars[column] = (("row",), group_column)

            new_ds = ds.assign_attrs(attrs).assign(**new_group_vars)
            new_datasets.append(new_ds)

        return new_datasets

    def convert_table(self):
        in_fmt = TableFormat.from_store(self.input)
        out_fmt = TableFormat.from_type(self.format)

        reader = in_fmt.reader(
            group_columns=self.group_columns,
            index_columns=self.index_columns,
            taql_where=self.taql_where,
        )
        writer = out_fmt.writer()

        datasets = reader(self.input, chunks=self.chunks)

        if exclude_columns := self.exclude.get("MAIN", False):
            datasets = [
                ds.drop_vars(exclude_columns, errors="ignore") for ds in datasets
            ]

        if isinstance(out_fmt, CasaFormat):
            # Reintroduce any grouping columns
            datasets = self._expand_group_columns(datasets)

        log.info("Input: '%s' %s", in_fmt, str(self.input))
        log.info("Output: '%s' %s", out_fmt, str(self.output))

        writes = [writer(datasets, self.output)]

        # Now do the subtables
        for table in list(in_fmt.subtables):
            if (
                table in {"SORTED_TABLE", "SOURCE"}
                or self.exclude.get(table, "") == "*"
            ):
                log.warning(f"Ignoring {table}")
                continue

            in_store = self.input.subtable_store(table)
            in_fmt = TableFormat.from_store(in_store)
            out_store = self.output.subtable_store(table)
            out_fmt = TableFormat.from_type(self.format, subtable=table)

            reader = in_fmt.reader()
            writer = out_fmt.writer()

            if isinstance(in_fmt, CasaFormat) and table in NONUNIFORM_SUBTABLES:
                datasets = reader(in_store, group_cols="__row__")
            else:
                datasets = reader(in_store)

            if exclude_columns := self.exclude.get(table, False):
                datasets = [
                    ds.drop_vars(exclude_columns, errors="ignore") for ds in datasets
                ]

            writes.append(writer(datasets, out_store))

        return writes
