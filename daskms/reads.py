# -*- coding: utf-8 -*-

import functools
import logging
from pathlib import Path
import warnings

import dask
import dask.array as da
import numpy as np

from daskms.columns import (
    arcae_dtype,
    column_metadata,
    ColumnMetadataError,
    dim_extents_array,
    infer_dtype,
)
from daskms.casa_table import CasaTable, build_index, ninstances
from daskms.constants import DASKMS_PARTITION_KEY
from daskms.ordering import (
    ordering_taql,
    row_ordering,
    group_ordering_taql,
    group_row_ordering,
)
from daskms.optimisation import inlined_array
from daskms.dataset import Dataset
from daskms.table import table_exists
from daskms.table_schemas import lookup_table_schema
from daskms.utils import table_path_split

_DEFAULT_ROW_CHUNKS = 10000

log = logging.getLogger(__name__)


def getter_wrapper(rows, *args):
    """Read a chunk of ``column`` out of the table.

    arcae reads a whole chunk in a single call: the row ids and the
    per-dimension chunk extents are combined into one index, and the data
    is read directly into the output buffer. arcae returns rows in the
    order requested, so no resorting is required.

    Cells of a ``variable`` (variably shaped) column that are smaller than
    ``col_shape`` are padded with :func:`pad_value`.
    """
    # Infer number of shape arguments
    nextent_args = len(args) - 5
    # Extract other arguments
    casa_table, column, col_shape, dtype, variable = args[nextent_args:]

    # args[:nextent_args] is one inclusive (blc, trc) pair per non-row
    # dimension of the column, defining the extent of this chunk
    extents = args[:nextent_args]

    # Handle dask compute_meta gracefully: it passes empty arrays
    # in place of the extent pairs
    if any(isinstance(e, np.ndarray) for e in extents):
        return np.empty((0,) * (nextent_args + 1), dtype=dtype)

    if nextent_args > 0:
        shape = tuple(trc - blc + 1 for blc, trc in extents)
    # Otherwise the full resolution data for each row is requested
    else:
        shape = col_shape

    shape = (len(rows),) + tuple(shape)

    if np.prod(shape) == 0:
        return np.empty(shape, dtype=dtype)

    table = casa_table.instance
    index = build_index(rows, extents)

    try:
        return _getcol(table, column, index, shape, dtype, variable)
    except IndexError as e:
        if not (variable and nextent_args > 0):
            raise

        # arcae rejects a selection that extends past the end of a cell.
        # This only happens if a dataset mixes cells of different shapes
        # and the column's non-row dimensions are also chunked
        raise IndexError(
            f"Chunk {index[1:]} of variably shaped column '{column}' extends "
            f"past the end of some of its cells. Group the table so that the "
            f"cells in each dataset share a shape (the default DATA_DESC_ID "
            f"grouping for a Measurement Set, or group_cols='__row__' for a "
            f"subtable), or leave the non-row dimensions of '{column}' unchunked."
        ) from e


def pad_value(dtype):
    """The value that pads cells of a variably shaped column"""
    dtype = np.dtype(dtype)

    if dtype == object:
        return ""
    elif dtype.kind in "fc":
        return np.nan
    elif dtype.kind == "b":
        return False

    return 0


def _getcol(table, column, index, shape, dtype, variable):
    if dtype == object:
        # arcae cannot read strings into a pre-allocated buffer
        if variable:
            return _variable_string_getcol(table, column, index, shape)

        return table.getcol(column, index=index)

    if variable:
        # arcae pads cells that are smaller than the result
        result = np.full(shape, pad_value(dtype), dtype=dtype)
    else:
        result = np.empty(shape, dtype=dtype)

    # arcae_dtype for the bool case: arcae would otherwise size a numpy
    # bool buffer as bit-packed Arrow and reject it
    table.getcol(column, index=index, result=result.view(arcae_dtype(dtype)))
    return result


def _variable_string_getcol(table, column, index, shape):
    """Read a variably shaped string ``column``, padding cells to ``shape``.

    arcae returns such a column as nested Arrow lists, rather than
    reading it into a pre-allocated buffer, so the padding happens here.
    It cannot select along secondary dimensions when doing so, so whole
    cells are read and the ``index[1:]`` slices applied to each of them.
    """
    rows, secondary = index[0], index[1:]

    if isinstance(rows, slice):
        rows = np.arange(rows.start, rows.stop)

    result = np.full(shape, pad_value(object), dtype=object)

    # arcae omits the column entirely if any requested cell is
    # undefined, so only read those that are
    (defined,) = np.nonzero(table.row_shapes(column, (rows,)).is_valid())

    if len(defined) == 0:
        return result

    arrow_table = table.to_arrow((rows[defined],), column)

    for i, cell in zip(defined, arrow_table.column(column).to_pylist()):
        cell = np.array(cell, dtype=object)[secondary]
        result[(i,) + tuple(slice(0, s) for s in cell.shape)] = cell

    return result


def _dataset_variable_factory(
    table_proxy,
    table_schema,
    select_cols,
    exemplar_row,
    sorted_rows,
    chunks,
    array_suffix,
):
    """
    Returns a dictionary of dask arrays representing
    a series of getcols on the appropriate table.

    Produces variables for inclusion in a Dataset.

    Parameters
    ----------
    table_proxy : :class:`daskms.casa_table.CasaTable`
        Table proxy object
    table_schema : dict
        Table schema
    select_cols : list of strings
        List of columns to return
    exemplar_row : int
        row id used to possibly extract an exemplar array in
        order to determine the column shape and dtype attributes
    sorted_rows : :class:`dask.array.Array`
        The rows to extract from the table, in the order
        they should appear in the dataset.
    chunks : dict
        Chunking strategy for the dataset.
    array_suffix : str
        dask array string prefix

    Returns
    -------
    dict
        A dictionary looking like :code:`{column: (arrays, dims)}`.
    """

    dataset_vars = {"ROWID": (("row",), sorted_rows)}

    # The shape of a variably shaped column is maximal over the dataset's
    # rows, which are materialised here, once, if such a column exists
    @functools.cache
    def dataset_rows():
        return sorted_rows.compute(scheduler="sync")

    for column in select_cols:
        try:
            meta = column_metadata(
                column,
                table_proxy,
                table_schema,
                chunks,
                exemplar_row,
                rows=dataset_rows,
            )
        except ColumnMetadataError as e:
            exc_info = logging.DEBUG >= log.getEffectiveLevel()
            log.warning("Ignoring '%s': %s", column, e, exc_info=exc_info)
            continue

        full_dims = ("row",) + meta.dims
        args = [sorted_rows, ("row",)]

        # We only need to pass in dimension extent arrays if
        # there is more than one chunk in any of the non-row columns.
        # Otherwise the whole of each row is read.
        if not all(len(c) == 1 for c in meta.chunks):
            for d, c in zip(meta.dims, meta.chunks):
                # Create an array describing the dimension chunk extents
                args.append(dim_extents_array(d, c))
                args.append((d,))

            new_axes = {}
        else:
            # We need to inform blockwise about the size of our
            # new dimensions as no arrays with them are supplied
            new_axes = {d: s for d, s in zip(meta.dims, meta.shape)}

        # Add other variables
        args.extend(
            [
                table_proxy,
                None,
                column,
                None,
                meta.shape,
                None,
                meta.dtype,
                None,
                meta.variable,
                None,
            ]
        )

        # Name of the dask array representing this column
        token = dask.base.tokenize(args)
        name = "~".join(("read", column, array_suffix)) + "-" + token

        # Construct the array

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=da.PerformanceWarning)
            dask_array = da.blockwise(
                getter_wrapper,
                full_dims,
                *args,
                name=name,
                new_axes=new_axes,
                dtype=meta.dtype,
            )

        dask_array = inlined_array(dask_array)

        # Assign into variable and dimension dataset
        dataset_vars[column] = (full_dims, dask_array)

    return dataset_vars


def _col_keyword_getter(table):
    """Gets column keywords for all columns in table"""
    return {c: table.getcolkeywords(c) for c in table.columns()}


class DatasetFactory(object):
    def __init__(self, table, select_cols, group_cols, index_cols, **kwargs):
        if not table_exists(table):
            raise ValueError(f"'{table}' does not appear to be a CASA Table")

        chunks = kwargs.pop("chunks", [{"row": _DEFAULT_ROW_CHUNKS}])

        # Create or promote chunks to a list of dicts
        if isinstance(chunks, dict):
            chunks = [chunks]
        elif not isinstance(chunks, (tuple, list)):
            raise TypeError("'chunks' must be a dict or sequence of dicts")

        self.canonical_name = table
        self.table_path = str(Path(*table_path_split(table)))
        self.select_cols = select_cols
        self.group_cols = [] if group_cols is None else group_cols
        self.index_cols = [] if index_cols is None else index_cols
        self.chunks = chunks
        self.table_schema = kwargs.pop("table_schema", None)
        self.taql_where = kwargs.pop("taql_where", "")
        self.table_keywords = kwargs.pop("table_keywords", False)
        self.column_keywords = kwargs.pop("column_keywords", False)
        self.table_proxy = kwargs.pop("table_proxy", False)
        self.context = kwargs.pop("context", None)

        if len(kwargs) > 0:
            raise ValueError(f"Unhandled kwargs: {kwargs}")

    def _casa_table_factory(self):
        return CasaTable.from_table(
            self.table_path, ninstances=ninstances(), readonly=True
        )

    def _table_schema(self):
        return lookup_table_schema(self.canonical_name, self.table_schema)

    def _single_dataset(self, table_proxy, orders, exemplar_row=0):
        _, t, s = table_path_split(self.canonical_name)
        short_table_name = "/".join((t, s)) if s else t

        table_schema = self._table_schema()
        select_cols = set(self.select_cols or table_proxy.instance.columns())
        variables = _dataset_variable_factory(
            table_proxy,
            table_schema,
            select_cols,
            exemplar_row,
            orders,
            self.chunks[0],
            short_table_name,
        )

        try:
            rowid = variables.pop("ROWID")
        except KeyError:
            coords = None
        else:
            coords = {"ROWID": rowid}

        attrs = {DASKMS_PARTITION_KEY: ()}
        dataset = Dataset(variables, coords=coords, attrs=attrs)
        return self.postprocess_dataset(
            dataset, table_proxy, exemplar_row, orders, self.chunks[0], short_table_name
        )

    def _group_datasets(self, table_proxy, groups, exemplar_rows, orders):
        _, t, s = table_path_split(self.canonical_name)
        short_table_name = "/".join((t, s)) if s else t
        table_schema = self._table_schema()

        datasets = []
        group_ids = list(zip(*groups))

        assert len(group_ids) == len(orders)

        # Select columns, excluding grouping columns
        select_cols = set(self.select_cols or table_proxy.instance.columns())
        select_cols -= set(self.group_cols)

        # Create a dataset for each group
        it = enumerate(zip(group_ids, exemplar_rows, orders))

        for g, (group_id, exemplar_row, order) in it:
            # Extract group chunks
            try:
                group_chunks = self.chunks[g]  # Get group chunking strategy
            except IndexError:
                group_chunks = self.chunks[-1]  # Re-use last group's chunks

            # Prefix dataset
            gid_str = ",".join(str(gid) for gid in group_id)
            array_suffix = f"[{gid_str}]-{short_table_name}"

            # Create dataset variables
            group_var_dims = _dataset_variable_factory(
                table_proxy,
                table_schema,
                select_cols,
                exemplar_row,
                order,
                group_chunks,
                array_suffix,
            )

            # Extract ROWID
            try:
                rowid = group_var_dims.pop("ROWID")
            except KeyError:
                coords = None
            else:
                coords = {"ROWID": rowid}

            # Assign values for the dataset's grouping columns
            # as attributes
            partitions = tuple(
                (c, g.dtype.name) for c, g in zip(self.group_cols, group_id)
            )
            attrs = {DASKMS_PARTITION_KEY: partitions}

            # Use python types which are json serializable
            group_id = [gid.item() for gid in group_id]
            attrs.update(zip(self.group_cols, group_id))

            dataset = Dataset(group_var_dims, attrs=attrs, coords=coords)
            dataset = self.postprocess_dataset(
                dataset, table_proxy, exemplar_row, order, group_chunks, array_suffix
            )
            datasets.append(dataset)

        return datasets

    def postprocess_dataset(
        self, dataset, table_proxy, exemplar_row, order, chunks, array_suffix
    ):
        if not self.context or self.context != "ms":
            return dataset

        # Fixup any non-standard columns
        # with dimensions like chan and corr
        try:
            chan = dataset.sizes["chan"]
            corr = dataset.sizes["corr"]
        except KeyError:
            return dataset

        schema_updates = {}

        for name, var in dataset.data_vars.items():
            new_dims = list(var.dims[1:])

            unassigned = {"chan", "corr"}

            for dim, dim_name in enumerate(var.dims[1:]):
                # An automicatically assigned dimension name
                if dim_name == f"{name}-{dim + 1}":
                    if dataset.sizes[dim_name] == chan and "chan" in unassigned:
                        new_dims[dim] = "chan"
                        unassigned.discard("chan")
                    elif dataset.sizes[dim_name] == corr and "corr" in unassigned:
                        new_dims[dim] = "corr"
                        unassigned.discard("corr")

            new_dims = tuple(new_dims)
            if var.dims[1:] != new_dims:
                schema_updates[name] = {"dims": new_dims}

        if not schema_updates:
            return dataset

        return dataset.assign(
            **_dataset_variable_factory(
                table_proxy,
                schema_updates,
                list(schema_updates.keys()),
                exemplar_row,
                order,
                chunks,
                array_suffix,
            )
        )

    def datasets(self):
        table_proxy = self._casa_table_factory()

        # No grouping case
        if len(self.group_cols) == 0:
            order_taql = ordering_taql(table_proxy, self.index_cols, self.taql_where)
            orders = row_ordering(order_taql, self.index_cols, self.chunks[0])
            datasets = [self._single_dataset(table_proxy, orders)]
        # Group by row
        elif len(self.group_cols) == 1 and self.group_cols[0] == "__row__":
            order_taql = ordering_taql(table_proxy, self.index_cols, self.taql_where)
            sorted_rows = row_ordering(
                order_taql,
                self.index_cols,
                # chunk ordering on each row
                dict(self.chunks[0], row=1),
            )

            # Produce a dataset for each chunk (block),
            # each containing a single row
            row_blocks = sorted_rows.blocks

            # Exemplar actually correspond to the sorted rows.
            # We reify them here so they can be assigned on each
            # dataset as an attribute
            np_sorted_row = sorted_rows.compute()

            datasets = [
                self._single_dataset(table_proxy, row_blocks[r], exemplar_row=er)
                for r, er in enumerate(np_sorted_row)
            ]
        # Grouping column case
        else:
            order_taql = group_ordering_taql(
                table_proxy, self.group_cols, self.index_cols, self.taql_where
            )
            orders = group_row_ordering(
                order_taql, self.group_cols, self.index_cols, self.chunks
            )

            groups = [order_taql.instance.getcol(g) for g in self.group_cols]
            # Cast to actual column dtype
            group_types = [
                infer_dtype(c, table_proxy.instance.getcoldesc(c))
                for c in self.group_cols
            ]
            groups = [g.astype(t) for g, t in zip(groups, group_types)]
            exemplar_rows = order_taql.instance.getcol("__firstrow__")
            assert len(orders) == len(exemplar_rows)

            datasets = self._group_datasets(table_proxy, groups, exemplar_rows, orders)

        ret = (datasets,)

        if self.table_keywords is True:
            ret += (table_proxy.instance.getkeywords(),)

        if self.column_keywords is True:
            ret += (_col_keyword_getter(table_proxy.instance),)

        if self.table_proxy is True:
            ret += (table_proxy,)

        if len(ret) == 1:
            return ret[0]

        return ret


def read_datasets(ms, columns, group_cols, index_cols, **kwargs):
    return DatasetFactory(ms, columns, group_cols, index_cols, **kwargs).datasets()
