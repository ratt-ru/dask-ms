# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from uuid import uuid4

import dask
import dask.array as da
from dask.array.core import normalize_chunks
import numpy as np

from daskms.columns import (
    arcae_dtype,
    column_metadata,
    ColumnMetadataError,
    infer_dtype,
)
from daskms.casa_table import CasaTable, build_index, ninstances
from daskms.constants import DASKMS_PARTITION_KEY
from daskms.dataset import Dataset
from daskms.structure import ROW_GROUP, structure_factory
from daskms.table import table_exists
from daskms.table_schemas import lookup_table_schema
from daskms.utils import table_path_split

_DEFAULT_ROW_CHUNKS = 10000

log = logging.getLogger(__name__)


class GroupChunkingError(ValueError):
    pass


def getter_wrapper(
    rows, casa_table, column, col_shape, dtype, variable, block_info=None
):
    """Read a block of ``column`` out of the table.

    Called through :func:`dask.array.map_blocks`, whose ``block_info``
    supplies the block's half-open extent along each non-row dimension.
    arcae reads a whole block in a single call: the row ids and the
    extents are combined into one index, and the data is read directly
    into the output buffer. arcae returns rows in the order requested,
    so no resorting is required.

    Cells of a ``variable`` (variably shaped) column that are smaller than
    ``col_shape`` are padded with :func:`pad_value`.
    """
    extents = block_info[None]["array-location"][1:]
    shape = (len(rows),) + tuple(stop - start for start, stop in extents)

    if np.prod(shape) == 0:
        return np.empty(shape, dtype=dtype)

    # Only select along the non-row dimensions if they are chunked.
    # Otherwise whole cells are read, which are padded if they are
    # smaller than col_shape
    if all(extent == (0, s) for extent, s in zip(map(tuple, extents), col_shape)):
        extents = ()

    table = casa_table.instance
    index = build_index(rows, extents)

    try:
        return _getcol(table, column, index, shape, dtype, variable)
    except IndexError as e:
        if not (variable and extents):
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


def _rowid_block(structure_factory, key, block_info=None):
    """A block of a partition's sorted row ids"""
    ((start, stop),) = block_info[None]["array-location"]
    return structure_factory.instance[key].rows[start:stop]


def rowid_array(structure_factory, partition, row_chunks, array_suffix):
    """A dask array of ``partition``'s sorted row ids, whose blocks
    are looked up in the structure when they are computed"""
    try:
        chunks = normalize_chunks(row_chunks, shape=(partition.nrow,))
    except ValueError as e:
        raise GroupChunkingError(
            f"{e}\n"
            f"Unable to match chunks '{row_chunks}' with shape "
            f"'{(partition.nrow,)}' for partition {partition.key}. "
            f"This can occur if too few chunk dictionaries have been "
            f"supplied for the number of groups and an earlier group's "
            f"chunking strategy is applied to a later one."
        ) from e

    token = dask.base.tokenize(structure_factory, partition.key, chunks)

    return da.map_blocks(
        _rowid_block,
        structure_factory,
        partition.key,
        chunks=chunks,
        dtype=np.int64,
        meta=np.empty((0,), dtype=np.int64),
        name=f"rowid~{array_suffix}-{token}",
    )


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
    partition,
    rowid,
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
    partition : :class:`daskms.structure.PartitionData`
        The partition of the table that the dataset holds. Its rows and
        exemplar row determine the shape of variably shaped columns.
    rowid : :class:`dask.array.Array`
        The partition's sorted row ids, in the order that they
        should appear in the dataset.
    chunks : dict
        Chunking strategy for the dataset.
    array_suffix : str
        dask array string prefix

    Returns
    -------
    dict
        A dictionary looking like :code:`{column: (arrays, dims)}`.
    """

    dataset_vars = {"ROWID": (("row",), rowid)}

    for column in select_cols:
        try:
            meta = column_metadata(
                column,
                table_proxy,
                table_schema,
                chunks,
                partition.exemplar_row,
                rows=partition.rows,
            )
        except ColumnMetadataError as e:
            exc_info = logging.DEBUG >= log.getEffectiveLevel()
            log.warning("Ignoring '%s': %s", column, e, exc_info=exc_info)
            continue

        full_dims = ("row",) + meta.dims
        ndim = len(full_dims)
        token = dask.base.tokenize(rowid.name, table_proxy, column, meta)
        name = "~".join(("read", column, array_suffix)) + "-" + token

        dask_array = da.map_blocks(
            getter_wrapper,
            rowid,
            table_proxy,
            column,
            meta.shape,
            meta.dtype,
            meta.variable,
            new_axis=list(range(1, ndim)),
            chunks=rowid.chunks + tuple(meta.chunks),
            dtype=meta.dtype,
            meta=np.empty((0,) * ndim, dtype=meta.dtype),
            name=name,
        )

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
        self.epoch = kwargs.pop("epoch", None) or uuid4().hex[:16]

        if len(kwargs) > 0:
            raise ValueError(f"Unhandled kwargs: {kwargs}")

    def _casa_table_factory(self):
        return CasaTable.from_table(
            self.table_path, ninstances=ninstances(), readonly=True
        )

    def _table_schema(self):
        return lookup_table_schema(self.canonical_name, self.table_schema)

    def _single_dataset(self, table_proxy, factory, partition):
        _, t, s = table_path_split(self.canonical_name)
        short_table_name = "/".join((t, s)) if s else t
        chunks = self.chunks[0]

        table_schema = self._table_schema()
        select_cols = set(self.select_cols or table_proxy.instance.columns())
        rowid = rowid_array(factory, partition, chunks["row"], short_table_name)
        variables = _dataset_variable_factory(
            table_proxy,
            table_schema,
            select_cols,
            partition,
            rowid,
            chunks,
            short_table_name,
        )

        try:
            coords = {"ROWID": variables.pop("ROWID")}
        except KeyError:
            coords = None

        attrs = {DASKMS_PARTITION_KEY: ()}
        dataset = Dataset(variables, coords=coords, attrs=attrs)
        return self.postprocess_dataset(
            dataset, table_proxy, partition, rowid, chunks, short_table_name
        )

    def _group_datasets(self, table_proxy, factory, partitions):
        _, t, s = table_path_split(self.canonical_name)
        short_table_name = "/".join((t, s)) if s else t
        table_schema = self._table_schema()

        # Cast group values to the actual column dtype
        group_types = [
            infer_dtype(c, table_proxy.instance.getcoldesc(c)) for c in self.group_cols
        ]

        datasets = []

        # Select columns, excluding grouping columns
        select_cols = set(self.select_cols or table_proxy.instance.columns())
        select_cols -= set(self.group_cols)

        # Create a dataset for each group
        for g, partition in enumerate(partitions):
            # Extract group chunks
            try:
                group_chunks = self.chunks[g]  # Get group chunking strategy
            except IndexError:
                group_chunks = self.chunks[-1]  # Re-use last group's chunks

            try:
                row_chunks = group_chunks["row"]
            except KeyError:
                raise ValueError(f"No row chunking scheme found in {group_chunks}!")

            group_id = [
                np.asarray(v).astype(t) for (_, v), t in zip(partition.key, group_types)
            ]

            # Prefix dataset
            gid_str = ",".join(str(gid) for gid in group_id)
            array_suffix = f"[{gid_str}]-{short_table_name}"

            # Create dataset variables
            rowid = rowid_array(factory, partition, row_chunks, array_suffix)
            group_var_dims = _dataset_variable_factory(
                table_proxy,
                table_schema,
                select_cols,
                partition,
                rowid,
                group_chunks,
                array_suffix,
            )

            # Extract ROWID
            try:
                coords = {"ROWID": group_var_dims.pop("ROWID")}
            except KeyError:
                coords = None

            # Assign values for the dataset's grouping columns
            # as attributes
            partitions_attr = tuple(
                (c, g.dtype.name) for c, g in zip(self.group_cols, group_id)
            )
            attrs = {DASKMS_PARTITION_KEY: partitions_attr}

            # Use python types which are json serializable
            group_id = [gid.item() for gid in group_id]
            attrs.update(zip(self.group_cols, group_id))

            dataset = Dataset(group_var_dims, attrs=attrs, coords=coords)
            dataset = self.postprocess_dataset(
                dataset, table_proxy, partition, rowid, group_chunks, array_suffix
            )
            datasets.append(dataset)

        return datasets

    def postprocess_dataset(
        self, dataset, table_proxy, partition, rowid, chunks, array_suffix
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
                partition,
                rowid,
                chunks,
                array_suffix,
            )
        )

    def datasets(self):
        table_proxy = self._casa_table_factory()
        factory = structure_factory(
            table_proxy,
            self.group_cols,
            self.index_cols,
            self.taql_where,
            self.epoch,
        )
        structure = factory.instance

        # No grouping, or grouping by row, where each
        # row becomes a dataset of its own
        if len(self.group_cols) == 0 or self.group_cols == [ROW_GROUP]:
            datasets = [
                self._single_dataset(table_proxy, factory, partition)
                for partition in structure.values()
            ]
        # Grouping column case
        else:
            datasets = self._group_datasets(
                table_proxy, factory, list(structure.values())
            )

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
