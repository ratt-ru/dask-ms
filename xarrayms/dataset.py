# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from collections.abc import Mapping  # python 3.8
except ImportError:
    from collections import Mapping

import logging

import dask
import dask.array as da
from dask.highlevelgraph import HighLevelGraph
import numpy as np
import pyrap.tables as pt

from xarrayms.columns import column_metadata, ColumnMetadataError
from xarrayms.ordering import (ordering_taql, row_ordering,
                               group_ordering_taql, group_row_ordering,
                               _gen_row_runs)
from xarrayms.table_proxy import TableProxy, READLOCK, WRITELOCK
from xarrayms.table_schemas import lookup_table_schema
from xarrayms.utils import short_table_name

log = logging.getLogger(__name__)

_DEFAULT_ROW_CHUNKS = 10000


# This class duplicates xarray's Frozen class in
# https://github.com/pydata/xarray/blob/master/xarray/core/utils.py
# See https://github.com/pydata/xarray/blob/master/LICENSE
class Frozen(Mapping):
    """
    Wrapper around an object implementing the Mapping interface
    to make it immutable.
    """
    __slots__ = "mapping"

    def __init__(self, mapping):
        self.mapping = mapping

    def __getitem__(self, key):
        return self.mapping[key]

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __contains__(self, key):
        return key in self.mapping

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.mapping)


class Dataset(object):
    """
    Poor man's xarray Dataset. It mostly exists so that xarray can
    be an optional dependency, as it in turn depends on pandas
    which is a fairly heavy dependency
    """
    def __init__(self, data_vars_and_dims, attrs=None):
        data_vars = {}
        dims = {}

        for k, v in data_vars_and_dims.items():
            if not isinstance(v, (tuple, list)) and len(v) != 2:
                raise ValueError("'%s' must be a (array, dims) tuple. "
                                 "Got a '%s' instead," % (k, type(v)))

            if v[0].ndim != len(v[1]):
                raise ValueError("Dimension schema '%s' does "
                                 "not match shape of associated array %s"
                                 % (v[0], v[1]))

            data_vars[k] = v[0]
            dims[k] = v[1]

        self._data_vars = data_vars
        self._dims = dims
        self._attrs = attrs or {}

    @property
    def attrs(self):
        return Frozen(self._attrs)

    @property
    def dims(self):
        return Frozen(self._dims)

    @property
    def chunks(self):
        chunks = {}

        for name, var in self._data_vars.items():
            if not isinstance(var, da.Array):
                continue

            for dim, c in zip(self._dims[name], var.chunks):
                if dim in chunks and c != chunks[dim]:
                    raise ValueError("Existing chunk size %d for "
                                     "dimension %s is inconsistent "
                                     "with the chunk size for the "
                                     "same dimension of array %s" %
                                     (c, dim, name))

                chunks[dim] = c

        return chunks

    @property
    def variables(self):
        return Frozen(self._data_vars)

    def assign(self, **kwargs):
        return Dataset(dict({k: (var, self._dims[k]) for k, var
                             in self._data_vars.items()}, **kwargs),
                       attrs=self._attrs)

    def __getattr__(self, name):
        try:
            return self._data_vars[name]
        except KeyError:
            pass

        try:
            return self._attrs[name]
        except KeyError:
            raise ValueError("Invalid Attribute %s" % name)


def dim_extents_array(dim, chunks):
    """
    Produces a an array of chunk extents for a given dimension.

    Parameters
    ----------
    dim : str
        Name of the dimension
    chunks : tuple of ints
        Dimension chunks

    Returns
    -------
    dim_extents : :class:`dask.array.Array`
        dask array where each chunk contains a single (start, end) tuple
        defining the start and end of the chunk. The end is inclusive
        in the python-casacore style.

        The array chunks match ``chunks`` and are innacurate, but
        are used to define chunk sizes of final outputs.

    Notes
    -----
    The returned array should never be computed directly, but
    rather used to produce dataset arrays.
    """

    name = "-".join((dim, dask.base.tokenize(dim, chunks)))
    layers = {}
    start = 0

    for i, c in enumerate(chunks):
        layers[(name, i)] = (start, start + c - 1)  # chunk end is inclusive
        start += c

    graph = HighLevelGraph.from_collections(name, layers, [])
    return da.Array(graph, name, chunks=(chunks,), dtype=np.object)


def getter_wrapper(row_orders, *args):
    """
    Wrapper running I/O operations
    within the table_proxy's associated executor
    """
    row_runs, resort = row_orders
    # Infer number of shape arguments
    nshape_args = len(args) - 3
    # Extract other arguments
    table_proxy, column, dtype = args[nshape_args:]

    # There are other dimensions beside row
    if nshape_args > 0:
        blc, trc = zip(*args[:nshape_args])
        shape = tuple(t - b + 1 for b, t in zip(blc, trc))
        result = np.empty((np.sum(row_runs[:, 1]),) + shape, dtype=dtype)
        io_fn = (object_getcolslice if isinstance(dtype, object)
                 else ndarray_getcolslice)

        # Submit table I/O on executor
        future = table_proxy._ex.submit(io_fn, row_runs, table_proxy,
                                        column, result,
                                        blc, trc, dtype)
    # Row only case
    else:
        result = np.empty((np.sum(row_runs[:, 1]),), dtype=dtype)
        io_fn = (object_getcol if isinstance(dtype, object)
                 else ndarray_getcol)

        # Submit table I/O on executor
        future = table_proxy._ex.submit(io_fn, row_runs, table_proxy,
                                        column, result, dtype)

    # Resort result if necessary
    if resort is not None:
        return future.result()[resort]

    return future.result()


def ndarray_getcol(row_runs, table_proxy, column, result, dtype):
    """ Get numpy array data """
    getcolnp = table_proxy._table.getcolnp
    rr = 0

    table_proxy._acquire(READLOCK)

    try:
        for rs, rl in row_runs:
            getcolnp(column, result[rr:rr + rl], startrow=rs, nrow=rl)
            rr += rl
    finally:
        table_proxy._release(READLOCK)

    return result


def ndarray_getcolslice(row_runs, table_proxy, column, result,
                        blc, trc, dtype):
    """ Get numpy array data """
    getcolslicenp = table_proxy._table.getcolslicenp
    rr = 0

    table_proxy._acquire(READLOCK)

    try:
        for rs, rl in row_runs:
            getcolslicenp(column, result[rr:rr + rl],
                          blc=blc, trc=trc,
                          startrow=rs, nrow=rl)
            rr += rl
    finally:
        table_proxy._release(READLOCK)

    return result


def object_getcol(row_runs, table_proxy, column, result, dtype):
    """ Get object list data """
    getcol = table_proxy._table.getcol
    rr = 0

    table_proxy._acquire(READLOCK)

    try:
        for rs, rl in row_runs:
            result[rr:rr + rl] = np.asarray(getcol(column, rs, rl),
                                            dtype=dtype)
            rr += rl
    finally:
        table_proxy._release(READLOCK)

    return result


def object_getcolslice(row_runs, table_proxy, column, result,
                       blc, trc, dtype):
    """ Get object list data """
    getcolslice = table_proxy._table.getcolslice
    rr = 0

    table_proxy._acquire(READLOCK)

    try:
        for rs, rl in row_runs:
            result[rr:rr + rl] = np.asarray(getcolslice(column, blc, trc,
                                                        startrow=rs, nrow=rl),
                                            dtype=dtype)
            rr += rl
    finally:
        table_proxy._release(READLOCK)

    return result


def _dataset_variable_factory(table_proxy, table_schema, select_cols,
                              exemplar_row, orders, chunks, array_prefix,
                              single_row=False):
    """
    Returns a dictionary of dask arrays representing
    a series of getcols on the appropriate table.

    Produces variables for inclusion in a Dataset.

    Parameters
    ----------
    table_proxy : :class:`xarrayms.table_proxy.TableProxy`
        Table proxy object
    table_schema : dict
        Table schema
    select_cols : list of strings
        List of columns to return
    exemplar_row : int
        row id used to possibly extract an exemplar array in
        order to determine the column shape and dtype attributes
    orders : tuple of :class:`dask.array.Array`
        A (sorted_rows, row_runs) tuple, specifying the
        appropriate rows to extract from the table.
    chunks : dict
        Chunking strategy for the dataset.
    array_prefix : str
        dask array string prefix
    single_row : bool, optional
        Defaults to False.
        Indicates whether the columns should be grouped on single rows.
        If `True` the single row column is squeezed (contracted)
        out of the array.

    Returns
    -------
    dict
        A dictionary looking like :code:`{column: (arrays, dims)}`.
    """

    sorted_rows, row_runs = orders
    dataset_vars = {"ROWID": (sorted_rows, ("row",))}

    for column in select_cols:
        try:
            shape, dims, dim_chunks, dtype = column_metadata(column,
                                                             table_proxy,
                                                             table_schema,
                                                             chunks,
                                                             exemplar_row)
        except ColumnMetadataError:
            log.warning("Ignoring column: '%s'", column, exc_info=True)
            continue

        assert len(dims) == len(shape), (dims, shape)

        # Name of the dask array representing this column
        name = "-".join((array_prefix, column))

        full_dims = ("row",) + dims
        args = [row_runs, ("row",)]

        # Add extent arrays
        for d, c in zip(dims, dim_chunks):
            args.append(dim_extents_array(d, c))
            args.append((d,))

        # Add other variables
        args.extend([table_proxy, None,
                     column, None,
                     dtype, None])

        # Construct the array
        dask_array = da.blockwise(getter_wrapper, full_dims,
                                  *args,
                                  name=name,
                                  dtype=dtype)

        # Squeeze out the single row if requested
        if single_row:
            dask_array = dask_array.squeeze(0)
            full_dims = full_dims[1:]

        # Assign into variable and dimension dataset
        dataset_vars[column] = (dask_array, full_dims)

    return dataset_vars


class DatasetFactory(object):
    def __init__(self, ms, select_cols, group_cols, index_cols, chunks=None):
        # Create or promote chunks to a list of dicts
        if chunks is None:
            chunks = [{'row': _DEFAULT_ROW_CHUNKS}]
        elif isinstance(chunks, dict):
            chunks = [chunks]
        elif not isinstance(chunks, (tuple, list)):
            raise TypeError("'chunks' must be a dict or sequence of dicts")

        self.ms = ms
        self.select_cols = select_cols
        self.group_cols = [] if group_cols is None else group_cols
        self.index_cols = [] if index_cols is None else index_cols
        self.chunks = chunks

    def _table_proxy(self):
        return TableProxy(pt.table, self.ms, ack=False,
                          readonly=True, lockoptions='user')

    def _table_schema(self):
        return lookup_table_schema(self.ms, None)

    def _single_dataset(self, orders, single_row=False, exemplar_row=0):
        table_proxy = self._table_proxy()
        table_schema = self._table_schema()
        select_cols = set(self.select_cols or table_proxy.colnames().result())

        variables = _dataset_variable_factory(table_proxy, table_schema,
                                              select_cols, exemplar_row,
                                              orders, self.chunks[0],
                                              short_table_name(self.ms),
                                              single_row=single_row)

        if single_row:
            return Dataset(variables, attrs={"ROWID": exemplar_row})
        else:
            return Dataset(variables)

    def _group_datasets(self, groups, exemplar_rows, orders):
        table_proxy = self._table_proxy()
        table_schema = self._table_schema()

        datasets = []
        group_ids = list(zip(*groups))

        assert len(group_ids) == len(orders)

        # Select columns, excluding grouping columns
        select_cols = set(self.select_cols or table_proxy.colnames().result())
        select_cols -= set(self.group_cols)

        # Create a dataset for each group
        it = enumerate(zip(group_ids, exemplar_rows, orders))

        for g, (group_id, exemplar_row, order) in it:
            # Extract group chunks
            try:
                group_chunks = self.chunks[g]   # Get group chunking strategy
            except IndexError:
                group_chunks = self.chunks[-1]  # Re-use last group's chunks

            # Prefix d
            gid_str = ",".join(str(gid) for gid in group_id)
            array_prefix = "%s-[%s]" % (short_table_name(self.ms), gid_str)

            # Create dataset variables
            group_var_dims = _dataset_variable_factory(table_proxy,
                                                       table_schema,
                                                       select_cols,
                                                       exemplar_row,
                                                       order, group_chunks,
                                                       array_prefix)

            # Assign values for the dataset's grouping columns
            # as attributes
            attrs = dict(zip(self.group_cols, group_id))
            datasets.append(Dataset(group_var_dims, attrs=attrs))

        return datasets

    def datasets(self):
        # No grouping case
        if len(self.group_cols) == 0:
            order_taql = ordering_taql(self.ms, self.index_cols)
            orders = row_ordering(order_taql, self.index_cols, self.chunks[0])
            return [self._single_dataset(orders)]
        # Group by row
        elif len(self.group_cols) == 1 and self.group_cols[0] == "__row__":
            order_taql = ordering_taql(self.ms, self.index_cols)
            sorted_rows, row_runs = row_ordering(order_taql,
                                                 self.index_cols,
                                                 # chunk ordering on each row
                                                 dict(self.chunks[0], row=1))

            # Produce a dataset for each chunk (block),
            # each containing a single row
            row_blocks = sorted_rows.blocks
            run_blocks = row_runs.blocks

            # Exemplar actually correspond to the sorted rows.
            # We reify them here so they can be assigned on each
            # dataset as an attribute
            np_sorted_row = sorted_rows.compute()

            return [self._single_dataset((row_blocks[r], run_blocks[r]),
                                         single_row=True, exemplar_row=er)
                    for r, er in enumerate(np_sorted_row)]
        # Grouping column case
        else:
            order_taql = group_ordering_taql(self.ms, self.group_cols,
                                             self.index_cols)
            orders = group_row_ordering(order_taql, self.group_cols,
                                        self.index_cols, self.chunks)

            groups = [order_taql.getcol(g).result() for g in self.group_cols]
            exemplar_rows = order_taql.getcol("__firstrow__").result()
            assert len(orders) == len(exemplar_rows)

            return self._group_datasets(groups, exemplar_rows, orders)


def dataset(ms, columns, group_cols, index_cols, chunks=None):
    return DatasetFactory(ms, columns, group_cols,
                          index_cols, chunks).datasets()


def putter_wrapper(row_orders, table_proxy, column, data):
    """
    Wrapper which should run I/O operations within
    the table_proxy's associated executor
    """
    row_runs, resort = row_orders

    if resort is not None:
        data = data[resort]

    table_proxy._ex.submit(array_putter, row_runs, table_proxy,
                           column, data).result()

    return np.full((1,) * len(data.shape), True)


def array_putter(row_runs, table_proxy, column, data):
    """ Put data into the table """
    putcol = table_proxy._table.putcol
    rr = 0

    table_proxy._acquire(WRITELOCK)

    try:
        for rs, rl in row_runs:
            putcol(column, data[rr:rr + rl], startrow=rs, nrow=rl)
            rr += rl

    finally:
        table_proxy._release(WRITELOCK)


def write_columns(ms, dataset, columns):
    table_proxy = TableProxy(pt.table, ms, ack=False,
                             readonly=False, lockoptions='user')
    writes = []
    row_order = dataset.ROWID.map_blocks(_gen_row_runs, sort_dir="write",
                                         dtype=np.object)
    dims = dataset.dims
    data_vars = dataset.variables

    for column in columns:
        try:
            column_array = data_vars[column]
        except KeyError:
            log.warning("Ignoring '%s' not present on the dataset.")
            continue

        try:
            col_dims = dims[column]
        except KeyError:
            raise ValueError("No 'dims' found for column '%s'" % column)

        for chunk in column_array.chunks[1:]:
            if not len(chunk) == 1:
                raise ValueError("Multiple chunks in non-row "
                                 "dimension unsupported for writes. "
                                 "Rechunk this data to perform a single "
                                 "write.")

        column_write = da.blockwise(putter_wrapper, col_dims,
                                    row_order, ("row",),
                                    table_proxy, None,
                                    column, None,
                                    column_array, col_dims,
                                    # All dims shrink to 1,
                                    # a single bool is returned
                                    adjust_chunks={d: 1 for d in col_dims},
                                    dtype=np.bool)

        writes.append(column_write.ravel())

    return da.concatenate(writes)
