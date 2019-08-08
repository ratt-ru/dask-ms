# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from collections.abc import Mapping  # python 3.8
except ImportError:
    from collections import Mapping

from collections import namedtuple
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


VariableEntry = namedtuple("VariableEntry", ["dims", "var", "attrs"])


class Dataset(object):
    """
    Poor man's xarray Dataset. It mostly exists so that xarray can
    be an optional dependency, as it in turn depends on pandas
    which is a fairly heavy dependency
    """
    def __init__(self, data_vars, attrs=None):
        self._data_vars = {}

        for k, v in data_vars.items():
            if isinstance(v, VariableEntry):
                self._data_vars[k] = v
                continue

            if not isinstance(v, (tuple, list)) and len(v) not in (2, 3):
                raise ValueError("'%s' must be a (dims, array) or "
                                 "(dims, array, attrs) tuple. "
                                 "Got '%s' instead," % (k, type(v)))

            dims = v[0]
            var = v[1]
            var_attrs = v[2] if len(v) > 2 else {}

            if len(dims) != var.ndim:
                raise ValueError("Dimension schema '%s' does "
                                 "not match shape of associated array %s"
                                 % (dims, var))

            self._data_vars[k] = VariableEntry(dims, var, var_attrs)

        self._attrs = attrs or {}

    @property
    def attrs(self):
        return Frozen(self._attrs)

    @property
    def dims(self):
        dims = {}

        for k, (var_dims, var, _) in self._data_vars.items():
            for d, s in zip(var_dims, var.shape):

                if d in dims and s != dims[d]:
                    raise ValueError("Existing dimension size %d for "
                                     "dimension '%s' is inconsistent "
                                     "with same dimension of array %s" %
                                     (s, d, k))

                dims[d] = s

        return dims

    sizes = dims

    @property
    def chunks(self):
        chunks = {}

        for k, (var_dims, var, _) in self._data_vars.items():
            if not isinstance(var, da.Array):
                continue

            for dim, c in zip(var_dims, var.chunks):
                if dim in chunks and c != chunks[dim]:
                    raise ValueError("Existing chunking %s for "
                                     "dimension '%s' is inconsistent "
                                     "with chunking %s for the "
                                     "same dimension of array %s" %
                                     (c, dim, chunks[dim], k))

                chunks[dim] = c

        return chunks

    @property
    def variables(self):
        return Frozen(self._data_vars)

    def assign(self, **kwargs):
        data_vars = self._data_vars.copy()

        for k, v in kwargs.items():
            if not isinstance(v, (list, tuple)):
                try:
                    current_var = data_vars[k]
                except KeyError:
                    raise ValueError("Couldn't find existing dimension schema "
                                     "during assignment of variable '%s'. "
                                     "Supply a full (dims, array) tuple."
                                     % k)
                else:
                    data_vars[k] = (current_var.dims, v, current_var.attrs)
            else:
                data_vars[k] = v

        return Dataset(data_vars, attrs=self._attrs)

    def __getattr__(self, name):
        try:
            return self._data_vars[name][1]
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
    # Infer number of shape arguments
    nextent_args = len(args) - 4
    # Extract other arguments
    table_proxy, column, col_shape, dtype = args[nextent_args:]

    # Handle dask compute_meta gracefully
    if len(row_orders) == 0:
        return np.empty((0,)*(nextent_args+1), dtype=dtype)

    row_runs, resort = row_orders

    # In this case, we've been passed dimension extent arrays
    # that define a slice of the column and we defer to getcolslice.
    if nextent_args > 0:
        blc, trc = zip(*args[:nextent_args])
        shape = tuple(t - b + 1 for b, t in zip(blc, trc))
        result = np.empty((np.sum(row_runs[:, 1]),) + shape, dtype=dtype)
        io_fn = (object_getcolslice if np.dtype == object
                 else ndarray_getcolslice)

        # Submit table I/O on executor
        future = table_proxy._ex.submit(io_fn, row_runs, table_proxy,
                                        column, result,
                                        blc, trc, dtype)
    # In this case, the full resolution data
    # for each row is requested, so we defer to getcol
    else:
        result = np.empty((np.sum(row_runs[:, 1]),) + col_shape, dtype=dtype)
        io_fn = (object_getcol if dtype == object
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
            # NOTE(sjperkins)
            # Dask wants ndarrays internally, so we asarray objects
            # the returning list of objects.
            # See https://github.com/ska-sa/xarray-ms/issues/42
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
    dataset_vars = {"ROWID": (("row",), sorted_rows)}

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

        full_dims = ("row",) + dims
        args = [row_runs, ("row",)]

        # We only need to pass in dimension extent arrays if
        # there is more than one chunk in any of the non-row columns.
        # In that case, we can getcol, otherwise getcolslice is required
        if not all(len(c) == 1 for c in dim_chunks):
            for d, c in zip(dims, dim_chunks):
                args.append(dim_extents_array(d, c))
                args.append((d,))

            new_axes = {}
        else:
            # We need to inform blockwise about the size of our
            # new dimensions as no arrays with them are supplied
            new_axes = {d: s for d, s in zip(dims, shape)}

        # Add other variables
        args.extend([table_proxy, None,
                     column, None,
                     shape, None,
                     dtype, None])

        # Name of the dask array representing this column
        token = dask.base.tokenize(args)
        name = "-".join((array_prefix, column, token))

        # Construct the array
        dask_array = da.blockwise(getter_wrapper, full_dims,
                                  *args,
                                  name=name,
                                  new_axes=new_axes,
                                  dtype=dtype)

        # Squeeze out the single row if requested
        if single_row:
            dask_array = dask_array.squeeze(0)
            full_dims = full_dims[1:]

        # Assign into variable and dimension dataset
        dataset_vars[column] = (full_dims, dask_array)

    return dataset_vars


class DatasetFactory(object):
    def __init__(self, ms, select_cols, group_cols, index_cols, **kwargs):
        chunks = kwargs.pop('chunks', [{'row': _DEFAULT_ROW_CHUNKS}])

        # Create or promote chunks to a list of dicts
        if isinstance(chunks, dict):
            chunks = [chunks]
        elif not isinstance(chunks, (tuple, list)):
            raise TypeError("'chunks' must be a dict or sequence of dicts")

        self.ms = ms
        self.select_cols = select_cols
        self.group_cols = [] if group_cols is None else group_cols
        self.index_cols = [] if index_cols is None else index_cols
        self.chunks = chunks
        self.table_schema = kwargs.pop('table_schema', None)
        self.taql_where = kwargs.pop('taql_where', '')

        if len(kwargs) > 0:
            raise ValueError("Unhandled kwargs: %s" % kwargs)

    def _table_proxy(self):
        return TableProxy(pt.table, self.ms, ack=False,
                          readonly=True, lockoptions='user')

    def _table_schema(self):
        return lookup_table_schema(self.ms, self.table_schema)

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
        table_proxy = self._table_proxy()

        # No grouping case
        if len(self.group_cols) == 0:
            order_taql = ordering_taql(table_proxy, self.index_cols,
                                       self.taql_where)
            orders = row_ordering(order_taql, self.index_cols, self.chunks[0])
            return [self._single_dataset(orders)]
        # Group by row
        elif len(self.group_cols) == 1 and self.group_cols[0] == "__row__":
            order_taql = ordering_taql(table_proxy, self.index_cols,
                                       self.taql_where)
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
            order_taql = group_ordering_taql(table_proxy, self.group_cols,
                                             self.index_cols, self.taql_where)
            orders = group_row_ordering(order_taql, self.group_cols,
                                        self.index_cols, self.chunks)

            groups = [order_taql.getcol(g).result() for g in self.group_cols]
            exemplar_rows = order_taql.getcol("__firstrow__").result()
            assert len(orders) == len(exemplar_rows)

            return self._group_datasets(groups, exemplar_rows, orders)


def dataset(ms, columns, group_cols, index_cols, **kwargs):
    return DatasetFactory(ms, columns, group_cols,
                          index_cols, **kwargs).datasets()


def putter_wrapper(row_orders, *args):
    """
    Wrapper which should run I/O operations within
    the table_proxy's associated executor
    """
    # Handle dask's compute_meta gracefully

    # Infer number of shape arguments
    nextent_args = len(args) - 3
    # Extract other arguments
    table_proxy, column, data = args[nextent_args:]

    # Handle dask compute_meta gracefully
    if len(row_orders) == 0:
        return np.empty((0,)*len(data.shape), dtype=np.bool)

    row_runs, resort = row_orders

    if resort is not None:
        data = data[resort]

    # There are other dimensions beside row
    if nextent_args > 0:
        blc, trc = zip(*args[:nextent_args])
        table_proxy._ex.submit(ndarray_putcolslice, row_runs, blc, trc,
                               table_proxy, column, data).result()
    else:
        table_proxy._ex.submit(ndarray_putcol, row_runs, table_proxy,
                               column, data).result()

    return np.full((1,) * len(data.shape), True)


def ndarray_putcol(row_runs, table_proxy, column, data):
    """ Put data into the table """
    putcol = table_proxy._table.putcol
    rr = 0

    # NOTE(sjperkins)
    # python-casacore wants to put lists of objects, but
    # because dask.array handles ndarrays we're passed
    # ndarrays of python objects (strings).
    # Without this conversion python-casacore can segfault
    # See https://github.com/ska-sa/xarray-ms/issues/42
    if data.dtype == np.object:
        data = data.tolist()

    table_proxy._acquire(WRITELOCK)

    try:
        for rs, rl in row_runs:
            putcol(column, data[rr:rr + rl], startrow=rs, nrow=rl)
            rr += rl

    finally:
        table_proxy._release(WRITELOCK)


def ndarray_putcolslice(row_runs, blc, trc, table_proxy, column, data):
    """ Put data into the table """
    putcolslice = table_proxy._table.putcolslice
    rr = 0

    # NOTE(sjperkins)
    # python-casacore wants to put lists of objects, but
    # because dask.array handles ndarrays we're passed
    # ndarrays of python objects (strings).
    # Without this conversion python-casacore can segfault
    # See https://github.com/ska-sa/xarray-ms/issues/42
    if data.dtype == np.object:
        data = data.tolist()

    table_proxy._acquire(WRITELOCK)

    try:
        for rs, rl in row_runs:
            putcolslice(column, data[rr:rr + rl], blc, trc,
                        startrow=rs, nrow=rl)
            rr += rl

    finally:
        table_proxy._release(WRITELOCK)


def write_dataset(table, datasets, columns):
    # Promote datasets to list
    if isinstance(datasets, tuple):
        datasets = list(datasets)
    else:
        datasets = [datasets]

    table_proxy = TableProxy(pt.table, table, ack=False,
                             readonly=False, lockoptions='user')

    table_columns = set(table_proxy.colnames().result())
    missing = set(columns) - table_columns

    from xarrayms.columns import infer_casa_type
    from pprint import pprint
    first_data_vars = datasets[0].variables

    for m in missing:
        (dims, var, attrs) = first_data_vars[m]

        dtype = var.dtype.type
        casa_type = infer_casa_type(dtype)
        # Dimensions other than row
        ndim = len(dims) - 1

        if ndim > 0:
            kw = {'options': 4, 'shape': var.shape[1:]}
        else:
            kw = {}

        default = "" if casa_type == "STRING" else dtype(0)
        col_desc = pt.makearrcoldesc(m, default, valuetype=casa_type,
                                     ndim=ndim, **kw)

        # An ndim of 0 seems to imply a scalar which is not the
        # same thing as not having dimensions other than row
        if ndim == 0:
            del col_desc['desc']['ndim']
            del col_desc['desc']['shape']

        pprint(col_desc)

        table_proxy.addcols(col_desc).result()

    table_name = short_table_name(table)
    writes = []

    for di, ds in enumerate(datasets):
        row_order = ds.ROWID.map_blocks(_gen_row_runs, sort_dir="write",
                                        dtype=np.object)
        data_vars = ds.variables

        for column in columns:
            try:
                column_entry = data_vars[column]
            except KeyError:
                log.warning("Ignoring '%s' not present "
                            "on dataset %d" % di)
                continue

            full_dims = column_entry.dims
            array = column_entry.var
            args = [row_order, ("row",)]

            # We only need to pass in dimension extent arrays if
            # there is more than one chunk in any of the non-row columns.
            # In that case, we can putcol, otherwise putcolslice is required
            if not all(len(c) == 1 for c in array.chunks[1:]):
                # Add extent arrays
                for d, c in zip(full_dims[1:], array.chunks[1:]):
                    args.append(dim_extents_array(d, c))
                    args.append((d,))

            # Add other variables
            args.extend([table_proxy, None,
                         column, None,
                         array, full_dims])

            # Name of the dask array representing this column
            token = dask.base.tokenize(di, args)
            name = "-".join((table_name, 'write', column, token))

            write_col = da.blockwise(putter_wrapper, full_dims,
                                     *args,
                                     # All dims shrink to 1,
                                     # a single bool is returned
                                     adjust_chunks={d: 1 for d in full_dims},
                                     name=name,
                                     dtype=np.bool)

            writes.append(write_col.ravel())

        return da.concatenate(writes)
