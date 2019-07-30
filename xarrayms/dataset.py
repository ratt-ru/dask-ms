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
from xarrayms.ordering import group_ordering_taql, row_ordering, _gen_row_runs
from xarrayms.table_proxy import TableProxy, READLOCK, WRITELOCK
from xarrayms.table_schemas import lookup_table_schema
from xarrayms.utils import short_table_name

log = logging.getLogger(__name__)


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
        table_proxy._acquire(WRITELOCK)

    return data


def _group_datasets(ms, select_cols, group_cols, groups, first_rows,
                    orders, chunks):
    table_proxy = TableProxy(pt.table, ms, ack=False,
                             readonly=True, lockoptions='user')

    table_schema = lookup_table_schema(ms, None)
    datasets = []
    group_ids = list(zip(*groups))

    assert len(group_ids) == len(orders)

    if not select_cols:
        select_cols = set(table_proxy.colnames().result()) - set(group_cols)

    it = enumerate(zip(group_ids, first_rows, orders))

    for g, (group_id, first_row, (sorted_rows, row_order)) in it:
        group_var_dims = {"ROWID": (sorted_rows, ("row",))}

        try:
            group_chunks = chunks[g]   # Get group chunking strategy
        except IndexError:
            group_chunks = chunks[-1]  # Try re-use the last group's chunks

        for column in select_cols:
            try:
                shape, dims, dim_chunks, dtype = column_metadata(column,
                                                                 table_proxy,
                                                                 table_schema,
                                                                 group_chunks)
            except ColumnMetadataError:
                log.warning("Ignoring column: '%s'", column, exc_info=True)
                continue

            assert len(dims) == len(shape), (dims, shape)

            # Name of the dask array representing this column
            name = "%s-[%s]-%s-" % (short_table_name(ms),
                                    ",".join(str(gid) for gid in group_id),
                                    column)

            # dask array dimension schema
            full_dims = ("row",) + dims

            # Construct the arguments
            args = []

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
                                      row_order, ("row",),
                                      *args,
                                      name=name,
                                      dtype=dtype)

            # Assign into variable and dimension dataset
            group_var_dims[column] = (dask_array, full_dims)

        attrs = dict(zip(group_cols, group_id))
        datasets.append(Dataset(group_var_dims, attrs=attrs))

    return datasets


def write_columns(ms, dataset, columns):
    table_proxy = TableProxy(pt.table, ms, ack=False,
                             readonly=False, lockoptions='user')
    writes = []

    rowids = dataset.ROWID
    row_order = rowids.map_blocks(_gen_row_runs, dtype=np.object)

    for column_name in columns:
        column_array = getattr(dataset, column_name)

        column_write = da.blockwise(putter_wrapper, ("row",),
                                    row_order, ("row",),
                                    table_proxy, None,
                                    column_name, None,
                                    column_array, ("row",),
                                    name="write-" + column_name + "-",
                                    dtype=np.bool)

        writes.append(column_write.ravel())

    return da.concatenate(writes)


_DEFAULT_ROW_CHUNKS = 10000


def dataset(ms, select_cols, group_cols, index_cols, chunks=None):
    # Create or promote chunks to a list of dicts
    if chunks is None:
        chunks = [{'row': _DEFAULT_ROW_CHUNKS}]
    elif isinstance(chunks, dict):
        chunks = [chunks]
    elif not isinstance(chunks, (tuple, list)):
        raise TypeError("'chunks' must be a dict or sequence of dicts")

    order_taql = group_ordering_taql(ms, group_cols, index_cols)
    orders = row_ordering(order_taql, group_cols, index_cols, chunks)

    groups = [order_taql.getcol(g).result() for g in group_cols]
    first_rows = order_taql.getcol("__firstrow__").result()
    assert len(orders) == len(first_rows)

    return _group_datasets(ms, select_cols, group_cols,
                           groups, first_rows,
                           orders, chunks)
