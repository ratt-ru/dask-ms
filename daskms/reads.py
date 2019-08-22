# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import dask
import dask.array as da
import numpy as np
import pyrap.tables as pt

from daskms.columns import (column_metadata, ColumnMetadataError,
                            dim_extents_array)
from daskms.ordering import (ordering_taql, row_ordering,
                             group_ordering_taql, group_row_ordering)
from daskms.dataset import Dataset
from daskms.table import table_exists
from daskms.table_proxy import TableProxy, READLOCK
from daskms.table_schemas import lookup_table_schema
from daskms.utils import short_table_name

_DEFAULT_ROW_CHUNKS = 10000

log = logging.getLogger(__name__)


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
            # See https://github.com/ska-sa/dask-ms/issues/42
            result[rr:rr + rl] = np.asarray(getcolslice(column, blc, trc,
                                                        startrow=rs, nrow=rl),
                                            dtype=dtype)
            rr += rl
    finally:
        table_proxy._release(READLOCK)

    return result


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


def _dataset_variable_factory(table_proxy, table_schema, select_cols,
                              exemplar_row, orders, chunks, array_prefix,
                              single_row=False):
    """
    Returns a dictionary of dask arrays representing
    a series of getcols on the appropriate table.

    Produces variables for inclusion in a Dataset.

    Parameters
    ----------
    table_proxy : :class:`daskms.table_proxy.TableProxy`
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
            meta = column_metadata(column, table_proxy, table_schema,
                                   chunks, exemplar_row)
        except ColumnMetadataError:
            log.warning("Ignoring column: '%s'", column, exc_info=True)
            continue

        full_dims = ("row",) + meta.dims
        args = [row_runs, ("row",)]

        # We only need to pass in dimension extent arrays if
        # there is more than one chunk in any of the non-row columns.
        # In that case, we can getcol, otherwise getcolslice is required
        if not all(len(c) == 1 for c in meta.chunks):
            for d, c in zip(meta.dims, meta.chunks):
                args.append(dim_extents_array(d, c))
                args.append((d,))

            new_axes = {}
        else:
            # We need to inform blockwise about the size of our
            # new dimensions as no arrays with them are supplied
            new_axes = {d: s for d, s in zip(meta.dims, meta.shape)}

        # Add other variables
        args.extend([table_proxy, None,
                     column, None,
                     meta.shape, None,
                     meta.dtype, None])

        # Name of the dask array representing this column
        token = dask.base.tokenize(args)
        name = "-".join((array_prefix, column, token))

        # Construct the array
        dask_array = da.blockwise(getter_wrapper, full_dims,
                                  *args,
                                  name=name,
                                  new_axes=new_axes,
                                  dtype=meta.dtype)

        # Squeeze out the single row if requested
        if single_row:
            dask_array = dask_array.squeeze(0)
            full_dims = full_dims[1:]

        # Assign into variable and dimension dataset
        dataset_vars[column] = (full_dims, dask_array, meta.attrs)

    return dataset_vars


class DatasetFactory(object):
    def __init__(self, table, select_cols, group_cols, index_cols, **kwargs):
        if not table_exists(table):
            raise ValueError("'%s' does not appear to be a CASA Table" % table)

        chunks = kwargs.pop('chunks', [{'row': _DEFAULT_ROW_CHUNKS}])

        # Create or promote chunks to a list of dicts
        if isinstance(chunks, dict):
            chunks = [chunks]
        elif not isinstance(chunks, (tuple, list)):
            raise TypeError("'chunks' must be a dict or sequence of dicts")

        self.table = table
        self.select_cols = select_cols
        self.group_cols = [] if group_cols is None else group_cols
        self.index_cols = [] if index_cols is None else index_cols
        self.chunks = chunks
        self.table_schema = kwargs.pop('table_schema', None)
        self.taql_where = kwargs.pop('taql_where', '')

        if len(kwargs) > 0:
            raise ValueError("Unhandled kwargs: %s" % kwargs)

    def _table_proxy(self):
        return TableProxy(pt.table, self.table, ack=False,
                          readonly=True, lockoptions='user')

    def _table_schema(self):
        return lookup_table_schema(self.table, self.table_schema)

    def _single_dataset(self, orders, single_row=False, exemplar_row=0):
        table_proxy = self._table_proxy()
        table_schema = self._table_schema()
        select_cols = set(self.select_cols or table_proxy.colnames().result())

        variables = _dataset_variable_factory(table_proxy, table_schema,
                                              select_cols, exemplar_row,
                                              orders, self.chunks[0],
                                              short_table_name(self.table),
                                              single_row=single_row)

        if single_row:
            # ROWID is assigned as an attribute
            variables.pop('ROWID', None)
            return Dataset(variables, attrs={"ROWID": exemplar_row})
        else:
            try:
                rowid = variables.pop("ROWID")
            except KeyError:
                coords = None
            else:
                coords = {"ROWID": rowid}

            return Dataset(variables, coords=coords)

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
            array_prefix = "%s-[%s]" % (short_table_name(self.table), gid_str)

            # Create dataset variables
            group_var_dims = _dataset_variable_factory(table_proxy,
                                                       table_schema,
                                                       select_cols,
                                                       exemplar_row,
                                                       order, group_chunks,
                                                       array_prefix)

            # Extract ROWID
            try:
                rowid = group_var_dims.pop("ROWID")
            except KeyError:
                coords = None
            else:
                coords = {"ROWID": rowid}

            # Assign values for the dataset's grouping columns
            # as attributes
            attrs = dict(zip(self.group_cols, group_id))
            datasets.append(Dataset(group_var_dims, attrs=attrs,
                                    coords=coords))

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


def read_datasets(ms, columns, group_cols, index_cols, **kwargs):
    return DatasetFactory(ms, columns, group_cols,
                          index_cols, **kwargs).datasets()
