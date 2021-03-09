# -*- coding: utf-8 -*-

import logging
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pyrap.tables as pt

from daskms.columns import (column_metadata, ColumnMetadataError,
                            dim_extents_array, infer_dtype)
from daskms.constants import DASKMS_PARTITION_KEY
from daskms.ordering import (ordering_taql, row_ordering,
                             group_ordering_taql, group_row_ordering)
from daskms.optimisation import inlined_array
from daskms.dataset import Dataset
from daskms.table_executor import executor_key
from daskms.table import table_exists
from daskms.table_proxy import TableProxy, READLOCK
from daskms.table_schemas import lookup_table_schema
from daskms.utils import table_path_split

_DEFAULT_ROW_CHUNKS = 10000

log = logging.getLogger(__name__)


def ndarray_getcol(row_runs, table_future, column, result, dtype):
    """ Get numpy array data """
    table = table_future.result()
    getcolnp = table.getcolnp
    rr = 0

    table.lock(write=False)

    try:
        for rs, rl in row_runs:
            getcolnp(column, result[rr:rr + rl], startrow=rs, nrow=rl)
            rr += rl
    finally:
        table.unlock()

    return result


def ndarray_getcolslice(row_runs, table_future, column, result,
                        blc, trc, dtype):
    """ Get numpy array data """
    table = table_future.result()
    getcolslicenp = table.getcolslicenp
    rr = 0

    table.lock(write=False)

    try:
        for rs, rl in row_runs:
            getcolslicenp(column, result[rr:rr + rl],
                          blc=blc, trc=trc,
                          startrow=rs, nrow=rl)
            rr += rl
    finally:
        table.unlock()

    return result


def object_getcol(row_runs, table_future, column, result, dtype):
    """ Get object list data """
    table = table_future.result()
    getcol = table.getcol
    rr = 0

    table.lock(write=False)

    try:
        for rs, rl in row_runs:
            data = getcol(column, rs, rl)

            # Multi-dimensional string arrays are returned as a
            # dict with 'array' and 'shape' keys. Massage the data.
            if isinstance(data, dict):
                data = (np.asarray(data['array'], dtype=dtype)
                          .reshape(data['shape']))

            # NOTE(sjperkins)
            # Dask wants ndarrays internally, so we asarray objects
            # the returning list of objects.
            # See https://github.com/ska-sa/dask-ms/issues/42
            result[rr:rr + rl] = np.asarray(data, dtype=dtype)

            rr += rl
    finally:
        table.unlock()

    return result


def object_getcolslice(row_runs, table_future, column, result,
                       blc, trc, dtype):
    """ Get object list data """
    table = table_future.result()
    getcolslice = table.getcolslice
    rr = 0

    table.lock(write=False)

    try:
        for rs, rl in row_runs:
            data = getcolslice(column, blc, trc, startrow=rs, nrow=rl)

            # Multi-dimensional string arrays are returned as a
            # dict with 'array' and 'shape' keys. Massage the data.
            if isinstance(data, dict):
                data = (np.asarray(data['array'], dtype=dtype)
                          .reshape(data['shape']))

            # NOTE(sjperkins)
            # Dask wants ndarrays internally, so we asarray objects
            # the returning list of objects.
            # See https://github.com/ska-sa/dask-ms/issues/42
            result[rr:rr + rl] = np.asarray(data, dtype=dtype)

            rr += rl
    finally:
        table.unlock()

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
        future = table_proxy._ex.submit(io_fn, row_runs,
                                        table_proxy._table_future,
                                        column, result,
                                        blc, trc, dtype)
    # In this case, the full resolution data
    # for each row is requested, so we defer to getcol
    else:
        result = np.empty((np.sum(row_runs[:, 1]),) + col_shape, dtype=dtype)
        io_fn = (object_getcol if dtype == object
                 else ndarray_getcol)

        # Submit table I/O on executor
        future = table_proxy._ex.submit(io_fn, row_runs,
                                        table_proxy._table_future,
                                        column, result, dtype)

    # Resort result if necessary
    if resort is not None:
        return future.result()[resort]

    return future.result()


def _dataset_variable_factory(table_proxy, table_schema, select_cols,
                              exemplar_row, orders, chunks, array_suffix):
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
    array_suffix : str
        dask array string prefix

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
        except ColumnMetadataError as e:
            exc_info = logging.DEBUG >= log.getEffectiveLevel()
            log.warning("Ignoring '%s': %s", column, e,
                        exc_info=exc_info)
            continue

        full_dims = ("row",) + meta.dims
        args = [row_runs, ("row",)]

        # We only need to pass in dimension extent arrays if
        # there is more than one chunk in any of the non-row columns.
        # In that case, we can getcol, otherwise getcolslice is required
        if not all(len(c) == 1 for c in meta.chunks):
            for d, c in zip(meta.dims, meta.chunks):
                # Create an array describing the dimension chunk extents
                args.append(dim_extents_array(d, c))
                args.append((d,))

            # Disable getcolslice caching
            # https://github.com/ska-sa/dask-ms/issues/92
            # https://github.com/casacore/casacore/issues/1018
            table_proxy.setmaxcachesize(column, 1).result()

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
        name = "~".join(("read", column, array_suffix)) + "-" + token

        # Construct the array
        dask_array = da.blockwise(getter_wrapper, full_dims,
                                  *args,
                                  name=name,
                                  new_axes=new_axes,
                                  dtype=meta.dtype)

        dask_array = inlined_array(dask_array)

        # Assign into variable and dimension dataset
        dataset_vars[column] = (full_dims, dask_array)

    return dataset_vars


def _col_keyword_getter(table):
    """ Gets column keywords for all columns in table """
    return {c: table.getcolkeywords(c) for c in table.colnames()}


class DatasetFactory(object):
    def __init__(self, table, select_cols, group_cols, index_cols, **kwargs):
        if not table_exists(table):
            raise ValueError(f"'{table}' does not appear to be a CASA Table")

        chunks = kwargs.pop('chunks', [{'row': _DEFAULT_ROW_CHUNKS}])

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
        self.table_schema = kwargs.pop('table_schema', None)
        self.taql_where = kwargs.pop('taql_where', '')
        self.table_keywords = kwargs.pop('table_keywords', False)
        self.column_keywords = kwargs.pop('column_keywords', False)
        self.table_proxy = kwargs.pop('table_proxy', False)

        if len(kwargs) > 0:
            raise ValueError(f"Unhandled kwargs: {kwargs}")

    def _table_proxy_factory(self):
        return TableProxy(pt.table, self.table_path, ack=False,
                          readonly=True, lockoptions='user',
                          __executor_key__=executor_key(self.canonical_name))

    def _table_schema(self):
        return lookup_table_schema(self.canonical_name, self.table_schema)

    def _single_dataset(self, table_proxy, orders, exemplar_row=0):
        _, t, s = table_path_split(self.canonical_name)
        short_table_name = "/".join((t, s)) if s else t

        table_schema = self._table_schema()
        select_cols = set(self.select_cols or table_proxy.colnames().result())
        variables = _dataset_variable_factory(table_proxy, table_schema,
                                              select_cols, exemplar_row,
                                              orders, self.chunks[0],
                                              short_table_name)

        try:
            rowid = variables.pop("ROWID")
        except KeyError:
            coords = None
        else:
            coords = {"ROWID": rowid}

        attrs = {DASKMS_PARTITION_KEY: ()}

        return Dataset(variables, coords=coords, attrs=attrs)

    def _group_datasets(self, table_proxy, groups, exemplar_rows, orders):
        _, t, s = table_path_split(self.canonical_name)
        short_table_name = '/'.join((t, s)) if s else t
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

            # Prefix dataset
            gid_str = ",".join(str(gid) for gid in group_id)
            array_suffix = f"[{gid_str}]-{short_table_name}"

            # Create dataset variables
            group_var_dims = _dataset_variable_factory(table_proxy,
                                                       table_schema,
                                                       select_cols,
                                                       exemplar_row,
                                                       order, group_chunks,
                                                       array_suffix)

            # Extract ROWID
            try:
                rowid = group_var_dims.pop("ROWID")
            except KeyError:
                coords = None
            else:
                coords = {"ROWID": rowid}

            # Assign values for the dataset's grouping columns
            # as attributes
            partitions = tuple((c, g.dtype.name) for c, g
                               in zip(self.group_cols, group_id))
            attrs = {DASKMS_PARTITION_KEY: partitions}

            # Use python types which are json serializable
            group_id = [gid.item() for gid in group_id]
            attrs.update(zip(self.group_cols, group_id))

            datasets.append(Dataset(group_var_dims, attrs=attrs,
                                    coords=coords))

        return datasets

    def datasets(self):
        table_proxy = self._table_proxy_factory()

        # No grouping case
        if len(self.group_cols) == 0:
            order_taql = ordering_taql(table_proxy, self.index_cols,
                                       self.taql_where)
            orders = row_ordering(order_taql, self.index_cols, self.chunks[0])
            datasets = [self._single_dataset(table_proxy, orders)]
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

            datasets = [self._single_dataset(table_proxy,
                                             (row_blocks[r], run_blocks[r]),
                                             exemplar_row=er)
                        for r, er in enumerate(np_sorted_row)]
        # Grouping column case
        else:
            order_taql = group_ordering_taql(table_proxy, self.group_cols,
                                             self.index_cols, self.taql_where)
            orders = group_row_ordering(order_taql, self.group_cols,
                                        self.index_cols, self.chunks)

            groups = [order_taql.getcol(g).result()
                      for g in self.group_cols]
            # Cast to actual column dtype
            group_types = [infer_dtype(c, table_proxy.getcoldesc(c).result())
                           for c in self.group_cols]
            groups = [g.astype(t) for g, t in zip(groups, group_types)]
            exemplar_rows = order_taql.getcol("__firstrow__").result()
            assert len(orders) == len(exemplar_rows)

            datasets = self._group_datasets(table_proxy, groups,
                                            exemplar_rows, orders)

        ret = (datasets,)

        if self.table_keywords is True:
            ret += (table_proxy.getkeywords().result(),)

        if self.column_keywords is True:
            keywords = table_proxy.submit(_col_keyword_getter, READLOCK)
            ret += (keywords.result(),)

        if self.table_proxy is True:
            ret += (table_proxy,)

        if len(ret) == 1:
            return ret[0]

        return ret


def read_datasets(ms, columns, group_cols, index_cols, **kwargs):
    return DatasetFactory(ms, columns, group_cols,
                          index_cols, **kwargs).datasets()
