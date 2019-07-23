# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import dask.array as da
import numpy as np
import pyrap.tables as pt

from xarrayms.columns import column_metadata, ColumnMetadataError
from xarrayms.ordering import group_ordering_taql, row_ordering
from xarrayms.table_proxy import TableProxy

log = logging.getLogger(__name__)


def short_table_name(table_name):
    """
    Returns the last part

    Parameters
    ----------
    table_name : str
        CASA table path

    Returns
    -------
    str
        Shortened path

    """
    return os.path.split(table_name.rstrip(os.sep))[1]


class Dataset(object):
    def __init__(self, data_vars=None, attrs=None):
        self._data_vars = data_vars or {}
        self._attrs = attrs or {}

    @property
    def attrs(self):
        return self._attrs

    @property
    def variables(self):
        return self._data_vars

    def __getattr__(self, name):
        try:
            return self._data_vars[name]
        except KeyError:
            pass

        try:
            return self._attrs[name]
        except KeyError:
            raise ValueError("Invalid Attribute %s" % name)


def getter_wrapper(row_orders, io_fn, table_proxy, column, shape, dtype):
    """
    Wrapper around ``io_fn`` which should run I/O operations
    within the table_proxy's associated executor
    """
    row_runs, resort = row_orders

    # (nrows,) + shape
    result = np.empty((np.sum(row_runs[:, 1]),) + shape, dtype=dtype)

    # Submit table I/O on executor
    result = table_proxy._ex.submit(io_fn, row_runs, table_proxy,
                                    column, result, dtype).result()

    # Resort result if necessary
    if resort is not None:
        return result[resort]

    return result


def ndarray_getter(row_runs, table_proxy, column, result, dtype):
    """ Get numpy array data """
    getcolnp = table_proxy._table.getcolnp
    rr = 0

    for rs, rl in row_runs:
        getcolnp(column, result[rr:rr + rl], rs, rl)
        rr += rl

    return result


def object_getter(row_runs, table_proxy, column, result, dtype):
    """ Get object list data """
    getcol = table_proxy._table.getcol
    rr = 0

    for rs, rl in row_runs:
        result[rr:rr + rl] = np.asarray(getcol(column, rs, rl), dtype=dtype)
        rr += rl

    return result


def _group_datasets(ms, select_cols, group_cols, groups, first_rows, orders):
    table_proxy = TableProxy(pt.table, ms, readonly=True, ack=False)

    datasets = []
    group_ids = list(zip(*groups))

    assert len(group_ids) == len(orders)

    if not select_cols:
        select_cols = set(table_proxy.colnames().result()) - set(group_cols)

    it = enumerate(zip(group_ids, first_rows, orders))

    for g, (group_id, first_row, (sorted_rows, row_order)) in it:
        group_vars = {"ROWID": sorted_rows}

        for column in select_cols:
            try:
                shape, dtype = column_metadata(table_proxy, column)
            except ColumnMetadataError:
                log.warning("Ignoring column: '%s'", column, exc_info=True)
                continue

            _get = object_getter if dtype == np.object else ndarray_getter

            name = "%s-[%s]-%s-" % (short_table_name(ms),
                                    ",".join(str(gid) for gid in group_id),
                                    column)

            group_vars[column] = da.blockwise(getter_wrapper, ("row",),
                                              row_order, ("row",),
                                              _get, None,
                                              table_proxy, None,
                                              column, None,
                                              shape, None,
                                              dtype, None,
                                              name=name,
                                              dtype=dtype)

        attrs = dict(zip(group_cols, group_id))
        datasets.append(Dataset(data_vars=group_vars, attrs=attrs))

    return datasets


_DEFAULT_ROW_CHUNKS = 10000


def dataset(ms, select_cols, group_cols, index_cols, chunks):
    row_chunks = chunks.get("row", _DEFAULT_ROW_CHUNKS)
    order_taql = group_ordering_taql(ms, group_cols, index_cols)
    orders = row_ordering(order_taql, group_cols, index_cols, row_chunks)

    groups = [order_taql.getcol(g).result() for g in group_cols]
    first_rows = order_taql.getcol("__firstrow__").result()
    assert len(orders) == len(first_rows)

    return _group_datasets(ms, select_cols, group_cols,
                           groups, first_rows, orders)
