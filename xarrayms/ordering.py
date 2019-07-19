# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask
import dask.array as da
from dask.highlevelgraph import HighLevelGraph
import numpy as np
import pyrap.tables as pt

from xarrayms.query import select_clause, groupby_clause
from xarrayms.table_proxy import TableProxy


def _gen_row_runs(rows, sort=True, sort_dir="read"):
    """
    Generate consecutive row runs, as well as sorting index
    if ``sort`` is True.
    """
    if sort is False:
        resort = None
    else:
        # Sort input rows and map them to their original location
        sorted_rows = np.sort(rows)
        argsort = np.searchsorted(sorted_rows, rows)

        # Generate index that recovers the original sort ordering
        if sort_dir == "read":
            resort = argsort
        elif sort_dir == "write":
            dtype = np.min_scalar_type(argsort.size)
            inv_argsort = np.empty_like(argsort, dtype=dtype)
            inv_argsort[argsort] = np.arange(argsort.size, dtype=dtype)
            resort = inv_argsort
        else:
            raise ValueError("Invalid sort_dir '%s'" % sort_dir)

        # Use sorted rows for creating row runs
        rows = sorted_rows

    diff = np.ediff1d(rows, to_begin=-10, to_end=-10)
    idx = np.nonzero(diff != 1)[0]
    start_and_len = np.empty((idx.size - 1, 2), dtype=np.int32)
    start_and_len[:, 0] = rows[idx[:-1]]
    start_and_len[:, 1] = np.diff(idx)

    return start_and_len, resort


def _group_name(column):
    return "GROUP_" + column


def _sorted_group_rows(taql_proxy, group, index_cols):
    """ Returns group rows sorted according to index_cols """
    rows = taql_proxy.getcellslice("__tablerow__", group, (-1,), (-1,))
    rows = rows.result()

    # No sorting, return early
    if len(index_cols) == 0:
        return rows

    # Sort rows according to group indexing columns
    sort_columns = [taql_proxy.getcellslice("GROUP_" + c, group,
                                            (-1,), (-1,))
                    for c in reversed(index_cols)]

    # Return sorted rows
    return rows[np.lexsort([c.result() for c in sort_columns])]


def _group_ordering_array(taql_proxy, index_cols, group,
                          group_nrows, group_row_chunks):
    """
    Returns
    -------
    order : :class:`dask.array.Array`
        ordering array, chunked on ``group_row_chunks``
    """
    token = dask.base.tokenize(taql_proxy, group, group_nrows)
    name = 'group-rows-' + token
    chunks = ((group_nrows,),)
    layers = {(name, 0): (_sorted_group_rows, taql_proxy, group, index_cols)}

    graph = HighLevelGraph.from_collections(name, layers, [])
    group_rows = da.Array(graph, name, chunks, dtype=np.int32)

    shape = (group_nrows,)
    group_row_chunks = da.core.normalize_chunks(group_row_chunks, shape=shape)
    group_rows = group_rows.rechunk(group_row_chunks)

    return group_rows.map_blocks(_gen_row_runs, dtype=np.object)


def group_ordering_taql(ms, group_cols, index_cols):
    index_group_cols = ["GAGGR(%s) as GROUP_%s" % (c, c) for c in index_cols]
    index_group_cols.append("GROWID() AS __tablerow__")
    index_group_cols.append("GCOUNT() as __tablerows__")
    index_group_cols.append("GROWID()[0] as __firstrow__")

    groupby = groupby_clause(group_cols)
    select = select_clause(group_cols + index_group_cols)
    query = "%s FROM '%s' %s" % (select, ms, groupby)

    return TableProxy(pt.taql, query)


def row_ordering(group_order_taql, group_cols, index_cols, row_chunks):
    nrows = group_order_taql.getcol("__tablerows__").result()

    return [_group_ordering_array(group_order_taql,
                                  index_cols, g, nrow,
                                  row_chunks)
            for g, nrow in enumerate(nrows)]