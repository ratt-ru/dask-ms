# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import dask
import dask.array as da
from dask.array.core import normalize_chunks
from dask.highlevelgraph import HighLevelGraph
import numpy as np
import pyrap.tables as pt

from xarrayms.query import select_clause, groupby_clause, orderby_clause
from xarrayms.table_proxy import TableProxy


class GroupChunkingError(Exception):
    pass


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


def _sorted_rows(taql_proxy, startrow, nrow):
    return taql_proxy.getcol("__tablerow__",
                             startrow=startrow,
                             nrow=nrow).result()


def ordering_taql(ms, index_cols):
    select = select_clause(["ROWID() as __tablerow__",
                            "0 as __firstrow__"])
    orderby = orderby_clause(index_cols)
    query = "%s\nFROM\n\t'%s'\n%s" % (select, ms, orderby)

    return TableProxy(pt.taql, query)


def row_ordering(taql_proxy, index_cols, chunks):
    nrows = taql_proxy.nrows().result()
    chunks = normalize_chunks(chunks['row'], shape=(nrows,))
    token = dask.base.tokenize(taql_proxy, index_cols, chunks, nrows)
    name = 'rows-' + token
    layers = {}
    start = 0

    for i, c in enumerate(chunks[0]):
        layers[(name, i)] = (_sorted_rows, taql_proxy, start, c)
        start += c

    graph = HighLevelGraph.from_collections(name, layers, [])
    rows = da.Array(graph, name, chunks=chunks, dtype=np.object)
    row_runs = rows.map_blocks(_gen_row_runs, dtype=np.object)

    return rows, row_runs


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


def _group_ordering_arrays(taql_proxy, index_cols, group,
                           group_nrows, group_row_chunks):
    """
    Returns
    -------
    sorted_rows : :class:`dask.array.Array`
        Sorted table rows chunked on ``group_row_chunks``.
    row_runs : :class:`dask.array.Array`.
        Array containing (row_run, resort) tuples.
        Should not be directly computed.
        Chunked on ``group_row_chunks``.
    """
    token = dask.base.tokenize(taql_proxy, group, group_nrows)
    name = 'group-rows-' + token
    chunks = ((group_nrows,),)
    layers = {(name, 0): (_sorted_group_rows, taql_proxy, group, index_cols)}

    graph = HighLevelGraph.from_collections(name, layers, [])
    group_rows = da.Array(graph, name, chunks, dtype=np.int32)

    try:
        shape = (group_nrows,)
        group_row_chunks = normalize_chunks(group_row_chunks, shape=shape)
    except ValueError as e:
        raise
        new_ex = GroupChunkingError("%s\n"
                                    "Unable to match chunks '%s' "
                                    "with shape '%s' for group '%d'. "
                                    "This can occur if too few chunk "
                                    "dictionaries have been supplied for "
                                    "the number of groups "
                                    "and an earlier group's chunking strategy "
                                    "is applied to a later one." %
                                    (str(e), group_row_chunks, shape, group))
        raise (new_ex, None, sys.exc_info()[2])
        raise new_ex from e

    group_rows = group_rows.rechunk(group_row_chunks)
    row_runs = group_rows.map_blocks(_gen_row_runs, dtype=np.object)

    return group_rows, row_runs


def group_ordering_taql(ms, group_cols, index_cols):
    if len(group_cols) == 0:
        raise ValueError("group_ordering_taql requires "
                         "len(group_cols) > 0")
    elif len(group_cols) == 1 and group_cols[0] == "__row__":
        pass
    else:
        index_group_cols = ["GAGGR(%s) as GROUP_%s" % (c, c)
                            for c in index_cols]
        # Group Row ID's
        index_group_cols.append("GROWID() AS __tablerow__")
        # Number of rows in the group
        index_group_cols.append("GCOUNT() as __tablerows__")
        # The first row of the group
        index_group_cols.append("GROWID()[0] as __firstrow__")

        groupby = groupby_clause(group_cols)
        select = select_clause(group_cols + index_group_cols)
        query = "%s\nFROM\n\t'%s'\n%s" % (select, ms, groupby)

    return TableProxy(pt.taql, query)


def group_row_ordering(group_order_taql, group_cols, index_cols, chunks):
    nrows = group_order_taql.getcol("__tablerows__").result()

    ordering_arrays = []

    for g, nrow in enumerate(nrows):
        try:
            # Try use this group's chunking scheme
            group_chunks = chunks[g]
        except IndexError:
            # Otherwise re-use the last group's
            group_chunks = chunks[-1]

        try:
            # Extract row chunking scheme
            group_row_chunks = group_chunks['row']
        except KeyError:
            raise ValueError("No row chunking scheme "
                             "found in %s!" % group_chunks)

        ordering_arrays.append(_group_ordering_arrays(group_order_taql,
                                                      index_cols, g, nrow,
                                                      group_row_chunks))

    return ordering_arrays
