# -*- coding: utf-8 -*-

import dask
import dask.array as da
from dask.array.core import normalize_chunks
from dask.highlevelgraph import HighLevelGraph
import numpy as np

from daskms.query import select_clause, groupby_clause, orderby_clause
from daskms.optimisation import cached_array
from daskms.table_proxy import TableProxy, taql_factory


class GroupChunkingError(Exception):
    pass


def row_run_factory(rows, sort='auto', sort_dir="read"):
    """
    Generate consecutive row runs, as well as sorting index
    if ``sort`` is True.
    """
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.int32)

    if sort == "auto":
        # Don't sort if rows monotically increase
        sort = False if np.all(np.diff(rows) >= 0) else True

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
            raise ValueError(f"Invalid sort_dir '{sort_dir}'")

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


def ordering_taql(table_proxy, index_cols, taql_where=''):
    select = select_clause(["ROWID() as __tablerow__"])
    orderby = "\n" + orderby_clause(index_cols)

    if taql_where != '':
        taql_where = f"\nWHERE\n\t{taql_where}"

    query = f"{select}\nFROM\n\t$1{taql_where}{orderby}"

    return TableProxy(taql_factory, query, tables=[table_proxy],
                      __executor_key__=table_proxy.executor_key)


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
    rows = da.Array(graph, name, chunks=chunks, dtype=np.int64)
    rows = cached_array(rows)
    row_runs = rows.map_blocks(row_run_factory, sort_dir="read",
                               dtype=object)
    row_runs = cached_array(row_runs)

    return rows, row_runs


def _sorted_group_rows(taql_proxy, group, index_cols):
    """ Returns group rows sorted according to index_cols """
    rows = taql_proxy.getcellslice("__tablerow__", group, (-1,), (-1,))
    rows = rows.result()

    # No sorting, return early
    if len(index_cols) == 0:
        return rows

    # Sort rows according to group indexing columns
    sort_columns = [taql_proxy.getcell("GROUP_" + c, group)
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
    group_rows = cached_array(group_rows)

    try:
        shape = (group_nrows,)
        group_row_chunks = normalize_chunks(group_row_chunks, shape=shape)
    except ValueError as e:
        raise GroupChunkingError("%s\n"
                                 "Unable to match chunks '%s' "
                                 "with shape '%s' for group '%d'. "
                                 "This can occur if too few chunk "
                                 "dictionaries have been supplied for "
                                 "the number of groups "
                                 "and an earlier group's chunking strategy "
                                 "is applied to a later one." %
                                 (str(e), group_row_chunks, shape, group))

    group_rows = group_rows.rechunk(group_row_chunks)
    row_runs = group_rows.map_blocks(row_run_factory, sort_dir="read",
                                     dtype=object)

    row_runs = cached_array(row_runs)

    return group_rows, row_runs


def group_ordering_taql(table_proxy, group_cols, index_cols, taql_where=''):
    if len(group_cols) == 0:
        raise ValueError("group_ordering_taql requires "
                         "len(group_cols) > 0")
    else:
        index_group_cols = [f"GAGGR({c}) as GROUP_{c}"
                            for c in index_cols]
        # Group Row ID's
        index_group_cols.append("GROWID() AS __tablerow__")
        # Number of rows in the group
        index_group_cols.append("GCOUNT() as __tablerows__")
        # The first row of the group
        index_group_cols.append("GROWID()[0] as __firstrow__")

        groupby = groupby_clause(group_cols)
        select = select_clause(group_cols + index_group_cols)

        if taql_where != '':
            taql_where = f"\nWHERE\n\t{taql_where}"

        query = f"{select}\nFROM\n\t$1{taql_where}\n{groupby}"

        return TableProxy(taql_factory, query, tables=[table_proxy],
                          __executor_key__=table_proxy.executor_key)

    raise RuntimeError("Invalid condition in group_ordering_taql")


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
            raise ValueError(f"No row chunking scheme "
                             f"found in {group_chunks}!")

        ordering_arrays.append(_group_ordering_arrays(group_order_taql,
                                                      index_cols, g, nrow,
                                                      group_row_chunks))

    return ordering_arrays
