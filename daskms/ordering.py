# -*- coding: utf-8 -*-

import dask
import dask.array as da
from dask.array.core import normalize_chunks
from dask.highlevelgraph import HighLevelGraph
import numpy as np

from daskms.casa_table import CasaTable
from daskms.query import select_clause, groupby_clause, orderby_clause
from daskms.optimisation import cached_array


class GroupChunkingError(Exception):
    pass


def _sorted_rows(taql_proxy, startrow, nrow):
    # arcae treats an empty slice as selecting the whole dimension
    if nrow == 0:
        return np.empty(0, dtype=np.int64)

    index = (slice(startrow, startrow + nrow),)
    return taql_proxy.instance.getcol("__tablerow__", index=index)


def ordering_taql(table_proxy, index_cols, taql_where=""):
    select = select_clause(["ROWID() as __tablerow__"])
    orderby = "\n" + orderby_clause(index_cols)

    if taql_where != "":
        taql_where = f"\nWHERE\n\t{taql_where}"

    query = f"{select}\nFROM\n\t$1{taql_where}{orderby}"

    return CasaTable.from_taql(query, (table_proxy,))


def row_ordering(taql_proxy, index_cols, chunks):
    nrows = taql_proxy.instance.nrow()
    chunks = normalize_chunks(chunks["row"], shape=(nrows,))
    token = dask.base.tokenize(taql_proxy, index_cols, chunks, nrows)
    name = "rows-" + token
    layers = {}
    start = 0

    for i, c in enumerate(chunks[0]):
        layers[(name, i)] = (_sorted_rows, taql_proxy, start, c)
        start += c

    graph = HighLevelGraph.from_collections(name, layers, [])
    rows = da.Array(graph, name, chunks=chunks, dtype=np.int64)

    return cached_array(rows)


def _sorted_group_rows(taql_proxy, group, index_cols):
    """Returns group rows sorted according to index_cols"""
    # The aggregated columns are ragged -- each group holds a different
    # number of rows -- so they must be read a single row at a time
    table = taql_proxy.instance
    index = (slice(group, group + 1),)
    rows = table.getcol("__tablerow__", index=index)[0]

    # No sorting, return early
    if len(index_cols) == 0:
        return rows

    # Sort rows according to group indexing columns
    sort_columns = [
        table.getcol(f"GROUP_{c}", index=index)[0] for c in reversed(index_cols)
    ]

    # Return sorted rows
    return rows[np.lexsort(sort_columns)]


def _group_ordering_arrays(
    taql_proxy, index_cols, group, group_nrows, group_row_chunks
):
    """
    Returns
    -------
    sorted_rows : :class:`dask.array.Array`
        Sorted table rows chunked on ``group_row_chunks``.
    """
    token = dask.base.tokenize(taql_proxy, group, group_nrows)
    name = "group-rows-" + token
    chunks = ((group_nrows,),)
    layers = {(name, 0): (_sorted_group_rows, taql_proxy, group, index_cols)}

    graph = HighLevelGraph.from_collections(name, layers, [])
    group_rows = da.Array(graph, name, chunks, dtype=np.int32)
    group_rows = cached_array(group_rows)

    try:
        shape = (group_nrows,)
        group_row_chunks = normalize_chunks(group_row_chunks, shape=shape)
    except ValueError as e:
        raise GroupChunkingError(
            "%s\n"
            "Unable to match chunks '%s' "
            "with shape '%s' for group '%d'. "
            "This can occur if too few chunk "
            "dictionaries have been supplied for "
            "the number of groups "
            "and an earlier group's chunking strategy "
            "is applied to a later one." % (str(e), group_row_chunks, shape, group)
        )

    return group_rows.rechunk(group_row_chunks)


def group_ordering_taql(table_proxy, group_cols, index_cols, taql_where=""):
    if len(group_cols) == 0:
        raise ValueError("group_ordering_taql requires len(group_cols) > 0")
    else:
        index_group_cols = [f"GAGGR({c}) as GROUP_{c}" for c in index_cols]
        # Group Row ID's
        index_group_cols.append("GROWID() AS __tablerow__")
        # Number of rows in the group
        index_group_cols.append("GCOUNT() as __tablerows__")
        # The first row of the group
        index_group_cols.append("GROWID()[0] as __firstrow__")

        groupby = groupby_clause(group_cols)
        select = select_clause(group_cols + index_group_cols)

        if taql_where != "":
            taql_where = f"\nWHERE\n\t{taql_where}"

        query = f"{select}\nFROM\n\t$1{taql_where}\n{groupby}"

        return CasaTable.from_taql(query, (table_proxy,))

    raise RuntimeError("Invalid condition in group_ordering_taql")


def group_row_ordering(group_order_taql, group_cols, index_cols, chunks):
    nrows = group_order_taql.instance.getcol("__tablerows__")

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
            group_row_chunks = group_chunks["row"]
        except KeyError:
            raise ValueError(f"No row chunking scheme found in {group_chunks}!")

        ordering_arrays.append(
            _group_ordering_arrays(
                group_order_taql, index_cols, g, nrow, group_row_chunks
            )
        )

    return ordering_arrays
