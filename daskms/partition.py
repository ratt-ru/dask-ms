"""Parallel partitioning and sorting of table indexing columns.

Adapted from xarray-ms's ``xarray_ms.backend.msv2.partition``.
"""

from __future__ import annotations

import concurrent.futures as cf
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt

PartitionKeyT = Tuple[Tuple[str, Any], ...]

# The dtypes that arcae's merge_np_partitions accepts
_MERGE_DTYPES = {np.dtype(np.int32), np.dtype(np.int64), np.dtype(np.float64)}


def _mergeable(
    values: npt.NDArray,
) -> Tuple[npt.NDArray, Callable[[npt.NDArray], npt.NDArray]]:
    """Map ``values`` onto a dtype that merge_np_partitions accepts,
    preserving their order, and return a function that maps them back"""
    dtype = values.dtype

    if dtype in _MERGE_DTYPES:
        return values, lambda v: v

    # Integers that int64 represents exactly, and floats that
    # float64 does, are widened
    if dtype.kind in "biu" and dtype.itemsize <= 4:
        return values.astype(np.int64), lambda v: v.astype(dtype)

    if dtype.kind == "f" and dtype.itemsize < 8:
        return values.astype(np.float64), lambda v: v.astype(dtype)

    # Otherwise, such as for strings, replace values with their rank
    uniques, ranks = np.unique(values, return_inverse=True)
    return ranks.astype(np.int64), lambda v: uniques[v]


class TablePartitioner:
    """Partitions rows by ``partitionby`` columns and sorts each partition
    by ``sortby`` columns, followed by ``other`` columns.

    ``other`` may contain ``"row"``, which adds each row's position
    in the input. As ``"row"`` is unique, placing it last makes the
    sort order fully determined.
    """

    _partitionby: List[str]
    _sortby: List[str]
    _other: List[str]

    def __init__(
        self,
        partitionby: Sequence[str],
        sortby: Sequence[str],
        other: Sequence[str] = (),
    ):
        self._partitionby = list(partitionby)
        self._sortby = list(sortby)
        self._other = list(other)

    def partition(
        self, index: Dict[str, npt.NDArray], pool: cf.ThreadPoolExecutor
    ) -> Dict[PartitionKeyT, Dict[str, npt.NDArray]]:
        """Partition and sort the 1D arrays in ``index``.

        Returns:
            A mapping from partition key to the partition's columns, in
            ascending key order. A key is a tuple of ``(column, value)``
            pairs in ``partitionby`` order, and is empty if there are
            no ``partitionby`` columns.
        """
        index = {k: np.asarray(v) for k, v in index.items()}
        nrows = {len(v) for v in index.values()}

        if len(nrows) > 1:
            raise ValueError(f"Index array length mismatch: {sorted(nrows)}")

        if len(nrows) == 0 and "row" not in self._other:
            raise ValueError("Empty index")

        nrow = nrows.pop() if nrows else 0

        if "row" in self._other:
            index["row"] = np.arange(nrow, dtype=np.int64)

        # Order columns by partitioning, then sorting, then other columns.
        # Remaining columns are carried along, and must also take part in
        # the sort for merge_np_partitions to work
        ordered = self._partitionby + self._sortby + self._other

        if missing := set(ordered) - set(index):
            raise ValueError(f"Columns {sorted(missing)} are missing from the index")

        ordered += [c for c in index if c not in ordered]

        if nrow == 0:
            return {}

        restore = {}
        columns = {}

        for column in ordered:
            columns[column], restore[column] = _mergeable(index[column])

        nworkers = getattr(pool, "_max_workers", 1)
        chunk = (nrow + nworkers - 1) // nworkers
        starts = list(range(0, nrow, chunk))

        # Sort each chunk in parallel
        def sort_chunk(start):
            chunk_columns = {k: v[start : start + chunk] for k, v in columns.items()}
            indices = np.lexsort(tuple(chunk_columns[k] for k in reversed(ordered)))
            return {k: v[indices] for k, v in chunk_columns.items()}

        from arcae.lib.arrow_tables import merge_np_partitions

        merged = merge_np_partitions(list(pool.map(sort_chunk, starts)))

        # Find partition edges in parallel. Each chunk includes the first
        # value of the next chunk, so that an edge between chunks is found
        def find_edges(start):
            if not self._partitionby:
                return np.empty(0, dtype=np.int64)

            values = [merged[k][start : start + chunk + 1] for k in self._partitionby]
            changed = np.logical_or.reduce([np.diff(v) > 0 for v in values])
            return np.flatnonzero(changed) + start + 1

        edges = list(pool.map(find_edges, starts))
        offsets = np.concatenate([[0], *edges, [nrow]])

        merged = {k: restore[k](v) for k, v in merged.items()}
        partitions: Dict[PartitionKeyT, Dict[str, npt.NDArray]] = {}

        for start, end in zip(offsets[:-1], offsets[1:]):
            key = tuple((k, merged[k][start].item()) for k in self._partitionby)
            partitions[key] = {k: v[start:end] for k, v in merged.items()}

        return partitions
