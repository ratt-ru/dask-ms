import concurrent.futures as cf
from typing import Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt

PartitionKeyT = Tuple[Tuple[str, int | str], ...]


class TablePartitioner:
    """Partitions and sorts MSv2 indexing columns"""

    _partitionby: List[str]
    _sortby: List[str]
    _other: List[str]

    def __init__(
        self, partitionby: Sequence[str], sortby: Sequence[str], other: Sequence[str]
    ):
        self._partitionby = list(partitionby)
        self._sortby = list(sortby)
        self._other = list(other)

    def partition(
        self, index: Dict[str, npt.NDArray], pool: cf.ThreadPoolExecutor
    ) -> Dict[PartitionKeyT, Dict[str, npt.NDArray]]:
        other = set(self._other)

        if len(index) == 0:
            raise ValueError(f"Empty index")

        if len(set_nrows := set(len(v) for v in index.values())) > 1:
            raise ValueError(f"Index array length mismatch: {list(set_nrows)}")

        nrows = next(iter(set_nrows))

        try:
            other.remove("row")
            index["row"] = np.arange(nrows, dtype=np.min_scalar_type(nrows))
        except KeyError:
            pass

        nrow = len(index)
        nworkers = pool._max_workers
        chunk = (nrow + nworkers - 1) // nworkers

        # Order columns by
        #
        # 1. Partitioning columns
        # 2. Sorting columns
        # 3. Others (such as row and INTERVAL)
        # 4. Remaining columns
        #
        # 4 is needed for the merge_np_partitions to work
        ordered_columns = self._partitionby + self._sortby + self._other
        ordered_columns += list(set(index.column_names) - set(ordered_columns))

        # Create a dictionary out of the pyarrow table
        table_dict = {k: index[k].to_numpy() for k in ordered_columns}
        # Partition the data over the workers in the pool
        partitions = [
            {k: v[s : s + chunk] for k, v in table_dict.items()}
            for s in range(0, nrow, chunk)
        ]

        # Sort each partition in parallel
        def sort_partition(p):
            sort_arrays = tuple(p[k] for k in reversed(ordered_columns))
            indices = np.lexsort(sort_arrays)
            return {k: v[indices] for k, v in p.items()}

        partitions = list(pool.map(sort_partition, partitions))
        # Merge partitions
        merged = merge_np_partitions(partitions)

        # Find the edges of the group partitions in parallel by
        # partitioning the sorted merged values into chunks, including
        # the starting value of the next chunk.
        starts = list(range(0, nrow, chunk))
        group_values = [
            {
                k: v[s : s + chunk + 1]
                for k, v in merged.items()
                if k in self._partitionby
            }
            for s in starts
        ]
        assert len(starts) == len(group_values)

        # Find the group start and end points in parallel by finding edges
        def find_edges(p, s):
            diffs = [np.diff(p[v]) > 0 for v in self._partitionby]
            return np.where(np.logical_or.reduce(diffs))[0] + s + 1

        edges = list(pool.map(find_edges, group_values, starts))
        group_offsets = np.concatenate([[0]] + edges + [[nrow]])

        # Create the grouped partitions
        groups: Dict[PartitionKeyT, Dict[str, npt.NDArray]] = {}

        for start, end in zip(group_offsets[:-1], group_offsets[1:]):
            key = tuple(sorted((k, merged[k][start].item()) for k in self._partitionby))
            groups[key] = {k: v[start:end] for k, v in merged.items()}

        return groups
