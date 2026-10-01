"""The row structure of a CASA table, partitioned and sorted for dask-ms.

A simpler variant of xarray-ms's ``MSv2Structure``: dask-ms needs no
(time, baseline) grid, only each partition's sorted row ids.
"""

from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import os
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt

from daskms.casa_table import CasaTable, taql_table
from daskms.multiton import Multiton
from daskms.partition import PartitionKeyT, TablePartitioner
from daskms.query import select_clause

ROW_GROUP = "__row__"
"""Special grouping column that places each row in its own partition"""

_TABLE_ROW = "__tablerow__"


@dataclasses.dataclass(frozen=True)
class PartitionData:
    """The rows of one partition of a table"""

    key: PartitionKeyT
    """``((column, value), ...)`` in grouping column order"""
    rows: npt.NDArray[np.int64]
    """Table row ids, sorted by the indexing columns"""
    exemplar_row: int
    """A representative table row, from which variably shaped
    columns take their shape if the partition has no rows"""

    @property
    def nrow(self) -> int:
        return len(self.rows)


class TableStructure(Mapping[PartitionKeyT, PartitionData]):
    """Holds the partitions of a CASA table.

    Rows are partitioned by ``group_cols`` and sorted by ``index_cols``,
    ties being broken by table row. Partitions are ordered by their first
    table row, or by their sorted order when grouping by ``"__row__"``.

    ``epoch`` is not used internally, but distinguishes :class:`Multiton`
    keys, so that a different epoch builds a new structure.
    """

    _partitions: Dict[PartitionKeyT, PartitionData]

    def __init__(
        self,
        table: CasaTable,
        group_cols: Sequence[str],
        index_cols: Sequence[str],
        taql_where: str = "",
        epoch: str = "",
    ):
        group_cols = list(group_cols)
        index_cols = list(index_cols)
        row_grouping = group_cols == [ROW_GROUP]

        if ROW_GROUP in group_cols and not row_grouping:
            raise ValueError(f"{ROW_GROUP!r} must be the only grouping column")

        partitionby = [] if row_grouping else group_cols
        sortby = [c for c in index_cols if c not in partitionby]
        index = self._read_index(table, partitionby + sortby, taql_where)

        nworkers = os.cpu_count() or 1
        partitioner = TablePartitioner(partitionby, sortby, [_TABLE_ROW])

        with cf.ThreadPoolExecutor(nworkers) as pool:
            partitions = partitioner.partition(index, pool)

        self._partitions = {}

        if row_grouping:
            rows = partitions[()][_TABLE_ROW] if partitions else []

            for row in rows:
                row = int(row)
                key = ((ROW_GROUP, row),)
                self._partitions[key] = _partition(key, np.array([row]), row)
        elif partitions:
            # Order partitions by their first table row, as TaQL's GROUPBY did
            items = [
                (int(p[_TABLE_ROW].min()), k, p[_TABLE_ROW])
                for k, p in partitions.items()
            ]

            for exemplar_row, key, rows in sorted(items, key=lambda i: i[0]):
                self._partitions[key] = _partition(key, rows, exemplar_row)
        elif not group_cols:
            # An ungrouped table always has a single, possibly empty, partition
            empty = np.empty(0, dtype=np.int64)
            self._partitions[()] = _partition((), empty, 0)

    @staticmethod
    def _read_index(
        table: CasaTable, columns: List[str], taql_where: str
    ) -> Dict[str, npt.NDArray]:
        """Read ``columns`` and the table row id of each selected row"""
        if taql_where:
            select = select_clause(columns + [f"ROWID() AS {_TABLE_ROW}"])
            query = f"{select}\nFROM\n\t$1\nWHERE\n\t{taql_where}"

            with taql_table(query, (table,)) as selection:
                return _to_numpy(selection, columns + [_TABLE_ROW])

        index = _to_numpy(table.instance, columns)
        index[_TABLE_ROW] = np.arange(table.instance.nrow(), dtype=np.int64)
        return index

    def __getitem__(self, key: PartitionKeyT) -> PartitionData:
        return self._partitions[key]

    def __iter__(self) -> Iterator[PartitionKeyT]:
        return iter(self._partitions)

    def __len__(self) -> int:
        return len(self._partitions)


def _partition(key: PartitionKeyT, rows, exemplar_row: int) -> PartitionData:
    rows = np.asarray(rows, dtype=np.int64)
    # Partitions are shared through the Multiton cache
    rows.flags.writeable = False
    return PartitionData(key, rows, exemplar_row)


def _to_numpy(table, columns: List[str]) -> Dict[str, npt.NDArray]:
    if not columns:
        return {}

    arrow_table = table.to_arrow(columns=columns)
    return {c: arrow_table[c].to_numpy() for c in columns}


StructureFactory: TypeAlias = Multiton[TableStructure]
"""A :class:`~daskms.multiton.Multiton` producing a :class:`TableStructure`"""


def structure_factory(
    table: CasaTable,
    group_cols: Sequence[str],
    index_cols: Sequence[str],
    taql_where: str = "",
    epoch: str = "",
) -> StructureFactory:
    """Return a :class:`~daskms.multiton.Multiton` producing the
    :class:`TableStructure` of ``table``"""
    return Multiton(
        TableStructure,
        table,
        tuple(group_cols),
        tuple(index_cols),
        taql_where,
        epoch,
    )


__all__: Tuple[str, ...] = (
    "PartitionData",
    "ROW_GROUP",
    "StructureFactory",
    "TableStructure",
    "structure_factory",
)
