# -*- coding: utf-8 -*-
"""Handles onto CASA tables, backed by :mod:`arcae`.

This replaces the ``TableProxy``/``Executor`` pair that dask-ms previously
used.  That design existed solely because ``python-casacore`` is neither
thread safe nor GIL releasing, so every operation on a table had to be
serialised onto a single dedicated thread.

arcae removes both constraints: it opens ``ninstances`` independent
casacore table instances, fans reads out across them under shared read
locks, funnels writes through a single instance under an exclusive write
lock, and releases the GIL throughout.  A table handle is therefore just a
picklable, cached reference to an :class:`arcae.Table`, and dask's own
scheduler supplies the parallelism.
"""

from __future__ import annotations

import atexit
import logging
from typing import Any, Dict, Sequence

import numpy as np
from cacheout import LRUCache

from daskms.config import config
from daskms.multiton import MultitonMetaclass

log = logging.getLogger(__name__)


def ninstances() -> int:
    """Number of independent casacore table instances to open per table.

    arcae fans reads out across these instances, so this bounds the read
    concurrency for a single table. Defaults to the size of dask's thread
    pool, which is what will be issuing the reads.
    """
    n = config.get("casa.ninstances", None)

    if n is None:
        import multiprocessing

        try:
            import dask

            n = dask.config.get("num_workers", None)
        except ImportError:
            n = None

        if n is None:
            n = multiprocessing.cpu_count()

    return max(1, int(n))


def row_index(row_runs):
    """Convert ``(start, length)`` row runs into an arcae row index.

    A single run becomes a slice, which arcae can read without
    materialising the intervening row numbers.
    """
    if len(row_runs) == 1:
        start, length = row_runs[0]
        return slice(int(start), int(start) + int(length))

    return np.concatenate([np.arange(s, s + l) for s, l in row_runs])


def build_index(row_runs, extents=()):
    """Build an arcae index from row runs and inclusive dimension extents.

    ``extents`` are ``(blc, trc)`` pairs in the python-casacore style,
    where ``trc`` is inclusive; arcae slices exclude their stop.
    """
    return (row_index(row_runs),) + tuple(
        slice(int(blc), int(trc) + 1) for blc, trc in extents
    )


# casacore's default TaQL style is Glish, which indexes arrays from one.
# python-casacore hides this by prefixing every query with "using style
# Python"; arcae does not. Without this prefix an expression such as
# GROWID()[0] silently selects a different element -- see
# daskms.ordering.group_ordering_taql, whose __firstrow__ depends on it.
TAQL_STYLE = "USING STYLE PYTHON"


def taql_style(query: str) -> str:
    """Prefix ``query`` with dask-ms's TaQL style"""
    return f"{TAQL_STYLE} {query}"


def open_table(table: str, ninstances: int = 1, readonly: bool = True):
    """Open an existing CASA table"""
    import arcae

    return arcae.table(table, ninstances=ninstances, readonly=readonly)


def taql_table(query: str, tables: Sequence["CasaTable"] = ()):
    """Execute a TAQL query, optionally against ``tables``.

    ``$1``, ``$2``... in ``query`` refer to the entries of ``tables``.
    """
    from arcae.lib.arrow_tables import Table

    return Table.from_taql(taql_style(query), [t.instance for t in tables])


def create_ms(
    table: str,
    subtable: str = "MAIN",
    table_desc: Dict[str, Any] | None = None,
    dminfo: Dict[str, Any] | None = None,
):
    """Create a Measurement Set, or one of its standard subtables.

    arcae links a subtable into its parent's keyword set as part of
    creation, so no separate keyword write is required.
    """
    from arcae.lib.arrow_tables import Table

    # NOTE: ninstances is positional between subtable and table_desc on the
    # 0.4.0-dev branch, so these must be passed by keyword
    return Table.ms_from_descriptor(
        table, subtable=subtable, table_desc=table_desc, dminfo=dminfo
    )


def create_table(
    table: str,
    table_desc: Dict[str, Any] | None = None,
    dminfo: Dict[str, Any] | None = None,
    nrow: int = 0,
):
    """Create a plain (non Measurement Set) CASA table"""
    from arcae.lib.arrow_tables import Table

    return Table.from_descriptor(table, table_desc=table_desc, dminfo=dminfo, nrow=nrow)


@atexit.register
def _close_cached_tables():
    """Close cached tables before the interpreter shuts down.

    Each arcae table owns isolation threads. Leaving tables open until
    interpreter shutdown hangs the process, so drop them while the
    runtime is still healthy enough to join those threads.
    """
    try:
        CasaTable._CACHE.clear()
    except Exception:  # pragma: no cover - best effort at shutdown
        log.debug("Error closing cached tables at exit", exc_info=True)


def ms_descriptor(subtable: str = "MAIN", complete: bool = False):
    """Return the descriptor for a Measurement Set or one of its subtables.

    ``complete`` selects the full descriptor rather than just the
    required columns.
    """
    from arcae.lib.arrow_tables import ms_descriptor as _ms_descriptor

    return _ms_descriptor(subtable, complete=complete)


class CasaTable(metaclass=MultitonMetaclass, cache_params={"cls": LRUCache}):
    """A picklable, hashable handle onto an :class:`arcae.Table`.

    The table itself is created lazily by the factory supplied on
    construction and held in a TTL cache, so the handle can be embedded in
    a dask graph and shipped to another thread or process.  Access the
    table through :attr:`instance`.
    """

    @classmethod
    def from_table(cls, table: str, ninstances: int = 1, readonly: bool = True):
        return cls(open_table, table, ninstances=ninstances, readonly=readonly)

    @classmethod
    def from_taql(cls, query: str, tables: Sequence["CasaTable"] = ()):
        return cls(taql_table, query, tables=tuple(tables))

    @property
    def name(self) -> str:
        return self.instance.name()

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._args[0] if self._args else ''})"

    __repr__ = __str__
