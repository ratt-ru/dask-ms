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
from typing import Any, ClassVar, Dict, Sequence, Tuple
from weakref import WeakSet

import numpy as np

from daskms.config import config
from daskms.multiton import FactoryFunctionT, Multiton

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


def clear_table_cache():
    """Drop every cached table without closing it.

    Tables are deliberately not closed here. A table still in use by a
    running task is kept alive by that task's own reference and closes
    when the last one goes away. Closing it here would tear it down
    underneath that task, and in casacore that deadlocks rather than
    fails: closing a table acquires a table lock of its own
    (``TableProxy::close`` -> ``flush`` -> ``keywordSet`` ->
    ``ColumnSet::userLock``), so the close and the in-flight write end
    up waiting on each other.
    """
    Multiton.clear_cache()


def close_cached_tables():
    """Close every cached arcae table, then empty the cache.

    Only safe once nothing else is using the tables -- see
    :func:`clear_table_cache`. This exists for interpreter shutdown,
    where each arcae table owns isolation threads and leaving them to be
    joined during finalisation hangs the process.
    """
    from arcae.lib.arrow_tables import Table

    # Iterate the cache rather than the live handles: a handle's instance
    # is built on first access, so asking the handles would open tables
    # here purely to close them again.
    for entry in list(Multiton._INSTANCE_CACHE.values()):
        table = entry[0]

        if isinstance(table, Table):
            try:
                table.close()
            except Exception:  # pragma: no cover - best effort
                log.debug("Error closing %s", table, exc_info=True)

    Multiton.clear_cache(Table)


@atexit.register
def _close_cached_tables_at_exit():
    try:
        close_cached_tables()
    except Exception:  # pragma: no cover - best effort at shutdown
        log.debug("Error closing cached tables at exit", exc_info=True)


def ms_descriptor(subtable: str = "MAIN", complete: bool = False):
    """Return the descriptor for a Measurement Set or one of its subtables.

    ``complete`` selects the full descriptor rather than just the
    required columns.
    """
    from arcae.lib.arrow_tables import ms_descriptor as _ms_descriptor

    return _ms_descriptor(subtable, complete=complete)


def _rebuild_casa_table(cls, factory, args, kw, ttl):
    """Reconstruct a :class:`CasaTable` (or subclass) from pickled state"""
    return cls(factory, *args, **kw).with_ttl(ttl)


class CasaTable(Multiton):
    """A picklable, hashable handle onto an :class:`arcae.Table`.

    The table itself is created lazily by the factory supplied on
    construction and held in a TTL cache, so the handle can be embedded in
    a dask graph and shipped to another thread or process.  Access the
    table through :attr:`instance`.
    """

    # __weakref__ is not in Multiton's __slots__, and the live-handle
    # registry needs handles to be weakly referenceable
    __slots__ = ("__weakref__",)

    #: Registry of live handles. Entries disappear once the last reference
    #: to a handle is dropped, which is what lets
    #: :func:`daskms.utils.assert_liveness` reason about table lifetimes
    #: independently of the instance cache.
    _INSTANCES: ClassVar[WeakSet] = WeakSet()

    def __init__(self, factory: FactoryFunctionT, *args: Any, **kw: Any):
        super().__init__(factory, *args, **kw)
        CasaTable._INSTANCES.add(self)

    def __reduce__(self) -> Tuple[Any, ...]:
        # Multiton.__reduce__ rebuilds a plain Multiton, which would drop
        # both this class's API and its liveness registration
        return (
            _rebuild_casa_table,
            (type(self), self._factory, self._args, self._kw, self._ttl),
        )

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

    @classmethod
    def invalidate(cls, table: str) -> None:
        """Force handles onto ``table`` to reopen it on next access.

        casacore cannot resync a table across a change in its column count:
        ``Table::lock`` throws "another process changed the number of
        columns" instead. python-casacore never hit this because stock
        casacore shares one ``PlainTable`` per path within a process, so
        adding a column was visible to every handle at once. arcae's
        casacore makes that cache thread-local, precisely so its readers and
        writer are independent, which means a handle opened before an
        ``addcols`` can never catch up (ska-sa/arcae#241).

        Evicting the cached instance is enough: the handle itself is
        unchanged, and its next :attr:`instance` access reopens the table
        with the new column. Handles derived from a stale one -- a TAQL
        query over it -- are invalidated too, since their query was run
        against the table that is being dropped.

        This deliberately includes the handle that did the ``addcols``. An
        arcae table is ``ninstances`` casacore tables, and only the one that
        took the write lock saw the new column; the rest are as stale as any
        other reader, so that handle has to be reopened as well.
        """
        stale = {
            handle
            for handle in cls._INSTANCES
            if handle._factory is open_table
            and handle._args
            and handle._args[0] == table
        }

        # Derivation can nest, so keep going until nothing new is reached
        while True:
            derived = {
                handle
                for handle in cls._INSTANCES
                if handle not in stale
                and any(t in stale for t in handle._kw.get("tables", ()))
            }

            if not derived:
                break

            stale |= derived

        for handle in stale:
            handle.release()
