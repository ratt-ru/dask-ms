from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from dask.sizeof import sizeof, getsizeof
from xarrayms.table_cache import TableCache


log = logging.getLogger(__name__)


_LOCK_MODE = 'user'


class TableProxy(object):
    """
    :class:`TableProxy` allows :class:`casacore.tables.table` objects
    to be easily embeddable in a dask graph.

    .. code-block:: python

        tp = TableProxy("WSRT.MS", readonly=True)
        tp("nrow")
        tp("getcol", "DATA", startrow=0, nrow=10)

    Methods on :class:`casacore.tables.table` are accessed by passing
    the method name, args and kwargs to the
    :func:`TableProxy.__call__` function.
    This allows the proxy to perform per-process locking when
    the underlying file is accessed.

    """

    def __init__(self, table_name, **table_kwargs):
        # Set the default lockoptions
        lockopts = table_kwargs.setdefault('lockoptions', _LOCK_MODE)

        # Complain if the lock mode was non-default
        if lockopts != _LOCK_MODE:
            log.warn("lockoptions='%s' ignored by TableProxy. "
                     "Locking is automatically handled "
                     "in '%s' mode", lockopts, _LOCK_MODE)

            table_kwargs["lockoptions"] = _LOCK_MODE

        self._table_name = table_name
        self._table_kwargs = table_kwargs

        # Should we request a write-lock?
        self._write_lock = table_kwargs.get('readonly', True) is False

        # Compute the table key once
        self._table_key = hash((table_name, frozenset(table_kwargs.items())))

    def __getstate__(self):
        return (self._table_name, self._table_kwargs)

    def __setstate__(self, state):
        self.__init__(state[0], **state[1])

    def __call__(self, fn, *args, **kwargs):
        # Don't lock for these functions
        fn_requires_lock = fn not in ("close", "done")
        lockopt = 1 + int(self._write_lock) if fn_requires_lock else 0

        with TableCache.instance().open(self._table_key, lockopt,
                                        self._table_name,
                                        self._table_kwargs) as table:
            return getattr(table, fn)(*args, **kwargs)


@sizeof.register(TableProxy)
def sizeof_table_proxy(o):
    """ Correctly size the Table Proxy """
    return (getsizeof(o._table_name) +
            getsizeof(o._table_kwargs) +
            getsizeof(o._write_lock))
