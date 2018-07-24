from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from dask.sizeof import sizeof, getsizeof
from xarrayms.table_cache import TableCache


log = logging.getLogger(__name__)


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

    def __init__(self, table_name, **kwargs):
        self._table_name = table_name
        # self._args = args
        self._kwargs = kwargs

        # Should we request a write-lock?
        self._write_lock = kwargs.get('readonly', True) is False

        # Force table locking mode
        self._lockoptions = 'user'

        # Remove any user supplied lockoptions
        # warning if they were present
        try:
            userlockopt = kwargs.pop('lockoptions')
        except KeyError:
            pass
        else:
            if userlockopt != self._lockoptions:
                log.warn("lockoptions='%s' ignored by TableProxy. "
                         "Locking is automatically handled "
                         "in '%s' mode", userlockopt, self._lockoptions)

    def __getstate__(self):
        return (self._table_name, self._kwargs)

    def __setstate__(self, state):
        self.__init__(state[0], **state[1])

    def __call__(self, fn, *args, **kwargs):
        # Don't lock for these functions
        fn_requires_lock = fn not in ("close", "done")

        with TableCache.instance().open(self._table_name,
                                        lockoptions=self._lockoptions,
                                        **self._kwargs) as table:
            try:
                # Acquire a lock and call the function
                if fn_requires_lock:
                    table.lock(write=self._write_lock)

                return getattr(table, fn)(*args, **kwargs)
            finally:
                # Release the lock
                if fn_requires_lock:
                    table.unlock()


@sizeof.register(TableProxy)
def sizeof_table_proxy(o):
    """ Correctly size the Table Proxy """
    return (getsizeof(o._table_name) +
            getsizeof(o._kwargs) +
            getsizeof(o._write_lock) +
            getsizeof(o._lockoptions))
