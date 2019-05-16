from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
import logging

from dask.sizeof import sizeof, getsizeof
from xarrayms.table_cache import TableCache, NOLOCK, READLOCK, WRITELOCK


log = logging.getLogger(__name__)


_USER_LOCKING = 'user'


class TableProxy(object):
    """
    :class:`TableProxy` allows :class:`casacore.tables.table` objects
    to be easily embeddable in a dask graph.

    .. code-block:: python

        tp = TableProxy("WSRT.MS", readonly=True)

        with tp.read_locked() as table:
            table.nrow()
            table.getcol("DATA", startrow=0, nrow=10)
    """

    def __init__(self, table_name, **table_kwargs):
        # Set the default lockoptions
        lockopts = table_kwargs.setdefault('lockoptions', _USER_LOCKING)

        # Complain if the lock mode was non-default
        if lockopts != _USER_LOCKING:
            log.warn("lockoptions='%s' ignored by TableProxy. "
                     "Locking is automatically handled "
                     "in '%s' mode", lockopts, _USER_LOCKING)

            table_kwargs["lockoptions"] = _USER_LOCKING

        # Open read-write for simplicities sake...
        table_kwargs['readonly'] = False

        self._table_name = table_name
        self._table_kwargs = table_kwargs
        self._table_key = TableCache.register(table_name, table_kwargs)

    def close(self):
        TableCache.deregister(self._table_key)

    def __getstate__(self):
        return (self._table_name, self._table_kwargs)

    def __setstate__(self, state):
        self.__init__(state[0], **state[1])

    def __del__(self):
        self.close()

    @contextmanager
    def unlocked(self):
        with TableCache.acquire(self._table_key, NOLOCK) as table:
            yield table

    @contextmanager
    def read_locked(self):
        with TableCache.acquire(self._table_key, READLOCK) as table:
            yield table

    @contextmanager
    def write_locked(self):
        with TableCache.acquire(self._table_key, WRITELOCK) as table:
            yield table


@sizeof.register(TableProxy)
def sizeof_table_proxy(o):
    """ Correctly size the Table Proxy """
    return (getsizeof(o._table_name) +
            getsizeof(o._table_kwargs) +
            getsizeof(o._table_key))
