from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import pyrap.tables as pt

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
        lockoptions = 'auto'

        try:
            userlockopt = kwargs.pop('lockoptions')
        except KeyError:
            pass
        else:
            log.warn("lockoptions='%s' ignored by TableProxy. "
                     "Locking is automatically handled "
                     "in '%s' mode", userlockopt, lockoptions)

        self._table = pt.table(table_name, lockoptions=lockoptions, **kwargs)
        self._table.unlock()

    def __getstate__(self):
        return (self._table_name, self._kwargs)

    def __setstate__(self, state):
        self.__init__(state[0], **state[1])

    def __call__(self, fn, *args, **kwargs):
        # Don't lock for these functions
        should_lock = fn not in ("close", "done")

        try:
            # Acquire a lock and call the function
            if should_lock:
                self._table.lock(write=self._write_lock)

            return getattr(self._table, fn)(*args, **kwargs)

        finally:
            # Release the lock
            if should_lock:
                self._table.unlock()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("ms")
    args = p.parse_args()

    tp = TableProxy(args.ms, readonly=True)
    tp("close")

    try:
        import cloudpickle
        ntp = cloudpickle.loads(cloudpickle.dumps(tp))
    except ImportError:
        pass
    except Exception:
        logging.warn("cloudpickle failed", exc_info=True)

    try:
        import dill
        ntp = dill.loads(dill.dumps(tp))
    except ImportError:
        pass
    except Exception:
        logging.warn("dill failed", exc_info=True)

    print(ntp("getcol", "DATA", startrow=0, nrow=1).shape)
