import logging
import threading
from pyrap.tables import table as Table
from pathlib import Path
from weakref import finalize


log = logging.getLogger(__name__)


def _map_create_parallel_table(cls, args, kwargs):
    """ Support pickling of kwargs in ParallelTable.__reduce__ """
    return cls(*args, **kwargs)


def _parallel_table_finalizer(_linked_tables):

    for table in _linked_tables.values():

        link_path = Path(table.name())
        table.close()
        link_path.unlink()


class ParallelTable(Table):

    def __init__(self, *args, **kwargs):

        self._args = args
        self._kwargs = kwargs

        self._linked_tables = {}
        self._table_path = args[0]  # TODO: This should be checked.

        super().__init__(*args, **kwargs)

        finalize(self, _parallel_table_finalizer, self._linked_tables)

    def __reduce__(self):
        """ Defer to _map_create_parallel_table to support kwarg pickling """
        return (
            _map_create_parallel_table,
            (ParallelTable, self._args, self._kwargs)
        )

    def getcol(self, *args, **kwargs):
        table = self._get_table(threading.get_ident())
        return table.getcol(*args, **kwargs)

    def getcolslice(self, *args, **kwargs):
        table = self._get_table(threading.get_ident())
        return table.getcolslice(*args, **kwargs)

    def getcolnp(self, *args, **kwargs):
        table = self._get_table(threading.get_ident())
        return table.getcolnp(*args, **kwargs)

    def getvarcol(self, *args, **kwargs):
        table = self._get_table(threading.get_ident())
        return table.getvarcol(*args, **kwargs)

    def getcell(self, *args, **kwargs):
        table = self._get_table(threading.get_ident())
        return table.getcell(*args, **kwargs)

    def getcellslice(self, *args, **kwargs):
        table = self._get_table(threading.get_ident())
        return table.getcellslice(*args, **kwargs)

    def getkeywords(self, *args, **kwargs):
        table = self._get_table(threading.get_ident())
        return table.getkeywords(*args, **kwargs)

    def getcolkeywords(self, *args, **kwargs):
        table = self._get_table(threading.get_ident())
        return table.getcolkeywords(*args, **kwargs)

    def _get_table(self, thread_id):

        try:
            table = self._linked_tables[thread_id]
        except KeyError:
            table_path = Path(self._table_path)
            table_name = table_path.name
            link_path = Path(f"/tmp/{thread_id}-{table_name}")

            link_path.symlink_to(table_path)

            self._linked_tables[thread_id] = table = Table(
                str(link_path),
                **self._kwargs
            )

        return table
