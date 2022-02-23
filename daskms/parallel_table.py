from pyrap.tables import table as Table
import logging
import threading
from pathlib import Path
from weakref import finalize


log = logging.getLogger(__name__)


# List of CASA Table methods to proxy and the appropriate locking mode
_parallel_methods = [
    ("getcol",),
    ("getcolslice",),
    ("getcolnp",),
    ("getvarcol",),
    ("getcell",),
    ("getcellslice",),
    ("getkeywords",),
    ("getcolkeywords",),
]


def parallel_method_factory(method):
    """
    Proxy pyrap.tables.table.method calls.

    Creates a private implementation function which performs
    the call locked according to to ``locktype``.

    The private implementation is accessed by a public ``method``
    which submits a call to the implementation
    on a concurrent.futures.ThreadPoolExecutor.
    """

    def _impl(table, args, kwargs):
        try:
            return getattr(table, method)(*args, **kwargs)
        except Exception:
            if logging.DEBUG >= log.getEffectiveLevel():
                log.exception("Exception in %s", method)
            raise

    _impl.__name__ = method + "_impl"
    _impl.__doc__ = ("Calls table.%s." %
                     (method))

    def public_method(self, *args, **kwargs):
        """
        Submits _impl(args, kwargs) to the executor
        and returns a Future
        """
        return _impl(self._tables[threading.get_ident()], args, kwargs)

    public_method.__name__ = method
    # public_method.__doc__ = _PROXY_DOCSTRING % method

    return public_method


class ParallelTableMetaClass(type):

    def __new__(cls, name, bases, dct):
        for (method,) in _parallel_methods:
            parallel_method = parallel_method_factory(method)
            dct[method] = parallel_method

        return type.__new__(cls, name, bases, dct)

    # def __call__(cls, *args, **kwargs):
    #     key = arg_hasher((cls,) + args + (kwargs,))

    #     with _table_lock:
    #         try:
    #             return _table_cache[key]
    #         except KeyError:
    #             instance = type.__call__(cls, *args, **kwargs)
    #             _table_cache[key] = instance
    #             return instance


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

    def getcolnp(self, *args, **kwargs):

        thread_id = threading.get_ident()

        return self.get_table(thread_id).getcolnp(*args, **kwargs)

    def get_table(self, thread_id):

        try:
            table = self._linked_tables[thread_id]
        except KeyError:
            table_path = Path(self._table_path)
            table_name = table_path.name
            link_path = Path(f"/tmp/{thread_id}-{table_name}")

            try:
                link_path.symlink_to(table_path)
            except FileExistsError:  # This should raise a warning.
                link_path.unlink()
                link_path.symlink_to(table_path)

            self._linked_tables[thread_id] = table = Table(
                str(link_path),
                **self._kwargs
            )

        return table
