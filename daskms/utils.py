# -*- coding: utf-8 -*-

from collections import OrderedDict
import logging
from pathlib import PurePath, Path
import re
import time

from dask.utils import funcname
# The numpy module may disappear during interpreter shutdown
# so explicitly import ndarray
from numpy import ndarray

from daskms.testing import in_pytest

log = logging.getLogger(__name__)


def natural_order(key):
    return tuple(int(c) if c.isdigit() else c.lower()
                 for c in re.split(r"(\d+)", str(key)))


def arg_hasher(args):
    """ Recursively hash data structures -- handles list and dicts """
    if isinstance(args, (tuple, list, set)):
        return hash(tuple(arg_hasher(v) for v in args))
    elif isinstance(args, dict):
        return hash(tuple((k, arg_hasher(v)) for k, v in sorted(args.items())))
    elif isinstance(args, ndarray):
        # NOTE(sjperkins)
        # https://stackoverflow.com/a/16592241/1611416
        # Slowish, but we shouldn't be passing
        # huge numpy arrays in the TableProxy constructor
        return hash(args.tostring())
    else:
        return hash(args)


def freeze(arg):
    if isinstance(arg, set):
        return tuple(map(freeze, sorted(arg)))
    elif isinstance(arg, (tuple, list)):
        return tuple(map(freeze, arg))
    elif isinstance(arg, (dict, OrderedDict)):
        return frozenset((k, freeze(v)) for k, v in sorted(arg.items()))
    elif isinstance(arg, ndarray):
        return freeze(arg.tolist())
    else:
        return arg


def promote_columns(columns, default):
    """
    Promotes `columns` to a list of columns.

    - None returns `default`
    - single string returns a list containing that string
    - tuple of strings returns a list of string

    Parameters
    ----------
    columns : str or list of str or None
        Table columns
    default : list of str
        Default columns

    Returns
    -------
    list of str
        List of columns
    """

    if columns is None:
        if not isinstance(default, list):
            raise TypeError("'default' must be a list")

        return default
    elif isinstance(columns, (tuple, list)):
        for c in columns:
            if not isinstance(c, str):
                raise TypeError("columns must be a list of strings")

        return list(columns)
    elif isinstance(columns, str):
        return [columns]

    raise TypeError("'columns' must be a string or a list of strings")


def table_path_split(path):
    """ Splits a table path into a (root, table, subtable) tuple """
    if not isinstance(path, PurePath):
        path = Path(path)

    root = path.parent
    parts = path.name.split("::", 1)

    if len(parts) == 1:
        table_name = parts[0]
        subtable = ""
    elif len(parts) == 2:
        table_name, subtable = parts
    else:
        raise RuntimeError("len(parts) not in (1, 2)")

    return root, table_name, subtable


def group_cols_str(group_cols):
    return f"group_cols={group_cols}"


def index_cols_str(index_cols):
    return f"index_cols={index_cols}"


def select_cols_str(select_cols):
    return f"select_cols={select_cols}"


def assert_liveness(table_proxies, executors, collect=True):
    """
    Asserts that the given number of TableProxy
    and Executor objects are alive.
    """
    from daskms.table_proxy import _table_cache
    from daskms.table_executor import _executor_cache
    import gc

    if collect:
        gc.collect()

    if table_proxies is not None and len(_table_cache) != table_proxies:
        lines = ["len(_table_cache)[%d] != %d" %
                 (len(_table_cache), table_proxies)]
        for i, v in enumerate(_table_cache.values()):
            lines.append("%d: %s is referred to by "
                         "the following objects" % (i, v))

            for r in gc.get_referrers(v):
                lines.append(f"\t{str(r)}")

        raise ValueError("\n".join(lines))

    if executors is not None and len(_executor_cache) != executors:
        lines = ["len(_executor_cache)[%d] != %d" %
                 (len(_executor_cache), executors)]
        for i, v in enumerate(_executor_cache.values()):
            lines.append("%d: %s is referred to by "
                         "the following objects" % (i, v))

            for r in gc.get_referrers(v):
                lines.append(f"\t{str(r)}")

        raise ValueError("\n".join(lines))


def log_call(fn):
    def _wrapper(*args, **kwargs):
        log.info("%s() start at %s", fn.__name__, time.clock())
        try:
            return fn(*args, **kwargs)
        except Exception:
            log.exception("%s() exception", fn.__name__)
            raise
        finally:
            log.info("%s() done at %s", fn.__name__, time.clock())

    return _wrapper


def requires(*args):
    import_errors = []
    msgs = []

    for a in args:
        if isinstance(a, ImportError):
            import_errors.append(a)
        elif isinstance(a, str):
            msgs.append(a)

    if import_errors:
        # Required dependencies are missing
        def decorator(fn):
            lines = [f"Optional extras required by "
                     f"{funcname(fn)} are missing due to "
                     f"the following ImportErrors:"]

            for i, e in enumerate(import_errors, 1):
                lines.append(f"{i}. {str(e)}")

            if msgs:
                lines.append("")
                lines.extend(msgs)

            msg = "\n".join(lines)

            def wrapper(*args, **kwargs):
                if in_pytest():
                    import pytest
                    pytest.skip(msg)
                else:
                    raise ImportError(msg) from import_errors[0]

            return wrapper
    else:
        # Return original function as is
        def decorator(fn):
            return fn

    return decorator
