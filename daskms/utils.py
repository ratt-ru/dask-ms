# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import six
import time

log = logging.getLogger(__name__)


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
            if not isinstance(c, six.string_types):
                raise TypeError("columns must be a list of strings")

        return list(columns)
    elif isinstance(columns, six.string_types):
        return [columns]

    raise TypeError("'columns' must be a string or a list of strings")


def short_table_name(table_name):
    """
    Returns the last part

    Parameters
    ----------
    table_name : str
        CASA table path

    Returns
    -------
    str
        Shortened path

    """
    return os.path.split(table_name.rstrip(os.sep))[1]


def group_cols_str(group_cols):
    return "group_cols=%s" % group_cols


def index_cols_str(index_cols):
    return "index_cols=%s" % index_cols


def select_cols_str(select_cols):
    return "select_cols=%s" % select_cols


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
                lines.append("\t%s" % str(r))

        raise ValueError("\n".join(lines))

    if executors is not None and len(_executor_cache) != executors:
        lines = ["len(_executor_cache)[%d] != %d" %
                 (len(_executor_cache), executors)]
        for i, v in enumerate(_executor_cache.values()):
            lines.append("%d: %s is referred to by "
                         "the following objects" % (i, v))

            for r in gc.get_referrers(v):
                lines.append("\t%s" % str(r))

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
