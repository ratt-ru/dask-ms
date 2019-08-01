# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


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
    from xarrayms.table_proxy import _table_cache
    from xarrayms.table_executor import _executor_cache
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


# https://www.zopatista.com/python/2014/03/14/cross-python-metaclasses/
def with_metaclass(mcls):
    def decorator(cls):
        body = vars(cls).copy()
        # clean out class body
        body.pop('__dict__', None)
        body.pop('__weakref__', None)
        return mcls(cls.__name__, cls.__bases__, body)

    return decorator
