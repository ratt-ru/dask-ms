# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def group_cols_str(group_cols):
    return "group_cols=%s" % group_cols


def index_cols_str(index_cols):
    return "index_cols=%s" % index_cols


def select_cols_str(select_cols):
    return "select_cols=%s" % select_cols


def assert_liveness(table_proxies, executors):
    """
    Asserts that the correct number of TableProxy
    and Executor objects are alive.
    """
    import gc
    from xarrayms.table_proxy import _table_cache
    from xarrayms.new_executor import _executor_cache

    gc.collect()

    assert len(_table_cache) == table_proxies
    assert len(_executor_cache) == executors


# https://www.zopatista.com/python/2014/03/14/cross-python-metaclasses/
def with_metaclass(mcls):
    def decorator(cls):
        body = vars(cls).copy()
        # clean out class body
        body.pop('__dict__', None)
        body.pop('__weakref__', None)
        return mcls(cls.__name__, cls.__bases__, body)

    return decorator
