# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


try:
    import cPickle as pickle
except ImportError:
    import pickle

import pyrap.tables as pt
import pytest

from xarrayms.table_proxy import TableProxy


def test_table_proxy(ms):
    tp = TableProxy(pt.table, ms, ack=False, readonly=False)
    tq = TableProxy(pt.taql, "SELECT UNIQUE ANTENNA1 FROM '%s'" % ms)

    assert tq._table.nrows() == 3


def test_table_proxy_pickling(ms):
    proxy = TableProxy(pt.table, ms, ack=False, readonly=False)
    proxy2 = pickle.loads(pickle.dumps(proxy))
    assert proxy is proxy2


def test_table_proxy_pickling(ms):
    proxy = TableProxy(pt.taql, "SELECT UNIQUE ANTENNA1 FROM '%s'" % ms)
    proxy2 = pickle.loads(pickle.dumps(proxy))
    assert proxy is proxy2
