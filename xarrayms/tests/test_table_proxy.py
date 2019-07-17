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

from xarrayms.table_proxy import TableProxy, Executor


def test_executor():
    ex = Executor()
    ex2 = Executor()

    assert ex is ex2

    ex3 = pickle.loads(pickle.dumps(ex))

    assert ex3 is ex

    assert ex.submit(lambda x: x*2, 4).result() == 8
    ex.shutdown(wait=True)
    ex3.shutdown(wait=False)

    # Executor should be shutdown at this point
    with pytest.raises(RuntimeError):
        ex2.submit(lambda x: x*2, 4)


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
