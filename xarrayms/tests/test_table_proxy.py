# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import cPickle as pickle
except ImportError:
    import pickle

import dask
from dask.highlevelgraph import HighLevelGraph
import dask.array as da
import numpy as np
import pyrap.tables as pt
import pytest

from xarrayms.table_proxy import (TableProxy, _table_cache,
                                  Executor, _executor_cache)


def test_executor():
    """ Test the executor """
    ex = Executor()
    ex2 = Executor()
    assert ex is ex2

    ex3 = pickle.loads(pickle.dumps(ex))

    assert ex3 is ex

    assert len(_executor_cache) == 1

    assert ex.submit(lambda x: x*2, 4).result() == 8
    ex.shutdown(wait=True)
    ex3.shutdown(wait=False)

    # Executor should be shutdown at this point
    with pytest.raises(RuntimeError):
        ex2.submit(lambda x: x*2, 4)

    assert len(_executor_cache) == 1

    # Force collection
    del ex, ex2, ex3

    # Check that callbacks
    assert len(_executor_cache) == 0


def test_table_proxy(ms):
    """ Base table proxy test """
    tp = TableProxy(pt.table, ms, ack=False, readonly=False)
    tq = TableProxy(pt.taql, "SELECT UNIQUE ANTENNA1 FROM '%s'" % ms)

    assert len(_table_cache) == 2
    assert len(_executor_cache) == 1

    assert tp._table.nrows() == 10
    assert tq._table.nrows() == 3

    del tp, tq

    assert len(_table_cache) == 0
    assert len(_executor_cache) == 0


def test_table_proxy_pickling(ms):
    """ Test table pickling """
    proxy = TableProxy(pt.table, ms, ack=False, readonly=False)
    proxy2 = pickle.loads(pickle.dumps(proxy))

    assert len(_table_cache) == 1
    assert len(_executor_cache) == 1

    assert proxy is proxy2

    del proxy, proxy2

    assert len(_table_cache) == 0
    assert len(_executor_cache) == 0


def test_taql_proxy_pickling(ms):
    """ Test taql pickling """
    proxy = TableProxy(pt.taql, "SELECT UNIQUE ANTENNA1 FROM '%s'" % ms)
    proxy2 = pickle.loads(pickle.dumps(proxy))

    assert len(_table_cache) == 1
    assert len(_executor_cache) == 1

    assert proxy is proxy2

    del proxy, proxy2

    assert len(_table_cache) == 0
    assert len(_executor_cache) == 0


def test_proxy_dask_embedding(ms):
    """
    Test that an embedded proxy in the graph stays alive
    and dies at the appropriate times
    """
    def _ant1_factory(ms):
        proxy = TableProxy(pt.table, ms, ack=False, readonly=False)
        nrows = proxy.nrows().result()

        name = 'ant1'
        row_chunk = 2
        layers = {}
        chunks = []

        for c, sr in enumerate(range(0, nrows, row_chunk)):
            er = min(sr + row_chunk, nrows)
            chunk_size = er - sr
            chunks.append(chunk_size)
            layers[(name, c)] = (proxy.getcol, "ANTENNA1", sr, chunk_size)

        # Create array
        graph = HighLevelGraph.from_collections(name, layers, [])
        ant1 = da.Array(graph, name, (tuple(chunks),), dtype=np.int32)
        # Evaluate futures
        return ant1.map_blocks(lambda f: f.result(), dtype=ant1.dtype)

    ant1 = _ant1_factory(ms)

    # Proxy and executor's are embedded in the graph
    assert len(_table_cache) == 1
    assert len(_executor_cache) == 1

    a1 = ant1.compute()
    assert isinstance(a1, np.ndarray)
    assert a1.shape == (10,)
    assert a1.dtype == np.int32

    # Delete the graph
    del ant1

    # Cache's are now clear
    assert len(_table_cache) == 0
    assert len(_executor_cache) == 0
