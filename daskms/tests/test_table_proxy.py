# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import cPickle as pickle
except ImportError:
    import pickle

from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal
import pyrap.tables as pt
import pytest

from daskms.table_proxy import TableProxy, taql_factory
from daskms.utils import assert_liveness


def test_table_proxy(ms):
    """ Base table proxy test """
    tp = TableProxy(pt.table, ms, ack=False, readonly=False)
    tq = TableProxy(pt.taql, "SELECT UNIQUE ANTENNA1 FROM '%s'" % ms)

    assert_liveness(2, 1)

    assert tp.nrows().result() == 10
    assert tq.nrows().result() == 3

    # Different tokens
    assert tokenize(tp) != tokenize(tq)

    del tp, tq

    assert_liveness(0, 0)


def test_table_proxy_pickling(ms):
    """ Test table pickling """
    proxy = TableProxy(pt.table, ms, ack=False, readonly=False)
    proxy2 = pickle.loads(pickle.dumps(proxy))

    assert_liveness(1, 1)

    # Same object and tokens
    assert proxy is proxy2
    assert tokenize(proxy) == tokenize(proxy2)

    del proxy, proxy2

    assert_liveness(0, 0)


def test_taql_proxy_pickling(ms):
    """ Test taql pickling """
    proxy = TableProxy(pt.taql, "SELECT UNIQUE ANTENNA1 FROM '%s'" % ms)
    proxy2 = pickle.loads(pickle.dumps(proxy))

    assert_liveness(1, 1)

    assert proxy is proxy2
    assert tokenize(proxy) == tokenize(proxy2)

    del proxy, proxy2
    assert_liveness(0, 0)


@pytest.mark.parametrize("reverse", [True, False])
def test_embedding_table_proxy_in_taql(ms, reverse):
    """ Test using a TableProxy to create a TAQL TableProxy """
    proxy = TableProxy(pt.table, ms, ack=False, readonly=True)
    query = "SELECT UNIQUE ANTENNA1 FROM $1"
    taql_proxy = TableProxy(taql_factory, query, tables=[proxy])
    assert_array_equal(taql_proxy.getcol("ANTENNA1").result(), [0, 1, 2])

    # TAQL and original table
    assert_liveness(2, 1)

    if reverse:
        del proxy
        # TAQL still references original table
        assert_liveness(2, 1)

        # Remove TAQL now results in everything clearing up
        del taql_proxy
        assert_liveness(0, 0)
    else:
        # Removing TAQL should leave original table
        del taql_proxy
        assert_liveness(1, 1)

        # Removing proxy removes the last
        del proxy
        assert_liveness(0, 0)


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
    assert_liveness(1, 1)

    a1 = ant1.compute()

    with pt.table(ms, readonly=False, ack=False) as T:
        assert_array_equal(a1, T.getcol("ANTENNA1"))

    # Delete the graph
    del ant1

    # Cache's are now clear
    assert_liveness(0, 0)
