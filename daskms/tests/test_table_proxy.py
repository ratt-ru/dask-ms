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

from daskms.table_executor import Executor, _executor_cache
from daskms.table_proxy import (TableProxy, _table_cache, taql_factory,
                                MismatchedLocks, READLOCK, WRITELOCK, NOLOCK)
from daskms.utils import assert_liveness


def test_executor():
    """ Test the executor """
    ex = Executor()
    ex2 = Executor()
    assert ex is ex2

    ex3 = pickle.loads(pickle.dumps(ex))

    assert ex3 is ex

    assert len(_executor_cache) == 1

    assert ex.impl.submit(lambda x: x*2, 4).result() == 8
    ex.shutdown(wait=True)
    ex3.shutdown(wait=False)

    # Executor should be shutdown at this point
    with pytest.raises(RuntimeError):
        ex2.impl.submit(lambda x: x*2, 4)

    assert len(_executor_cache) == 1

    # Force collection
    del ex, ex2, ex3

    # Check that callbacks
    assert len(_executor_cache) == 0


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


@pytest.mark.parametrize("lockseq", [
    ["ar", "an", "aw", "dn", "dr", "dw"],
    ["ar", "aw", "an", "dn", "dr", "dw"],
    ["ar", "dr", "an", "dn", "aw", "dw"],
    ["ar", "aw", "an", "dn", "dr", "dw"],
    ["ar", "dr", "an", "dn", "aw", "dw"],
    ["ar", "aw", "aw", "dw", "dr", "dw"],
    ["aw", "aw", "ar", "dw", "dr", "dw"],
    pytest.param(["ar", "aw", "aw", "dr", "dw", "dr"],
                 marks=pytest.mark.xfail(raises=MismatchedLocks)),
    pytest.param(["ar", "aw", "aw", "dr", "dw"],
                 marks=pytest.mark.xfail(reason="Acquired three locks "
                                                "only released two")),
])
def test_table_proxy_locks(ms, lockseq):
    assert len(_table_cache) == 0

    table_proxy = TableProxy(pt.table, ms, readonly=False, ack=False)

    reads = 0
    writes = 0

    fn_map = {'a': table_proxy._acquire, 'd': table_proxy._release}
    lock_map = {'n': NOLOCK, 'r': READLOCK, 'w': WRITELOCK}

    for action, lock in lockseq:
        try:
            fn = fn_map[action]
        except KeyError:
            raise ValueError("Invalid action '%s'" % action)

        try:
            locktype = lock_map[lock]
        except KeyError:
            raise ValueError("Invalid lock type '%s'" % locktype)

        # Increment/decrement on acquire/release
        if action == "a":
            if locktype == READLOCK:
                reads += 1
            elif locktype == WRITELOCK:
                writes += 1
        elif action == "d":
            if locktype == READLOCK:
                reads -= 1
            elif locktype == WRITELOCK:
                writes -= 1

        fn(locktype)

        # Check invariants
        have_locks = reads + writes > 0
        assert table_proxy._readlocks == reads
        assert table_proxy._writelocks == writes
        assert table_proxy._table.haslock(table_proxy._write) is have_locks
        assert table_proxy._write is (writes > 0)

    # Check invariants
    have_locks = reads + writes > 0
    assert reads == 0
    assert writes == 0
    assert table_proxy._readlocks == reads
    assert table_proxy._writelocks == writes
    assert table_proxy._table.haslock(table_proxy._write) is have_locks
    assert table_proxy._write is (writes > 0)
