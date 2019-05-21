from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np
from numpy.testing import assert_array_equal
import pyrap.tables as pt
import pytest

from xarrayms.table_executor import (TableExecutor, TableProxy,
                                     TableWrapper, MismatchedLocks)


def test_table_proxy(ms):
    proxy = TableProxy(ms)
    data = proxy.getcol("STATE_ID", startrow=0, nrow=10).result()
    new_data = np.arange(data.size, 0, -1)
    proxy.putcol("STATE_ID", new_data).result()

    result = np.empty_like(data)
    proxy.getcolnp("STATE_ID", result, startrow=0, nrow=10).result()

    assert_array_equal(new_data, result)


def test_table_executor_with_proxies(tmpdir, ms):
    from xarrayms.table_executor import _thread_local

    # Clear any state in the TableExecutor
    TableExecutor.close(wait=True)

    ms2 = os.path.join(str(tmpdir), os.path.split(ms)[1])
    shutil.copytree(ms, ms2)

    with pt.table(ms, ack=False) as T:
        time = T.getcol("TIME")

    proxy_one = TableProxy(ms)
    proxy_two = TableProxy(ms2)
    proxy_three = TableProxy(ms)

    # Extract executors from the cache
    cache = TableExecutor._TableExecutor__cache
    refcounts = TableExecutor._TableExecutor__refcounts
    ex1 = cache[ms]
    ex2 = cache[ms2]

    # table name should be in thread local, but not the wrapper
    assert ex1.submit(getattr, _thread_local, "table_name").result() == ms
    assert ex1.submit(getattr, _thread_local, "wrapper", None).result() is None
    # 2 references to ms
    assert refcounts[ms] == 2

    assert ex2.submit(getattr, _thread_local, "table_name").result() == ms2
    assert ex2.submit(getattr, _thread_local, "wrapper", None).result() is None
    # 1 reference to ms
    assert refcounts[ms2] == 1
    assert sorted(cache.keys()) == sorted([ms, ms2])

    # Request data, check that it's valid and
    # check that the wrapper has been created
    assert_array_equal(proxy_one.getcol("TIME").result(), time)
    assert_array_equal(proxy_two.getcol("TIME").result(), time)
    tab1 = ex1.submit(getattr, _thread_local, "wrapper", None).result()
    assert tab1 is not None
    tab2 = ex2.submit(getattr, _thread_local, "wrapper", None).result()
    assert tab2 is not None

    # Close the first proxy, there should be one reference to  ms now
    proxy_one.close()
    assert refcounts[ms] == 1
    assert sorted(cache.keys()) == sorted([ms, ms2])

    # Wrapper still exists on the first executor
    assert ex1.submit(getattr, _thread_local, "table_name").result() == ms
    res = ex1.submit(getattr, _thread_local, "wrapper", None).result()
    assert res is not None

    # Close the third proxy, there should be no references to ms now
    proxy_three.close()

    # Wrapper still exists on the second executor
    assert ex2.submit(getattr, _thread_local, "table_name").result() == ms2
    res = ex2.submit(getattr, _thread_local, "wrapper", None).result()
    assert res is not None

    assert ms not in refcounts
    assert sorted(cache.keys()) == [ms2]

    # Executor has been shutdown
    match_str = "cannot schedule new futures after shutdown"
    with pytest.raises(RuntimeError, match=match_str):
        ex1.submit(lambda: True).result()

    # Close the last proxy, there should be nothing left
    proxy_two.close()
    assert ms2 not in refcounts
    assert len(cache) == 0

    with pytest.raises(RuntimeError, match=match_str):
        ex2.submit(lambda: True).result()

    # Re-create
    proxy_one = TableProxy(ms)
    proxy_two = TableProxy(ms2)
    proxy_three = TableProxy(ms)

    assert sorted(cache.keys()) == sorted([ms, ms2])

    TableExecutor.close(wait=True)
    assert len(cache) == 0

    # Table's should be closed but force close before
    # the temporary directory disappears
    tab1.table.close()
    tab2.table.close()


@pytest.mark.parametrize("lockseq", [
    ["a1", "a0", "a2", "r0", "r1", "r2"],
    ["a1", "a2", "a0", "r0", "r1", "r2"],
    ["a1", "r1", "a0", "r0", "a2", "r2"],
    ["a1", "a2", "a0", "r0", "r1", "r2"],
    ["a1", "r1", "a0", "r0", "a2", "r2"],
    ["a1", "a2", "a2", "r2", "r1", "r2"],
    ["a2", "a2", "a1", "r2", "r1", "r2"],
    pytest.param(["a1", "a2", "a2", "r1", "r2", "r1"],
                 marks=pytest.mark.xfail(raises=MismatchedLocks)),
    pytest.param(["a1", "a2", "a2", "r1", "r2"],
                 marks=pytest.mark.xfail(reason="Acquired three locks "
                                                "only released two")),

])
def test_table_wrapper_locks(ms, lockseq):
    table_wrapper = TableWrapper(ms)

    reads = 0
    writes = 0

    for action, locktype in lockseq:
        locktype = int(locktype)

        if action == "a":
            if locktype == 1:
                reads += 1
            elif locktype == 2:
                writes += 1

            table_wrapper.acquire(locktype)
        elif action == "r":
            if locktype == 1:
                reads -= 1
            elif locktype == 2:
                writes -= 1

            table_wrapper.release(locktype)
        else:
            raise ValueError("Invalid action %s" % action)

        # Check invariants
        have_locks = reads + writes > 0
        assert table_wrapper.readlocks == reads
        assert table_wrapper.writelocks == writes
        assert table_wrapper.table.haslock(table_wrapper.write) is have_locks
        assert table_wrapper.write is (writes > 0)

    # Check invariants
    have_locks = reads + writes > 0
    assert reads == 0
    assert writes == 0
    assert table_wrapper.readlocks == reads
    assert table_wrapper.writelocks == writes
    assert table_wrapper.table.haslock(table_wrapper.write) is have_locks
    assert table_wrapper.write is (writes > 0)
