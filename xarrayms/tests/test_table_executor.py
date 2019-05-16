from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from xarrayms.table_executor import (TableProxy, TableWrapper, MismatchedLocks)


def test_table_proxy(ms):
    proxy = TableProxy(ms)
    data = proxy.getcol("STATE_ID", startrow=0, nrow=10).result()
    new_data = np.arange(data.size, 0, -1)
    proxy.putcol("STATE_ID", new_data).result()

    result = np.empty_like(data)
    proxy.getcolnp("STATE_ID", result, startrow=0, nrow=10).result()

    assert_array_equal(new_data, result)


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
