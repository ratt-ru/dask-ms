from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyrap.tables as pt

from xarrayms.table_cache import TableCache, TableWrapper, MismatchedLocks
import pytest


def test_table_cache(ms):
    with TableCache.instance().open(1, 1, ms, {'ack': False}) as table:
        ant1 = table.getcol("ANTENNA1")  # noqa
        ant2 = table.getcol("ANTENNA2")  # noqa


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
def test_locks(ms, lockseq):
    table_wrapper = TableWrapper(ms, {'ack': False, 'readonly': False})

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


def test_context_lock(ms):
    table_wrapper = TableWrapper(ms, {'ack': False})

    with table_wrapper.locked_table(0) as T:
        assert T.nrows() == 10

    with table_wrapper.locked_table(1) as T:
        assert T.nrows() == 10

    with table_wrapper.locked_table(2) as T:
        assert T.nrows() == 10
