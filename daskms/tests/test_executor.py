# -*- coding: utf-8 -*-

try:
    import cPickle as pickle
except ImportError:
    import pickle

import pytest

from daskms.table_executor import Executor, _executor_cache, executor_key


def test_executor():
    """ Test the executor """
    ex = Executor()
    ex2 = Executor()
    assert ex is ex2

    ex3 = pickle.loads(pickle.dumps(ex))

    assert ex3 is ex

    assert len(_executor_cache) == 1

    assert ex.impl.submit(lambda x: x*2, 4).result() == 8
    ex.impl.shutdown(wait=True)
    ex3.impl.shutdown(wait=False)

    # Executor should be shutdown at this point
    with pytest.raises(RuntimeError):
        ex2.impl.submit(lambda x: x*2, 4)

    assert len(_executor_cache) == 1

    # Force collection
    del ex, ex2, ex3

    # Check that callbacks
    assert len(_executor_cache) == 0


def test_executor_keys():
    """ Test executor keys """
    ex = Executor("foo")
    ex2 = Executor("bar")
    ex3 = Executor("foo")
    ex4 = Executor()

    assert len(_executor_cache) == 3

    assert ex is not ex2
    assert ex is ex3
    assert pickle.loads(pickle.dumps(ex)) is ex3
    assert pickle.loads(pickle.dumps(ex2)) is not ex3

    del ex, ex2, ex3, ex4

    assert len(_executor_cache) == 0


@pytest.mark.parametrize("key, result", [
    ('/home/moriarty/test.ms/', '/home/moriarty/test.ms'),
    ('/home/moriarty/test.ms', '/home/moriarty/test.ms'),
    ('/home/moriarty/test.ms/::FIELD', '/home/moriarty/test.ms'),
    ('/home/moriarty/test.ms::FIELD', '/home/moriarty/test.ms'),
    ('/home/moriarty/test.ms::FIELD/', '/home/moriarty/test.ms'),
    ('/home/moriarty/test.ms::QUX/', '/home/moriarty/test.ms'),
])
def test_executor_key(key, result):
    assert executor_key(key) == result
