"""Keep this file in sync with the codex-africanus version"""

import pickle

import numpy as np
import pytest

from daskms.patterns import (
    Multiton, LazyProxy, LazyProxyMultiton)


class DummyResource:
    def __init__(self, arg, tracker, kw=None):
        self.arg = arg
        self.tracker = tracker
        self.kw = kw
        self.value = None

    def set(self, value):
        self.value = value

    def close(self):
        print(f"Closing {self.arg}")
        self.tracker.closed = True


class Tracker:
    def __init__(self):
        self.closed = False

    def __eq__(self, other):
        return True

    def __hash__(self):
        return 0


@pytest.mark.parametrize("finalise", [True, False])
@pytest.mark.parametrize("cls", [LazyProxyMultiton, LazyProxy])
def test_lazy(cls, finalise):
    def _inner(tracker):
        if finalise:
            fn = (DummyResource, DummyResource.close)
        else:
            fn = DummyResource

        obj = cls(fn, "test.txt", tracker, kw="w")
        obj.set(5)

        assert obj.arg == "test.txt"
        assert obj.kw == "w"
        assert obj.value == 5

        obj2 = pickle.loads(pickle.dumps(obj))
        assert obj.__lazy_eq__(obj2)
        assert obj.__lazy_hash__() == obj2.__lazy_hash__()

        if cls is LazyProxyMultiton:
            assert obj is obj2
        else:
            assert obj is not obj2

    tracker = Tracker()
    assert tracker.closed is False
    _inner(tracker)
    assert tracker.closed is finalise

    array = LazyProxy(np.ndarray, [0, 1, 2])
    array.shape


def test_lazy_resource(tmp_path):
    filename = tmp_path / "data.txt"

    with open(str(filename), mode="w") as f:
        f.write("1,2,3,4,5,6,7,8,9,0")

    import dask.array as da
    import numpy as np

    def _open_file(*args, **kwargs):
        print(f"Opening {args} {kwargs}")
        return open(*args, **kwargs)

    def _close_file(f):
        print(f"Closing {f}")
        return f.close()

    file_proxy = LazyProxy((_open_file, _close_file), str(filename), mode="r")

    import multiprocessing as mp

    Pool = mp.get_context("spawn").Pool
    pool_proxy = LazyProxy(Pool, 8)

    def reader(file_proxy, pool_proxy, other):
        values = file_proxy.read().split(",")
        out = pool_proxy.apply(int, args=("123456",))  # noqa
        return np.array(list(map(int, values))) + other + out

    values = da.blockwise(reader, "r",
                          file_proxy, None,
                          pool_proxy, None,
                          da.arange(100, chunks=10), "r",
                          meta=np.empty((0,), np.object))
    values.compute(scheduler="processes")


def test_multiton():
    class A(metaclass=Multiton):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class B(metaclass=Multiton):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class C(metaclass=Multiton):
        def __init__(self, a, b):
            self.a = a
            self.b = b

    a1 = A(1, 2, 3)
    a2 = A(1, 2)
    a3 = A(1)
    b1 = B(1)

    assert a1 is A(1, 2, 3)
    assert a2 is A(1, 2)
    assert a3 is A(1)
    assert b1 is B(1)
    assert a1 is not a2
    assert a1 is not a3
    assert a1 is not b1
