"""Keep this file in sync with the codex-africanus version"""

import os
import pickle

import numpy as np
import pytest

from daskms.patterns import (
    Multiton,
    PersistentLazyProxyMultiton,
    PersistentMultiton,
    LazyProxy,
    LazyProxyMultiton)


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


class PersistentA(metaclass=PersistentMultiton):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def from_args(cls, args, kwargs):
        return cls(*args, **kwargs)

    def __reduce__(self):
        return (self.from_args, (self.args, self.kwargs))


def test_persistent_multiton():
    a = PersistentA(1)
    assert a is PersistentA(1)
    assert pickle.loads(pickle.dumps(a)) is PersistentA(1)

    assert len(PersistentA._PersistentMultiton__cache) == 1
    assert next(iter(PersistentA._PersistentMultiton__cache.values())) is a

    a.__forget_multiton__()
    assert len(PersistentA._PersistentMultiton__cache) == 0


# CASA Table Locking Modes
NOLOCK = 0
READLOCK = 1
WRITELOCK = 2

# List of CASA Table methods to proxy and the appropriate locking mode
_proxied_methods = [
    # Queries
    ("nrows", READLOCK),
    ("colnames", READLOCK),
    ("getcoldesc", READLOCK),
    ("getdminfo", READLOCK),
    ("iswritable", READLOCK),
    # Modification
    ("addrows", WRITELOCK),
    ("addcols", WRITELOCK),
    ("setmaxcachesize", WRITELOCK),
    # Reads
    ("getcol", READLOCK),
    ("getcolslice", READLOCK),
    ("getcolnp", READLOCK),
    ("getvarcol", READLOCK),
    ("getcell", READLOCK),
    ("getcellslice", READLOCK),
    ("getkeywords", READLOCK),
    ("getcolkeywords", READLOCK),
    # Writes
    ("putcol", WRITELOCK),
    ("putcolslice", WRITELOCK),
    ("putcolnp", WRITELOCK),
    ("putvarcol", WRITELOCK),
    ("putcellslice", WRITELOCK),
    ("putkeyword", WRITELOCK),
    ("putkeywords", WRITELOCK),
    ("putcolkeywords", WRITELOCK)]


def proxied_method_factory(cls, method, locktype):
    """
    Proxy pyrap.tables.table.method calls.

    Creates a private implementation function which performs
    the call locked according to to ``locktype``.

    The private implementation is accessed by a public ``method``
    which submits a call to the implementation
    on a concurrent.futures.ThreadPoolExecutor.
    """
    if locktype == NOLOCK:
        runner = __nolock_runner
    elif locktype == READLOCK:
        runner = __read_runner
    elif locktype == WRITELOCK:
        runner = __write_runner
    else:
        raise ValueError(f"Invalid locktype {locktype}")

    def public_method(self, *args, **kwargs):
        """
        Submits _impl(args, kwargs) to the executor
        and returns a Future
        """
        return self._ex.submit(runner, self.proxy, method, args, kwargs)

    public_method.__name__ = method
    public_method.__doc__ = f"Call table.{method}"

    return public_method


def __nolock_runner(proxy, method, args, kwargs):
    try:
        return getattr(proxy, method)(*args, **kwargs)
    except Exception as e:
        print(str(e))


def __read_runner(proxy, method, args, kwargs):
    proxy.lock(False)

    try:
        return getattr(proxy, method)(*args, **kwargs)
    except Exception as e:
        print(str(e))
    finally:
        proxy.unlock()


def __write_runner(proxy, method, args, kwargs):
    proxy.lock(True)

    try:
        return getattr(proxy, method)(*args, **kwargs)
    except Exception as e:
        print(str(e))
    finally:
        proxy.unlock()


class TableProxyMetaClass(Multiton):
    def __new__(cls, name, bases, dct):
        for method, locktype in _proxied_methods:
            dct[method] = proxied_method_factory(cls, method, locktype)

        return super().__new__(cls, name, bases, dct)


class TableProxy(metaclass=TableProxyMetaClass):
    def __init__(self, factory, *args, **kwargs):
        import weakref
        import concurrent.futures as cf
        import multiprocessing

        self.factory = factory
        self.args = args
        self.kwargs = kwargs
        self.proxy = proxy = PersistentLazyProxyMultiton(
                                            self.factory,
                                            *self.args,
                                            **self.kwargs)

        spawn_ctx = multiprocessing.get_context("spawn")
        # self._ex = executor = cf.ProcessPoolExecutor(1, mp_context=spawn_ctx)
        self._ex = executor = \
            LazyProxyMultiton(cf.ProcessPoolExecutor, 1, mp_context=spawn_ctx)
        # self._ex = executor = cf.ThreadPoolExecutor(1)

        weakref.finalize(self, self.finaliser, proxy, executor)

    @staticmethod
    def finaliser(proxy, executor):
        nprocess = len(executor._processes)
        list(executor.map(proxy.__forget_multiton__, [None]*nprocess))
        print(f"Finalising {proxy}")
        proxy.__forget_multiton__()
        executor.shutdown(wait=True)

    @classmethod
    def from_args(cls, factory, args, kwargs):
        return cls(factory, *args, **kwargs)

    def __reduce__(self):
        return (self.from_args, (self.factory, self.args, self.kwargs))


class Resource:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        print(f"Creating Resource in {os.getpid()} {args} {kwargs}")

    def execute(self, *args, **kwargs):
        print(f"execute in {os.getpid()} {args} {kwargs}")

    def close(self):
        print(f"Closing Resource in {os.getpid()} {self.args} {self.kwargs}")


def process_fn(proxy, *args, **kwargs):
    return proxy.execute(*args, **kwargs)


def test_persistent_multiton_in_process_pool():
    import concurrent.futures as cf
    import multiprocessing
    spawn_ctx = multiprocessing.get_context("spawn")

    pool = cf.ProcessPoolExecutor(4, mp_context=spawn_ctx)
    proxy = PersistentLazyProxyMultiton((Resource, Resource.close))

    pool.submit(process_fn, proxy, 1, 2).result()
    for r in pool.map(proxy.__forget_multiton__, [None]*len(pool._processes)):
        print(r)

    print("Shutting down")
    pool.shutdown(wait=True)


def test_ms_in_persistent_multiton(ms):
    import pyrap.tables as pt
    proxy = TableProxy(pt.table, ms, ack=True)
    print(proxy.nrows().result())

    def ranges(length, chunk):
        n = 0

        while n < length:
            yield n, chunk
            n += chunk

    futures = [proxy.getcol("TIME", startrow=s, nrow=n)
               for s, n in ranges(10, 1)]

    import concurrent.futures as cf
    for f in cf.as_completed(futures):
        print(f.result())


def test_cloudpicklable(ms):
    import pyrap.tables as pt
    proxy = TableProxy(pt.table, ms, ack=True)

    import cloudpickle
    dumped = cloudpickle.dumps(proxy)
    loaded = cloudpickle.loads(dumped)

    assert proxy is loaded


def test_serializable(ms):
    import pyrap.tables as pt
    proxy = TableProxy(pt.table, ms, ack=True)

    from distributed.protocol import serialize, deserialize
    header, frames = serialize(proxy)
    deserialized = deserialize(header, frames)

    assert proxy is deserialized


def colgetter(proxy, fn_name, col_name, startrow, nrow):
    fn = getattr(proxy, fn_name)

    return fn(col_name, startrow=startrow[0], nrow=nrow).result()


@pytest.mark.parametrize("scheduler",
                         ["sync", "threads", "processes", "distributed"])
def test_blockwise_read(ms, scheduler):
    import pyrap.tables as pt

    proxy = TableProxy(pt.table, ms, ack=True)

    import dask
    import dask.array as da
    from dask.distributed import Client, LocalCluster

    if scheduler == "distributed":

        dask.config.set({"distributed.worker.daemon": False})

        cluster = LocalCluster(
            processes=True,
            n_workers=2,
            threads_per_worker=2,
            memory_limit=0
        )

        client = Client(cluster)  # noqa

    starts = da.arange(0, 10, 1, chunks=1)
    nrow = 1

    foo = da.blockwise(colgetter, "r",
                       proxy, None,
                       "getcol", None,
                       "TIME", None,
                       starts, "r",
                       nrow, None,
                       dtype=np.float64,
                       meta=np.empty((0,), dtype=object))

    foo.compute(scheduler=scheduler, num_workers=2)


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
