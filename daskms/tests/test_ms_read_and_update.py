# -*- coding: utf-8 -*-

import sys
import types

import dask
import dask.array as da
import numpy as np
import pytest
from numpy.testing import assert_array_equal

try:
    from dask.optimization import key_split
except ImportError:
    from dask.utils import key_split

from daskms.array_api_utils import _issubclass_fast
from daskms.constants import DASKMS_PARTITION_KEY
from daskms.dask_ms import xds_from_ms, xds_from_table, xds_to_table
from daskms.patterns import lazy_import
from daskms.query import orderby_clause, where_clause
from daskms.table_proxy import TableProxy, taql_factory
from daskms.utils import (
    group_cols_str,
    index_cols_str,
    select_cols_str,
    table_path_split,
)

ct = lazy_import("casacore.tables")

PY_37_GTE = sys.version_info[:2] >= (3, 7)


def _make_linear_ramp_array(shape, dtype):
    return np.reshape(np.arange(np.prod(shape), dtype=dtype), shape)


@pytest.mark.parametrize(
    "group_cols",
    [
        [],
        ["FIELD_ID", "DATA_DESC_ID"],
        ["DATA_DESC_ID"],
        ["DATA_DESC_ID", "SCAN_NUMBER"],
    ],
    ids=group_cols_str,
)
@pytest.mark.parametrize(
    "index_cols",
    [
        ["ANTENNA2", "ANTENNA1", "TIME"],
        ["TIME", "ANTENNA1", "ANTENNA2"],
        ["ANTENNA1", "ANTENNA2", "TIME"],
    ],
    ids=index_cols_str,
)
@pytest.mark.parametrize(
    "select_cols", [["TIME", "ANTENNA1", "DATA"]], ids=select_cols_str
)
def test_ms_read(ms, group_cols, index_cols, select_cols):
    xds = xds_from_ms(
        ms,
        columns=select_cols,
        group_cols=group_cols,
        index_cols=index_cols,
        chunks={"row": 2},
    )

    order = orderby_clause(index_cols)
    np_column_data = []

    with TableProxy(ct.table, ms, lockoptions="auto", ack=False) as T:
        for ds in xds:
            assert "ROWID" in ds.coords
            group_col_values = [ds.attrs[a] for a in group_cols]
            where = where_clause(group_cols, group_col_values)
            query = f"SELECT * FROM $1 {where} {order}"

            with TableProxy(taql_factory, query, tables=[T]) as Q:
                column_data = {c: Q.getcol(c).result() for c in select_cols}
                np_column_data.append(column_data)

    del T

    for d, (ds, column_data) in enumerate(zip(xds, np_column_data)):
        for c in select_cols:
            dask_data = ds.data_vars[c].data.compute()
            assert_array_equal(column_data[c], dask_data)


@pytest.mark.parametrize(
    "group_cols",
    [
        [],
        ["FIELD_ID", "DATA_DESC_ID"],
        ["DATA_DESC_ID"],
        ["DATA_DESC_ID", "SCAN_NUMBER"],
    ],
    ids=group_cols_str,
)
@pytest.mark.parametrize(
    "index_cols",
    [
        ["ANTENNA2", "ANTENNA1", "TIME"],
        ["TIME", "ANTENNA1", "ANTENNA2"],
        ["ANTENNA1", "ANTENNA2", "TIME"],
    ],
    ids=index_cols_str,
)
@pytest.mark.parametrize("select_cols", [["DATA", "STATE_ID"]])
def test_ms_update(ms, group_cols, index_cols, select_cols):
    # Zero everything to be sure
    with TableProxy(ct.table, ms, readonly=False, lockoptions="auto", ack=False) as T:
        nrows = T.nrows().result()
        T.putcol("STATE_ID", np.full(nrows, 0, dtype=np.int32)).result()
        data = np.zeros_like(T.getcol("DATA").result())
        data_dtype = data.dtype
        T.putcol("DATA", data).result()

    xds = xds_from_ms(
        ms,
        columns=select_cols,
        group_cols=group_cols,
        index_cols=index_cols,
        chunks={"row": 2},
    )

    written_states = []
    written_data = []
    writes = []

    # Write out STATE_ID and DATA
    for i, ds in enumerate(xds):
        dims = ds.sizes
        chunks = ds.chunks
        state = da.arange(i, i + dims["row"], chunks=chunks["row"])
        state = state.astype(np.int32)
        written_states.append(state)

        data = da.arange(i, i + dims["row"] * dims["chan"] * dims["corr"])
        data = data.reshape(dims["row"], dims["chan"], dims["corr"])
        data = data.rechunk((chunks["row"], chunks["chan"], chunks["corr"]))
        data = data.astype(data_dtype)
        written_data.append(data)

        nds = ds.assign(
            STATE_ID=(("row",), state), DATA=(("row", "chan", "corr"), data)
        )

        (write,) = xds_to_table(nds, ms, ["STATE_ID", "DATA"])

        for k, _ in nds.attrs[DASKMS_PARTITION_KEY]:
            assert getattr(write, k) == getattr(nds, k)

        writes.append(write)

    # Do all writes in parallel
    dask.compute(writes)

    xds = xds_from_ms(
        ms,
        columns=select_cols,
        group_cols=group_cols,
        index_cols=index_cols,
        chunks={"row": 2},
    )

    # Check that state and data have been correctly written
    it = enumerate(zip(xds, written_states, written_data))
    for i, (ds, state, data) in it:
        assert_array_equal(ds.STATE_ID.data, state)
        assert_array_equal(ds.DATA.data, data)


@pytest.mark.parametrize(
    "index_cols",
    [
        ["ANTENNA2", "ANTENNA1", "TIME"],
        ["TIME", "ANTENNA1", "ANTENNA2"],
        ["ANTENNA1", "ANTENNA2", "TIME"],
    ],
    ids=index_cols_str,
)
def test_row_query(ms, index_cols):
    T = TableProxy(ct.table, ms, readonly=True, lockoptions="auto", ack=False)

    # Get the expected row ordering by lexically
    # sorting the indexing columns
    cols = [(name, T.getcol(name).result()) for name in index_cols]
    expected_rows = np.lexsort(tuple(c for n, c in reversed(cols)))

    del T

    xds = xds_from_ms(
        ms,
        columns=index_cols,
        group_cols="__row__",
        index_cols=index_cols,
        chunks={"row": 2},
    )

    actual_rows = da.concatenate([ds.ROWID.data for ds in xds])
    assert_array_equal(actual_rows, expected_rows)


@pytest.mark.parametrize(
    "index_cols", [["TIME", "ANTENNA1", "ANTENNA2"]], ids=index_cols_str
)
def test_taql_where(ms, index_cols):
    # three cases test here, corresponding to the
    # if-elif-else ladder in xds_from_table

    # No group_cols case
    xds = xds_from_table(
        ms, taql_where="FIELD_ID >= 0 AND FIELD_ID < 2", columns=["FIELD_ID"]
    )

    assert len(xds) == 1
    assert_array_equal(xds[0].FIELD_ID.data, [0, 0, 0, 1, 1, 1, 1])

    # Group columns case
    xds = xds_from_table(
        ms,
        taql_where="FIELD_ID >= 0 AND FIELD_ID < 2",
        group_cols=["DATA_DESC_ID", "SCAN_NUMBER"],
        columns=["FIELD_ID"],
    )

    assert len(xds) == 2

    # Check group id's
    assert xds[0].DATA_DESC_ID == 0 and xds[0].SCAN_NUMBER == 0
    assert xds[1].DATA_DESC_ID == 0 and xds[1].SCAN_NUMBER == 1

    # Check field id's in each group
    fields = da.concatenate([ds.FIELD_ID.data for ds in xds])
    assert_array_equal(fields, [0, 0, 1, 1, 0, 1, 1])

    # Group columns case
    xds = xds_from_table(
        ms,
        taql_where="FIELD_ID >= 0 AND FIELD_ID < 2",
        group_cols=["DATA_DESC_ID", "FIELD_ID"],
        columns=["FIELD_ID"],
    )

    assert len(xds) == 2

    # Check group id's, no DATA_DESC_ID == 1 because it only
    # contains FIELD_ID == 2
    assert xds[0].DATA_DESC_ID == 0 and xds[0].FIELD_ID == 0
    assert xds[1].DATA_DESC_ID == 0 and xds[1].FIELD_ID == 1

    # Group on each row
    xds = xds_from_table(
        ms,
        taql_where="FIELD_ID >= 0 AND FIELD_ID < 2",
        group_cols=["__row__"],
        columns=["FIELD_ID"],
    )

    assert len(xds) == 7

    fields = da.concatenate([ds.FIELD_ID.data for ds in xds])
    assert_array_equal(fields, [0, 0, 0, 1, 1, 1, 1])


def _proc_map_fn(args):
    import dask.threaded as dt

    # No dask pools are spun up
    with dt.pools_lock:
        assert dt.default_pool is None
        assert len(dt.pools) == 0

    try:
        ms, i = args
        xds = xds_from_ms(ms, columns=["STATE_ID"], group_cols=["FIELD_ID"])
        xds[i] = xds[i].assign(STATE_ID=(("row",), xds[i].STATE_ID.data + i))
        write = xds_to_table(xds[i], ms, ["STATE_ID"])
        dask.compute(write)
    except Exception as e:
        raise e

    return True


@pytest.mark.parametrize("nprocs", [3])
def test_multiprocess_table(ms, nprocs):
    from multiprocessing import get_context

    pool = get_context("spawn").Pool(nprocs)

    try:
        args = [tuple((ms, i)) for i in range(nprocs)]
        assert all(pool.map(_proc_map_fn, args))
    finally:
        pool.close()


@pytest.mark.parametrize(
    "group_cols",
    [["FIELD_ID", "DATA_DESC_ID"], ["DATA_DESC_ID", "SCAN_NUMBER"]],
    ids=group_cols_str,
)
@pytest.mark.parametrize(
    "index_cols", [["TIME", "ANTENNA1", "ANTENNA2"]], ids=index_cols_str
)
def test_multireadwrite(ms, group_cols, index_cols):
    xds = xds_from_ms(ms, group_cols=group_cols, index_cols=index_cols)

    nds = [ds.copy() for ds in xds]
    writes = [
        xds_to_table(sds, ms, [k for k in sds.data_vars.keys() if k != "ROWID"])
        for sds in nds
    ]

    da.compute(writes)


def test_column_promotion(ms):
    """Test singleton columns promoted to lists"""
    xds = xds_from_ms(ms, group_cols="SCAN_NUMBER", columns=("DATA",))

    for ds in xds:
        assert "DATA" in ds.data_vars
        assert "SCAN_NUMBER" in ds.attrs
        assert ds.attrs[DASKMS_PARTITION_KEY] == (("SCAN_NUMBER", "int32"),)


def test_read_array_names(ms):
    _, short_name, _ = table_path_split(ms)
    datasets = xds_from_ms(ms)

    for ds in datasets:
        for k, v in ds.data_vars.items():
            product = "~[" + str(ds.FIELD_ID) + "," + str(ds.DATA_DESC_ID) + "]"
            prefix = "".join(("read~", k, product))
            assert key_split(v.data.name) == prefix


def test_write_array_names(ms, tmp_path):
    _, short_name, _ = table_path_split(ms)
    datasets = xds_from_ms(ms)

    out_table = str(tmp_path / short_name)

    writes = xds_to_table(datasets, out_table, "ALL")

    for ds in writes:
        for k, v in ds.data_vars.items():
            prefix = "".join(("write~", k))
            assert key_split(v.data.name) == prefix


def test_mismatched_rowid(ms):
    xdsl = xds_from_ms(
        ms, group_cols=["SCAN_NUMBER"], chunks={"row": -1}, columns=["DATA"]
    )

    # NOTE: Remove this line to make this test pass.
    xds = xdsl[0]
    xds = xds.assign_coords(
        **{"ROWID": (("row",), da.arange(xds.sizes["row"], chunks=2))}
    )

    with pytest.raises(ValueError, match="ROWID shape and/or chunking"):
        xds_to_table(xds, ms, columns=["DATA"])


def test_request_rowid(ms):
    xdsl = xds_from_ms(ms, columns=["TIME", "ROWID"])  # noqa


class _ArrayLike:
    """Minimal array API compatible object, standing in for JAX, CuPy, PyTorch, etc."""

    def __init__(self, data):
        self._data = np.asarray(data)

    def __array_namespace__(self, api_version=None):
        return np

    def __array__(self, dtype=None, copy=None):
        return self._data if dtype is None else self._data.astype(dtype)

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self):
        return self._data.ndim

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)


def test_array_protocol_write(ms):
    """Arrays implementing the array API (JAX, CuPy, PyTorch, ...) must be writable.

    Uses da.from_array: some libraries (JAX, zarr) are converted to numpy
    eagerly during graph construction, so this test covers the case where
    the chunk arrives at putter_wrapper already converted.
    """
    xds = xds_from_ms(ms, columns=["DATA"], group_cols=[], chunks={"row": 2})[0]
    dims = xds.sizes
    chunks = xds.chunks

    array_like = _ArrayLike(
        _make_linear_ramp_array(
            (dims["row"], dims["chan"], dims["corr"]), dtype=np.complex64
        )
    )
    da_data = da.from_array(
        array_like, chunks=(chunks["row"], dims["chan"], dims["corr"])
    )

    nds = xds.assign(DATA=(("row", "chan", "corr"), da_data))
    (write,) = xds_to_table(nds, ms, ["DATA"])
    dask.compute(write)

    result = xds_from_ms(ms, columns=["DATA"], group_cols=[], chunks={"row": 2})[0]
    assert_array_equal(
        result.DATA.data, _make_linear_ramp_array(result.DATA.shape, result.DATA.dtype)
    )


def test_map_blocks_array_protocol_write(ms):
    """da.map_blocks returning non-numpy chunks must be writable.

    This is the tab-sim / JAX pattern: a JAX-jitted function is wrapped in
    map_blocks, its output (a jax.Array) becomes the raw dask chunk and is
    never eagerly converted to numpy.  putter_wrapper must handle it via
    is_array_api_obj detection.
    """
    xds = xds_from_ms(ms, columns=["DATA"], group_cols=[], chunks={"row": 2})[0]
    dims = xds.sizes
    chunks = xds.chunks

    np_data = _make_linear_ramp_array(
        (dims["row"], dims["chan"], dims["corr"]), dtype=np.complex64
    )
    da_input = da.from_array(
        np_data, chunks=(chunks["row"], dims["chan"], dims["corr"])
    )

    # Simulate a JAX/torch function that returns non-numpy chunks
    da_data = da.map_blocks(
        lambda block: _ArrayLike(block), da_input, dtype=np_data.dtype
    )

    nds = xds.assign(DATA=(("row", "chan", "corr"), da_data))
    (write,) = xds_to_table(nds, ms, ["DATA"])
    dask.compute(write)

    result = xds_from_ms(ms, columns=["DATA"], group_cols=[], chunks={"row": 2})[0]
    assert_array_equal(result.DATA.data, np_data)


def test_from_delayed_array_protocol_write(ms):
    """da.from_delayed with non-numpy chunks must be writable.

    This is the PyTorch pattern: libraries whose dtype objects are not
    numpy-compatible must be wrapped via from_delayed with an explicit dtype.
    The chunk is never inspected at graph construction time and arrives at
    putter_wrapper unconverted.
    """
    xds = xds_from_ms(ms, columns=["DATA"], group_cols=[], chunks={"row": 2})[0]
    dims = xds.sizes
    chunks = xds.chunks

    np_data = _make_linear_ramp_array(
        (dims["row"], dims["chan"], dims["corr"]), dtype=np.complex64
    )
    row_chunks = chunks["row"]
    nchan, ncorr = dims["chan"], dims["corr"]

    parts, row = [], 0
    for rc in row_chunks:
        chunk = _ArrayLike(np_data[row : row + rc])
        parts.append(
            da.from_delayed(
                dask.delayed(chunk), shape=(rc, nchan, ncorr), dtype=np_data.dtype
            )
        )
        row += rc
    da_data = da.concatenate(parts, axis=0)

    nds = xds.assign(DATA=(("row", "chan", "corr"), da_data))
    (write,) = xds_to_table(nds, ms, ["DATA"])
    dask.compute(write)

    result = xds_from_ms(ms, columns=["DATA"], group_cols=[], chunks={"row": 2})[0]
    assert_array_equal(result.DATA.data, np_data)


def test_delayed_chain_array_protocol_write(ms):
    """Chained dask.delayed returning non-numpy chunks must be writable.

    Represents a deep computation graph where the final step of each chain
    returns a non-numpy array-protocol object, as would happen when multiple
    delayed JAX operations are composed before writing.
    """
    xds = xds_from_ms(ms, columns=["DATA"], group_cols=[], chunks={"row": 2})[0]
    dims = xds.sizes
    chunks = xds.chunks

    np_data = _make_linear_ramp_array(
        (dims["row"], dims["chan"], dims["corr"]), dtype=np.complex64
    )
    row_chunks = chunks["row"]
    nchan, ncorr = dims["chan"], dims["corr"]

    @dask.delayed
    def step1(arr):
        return arr * 1.0  # numpy in

    @dask.delayed
    def step2(arr):
        return _ArrayLike(
            arr
        )  # non-numpy out — the final step returns an array API object

    parts, row = [], 0
    for rc in row_chunks:
        chunk = np_data[row : row + rc]
        result = step2(step1(chunk))
        parts.append(
            da.from_delayed(result, shape=(rc, nchan, ncorr), dtype=np_data.dtype)
        )
        row += rc
    da_data = da.concatenate(parts, axis=0)

    nds = xds.assign(DATA=(("row", "chan", "corr"), da_data))
    (write,) = xds_to_table(nds, ms, ["DATA"])
    dask.compute(write)

    result = xds_from_ms(ms, columns=["DATA"], group_cols=[], chunks={"row": 2})[0]
    assert_array_equal(result.DATA.data, np_data)


def test_postprocess_ms(ms):
    """Test that postprocessing of MS variables identifies chan/corr like data"""
    xdsl = xds_from_ms(ms)

    def _array(ds, dims):
        shape = tuple(ds.sizes[d] for d in dims)
        chunks = tuple(ds.chunks[d] for d in dims)
        return (dims, da.random.random(size=shape, chunks=chunks))

    # Write some non-standard columns back to the MS
    for i, ds in enumerate(xdsl):
        xdsl[i] = ds.assign(
            **{
                "BITFLAG": _array(ds, ("row", "chan", "corr")),
                "HAS_CORRS": _array(ds, ("row", "corr")),
                "HAS_CHANS": _array(ds, ("row", "chan")),
            }
        )

    dask.compute(xds_to_table(xdsl, ms))

    for ds in xds_from_ms(ms, chunks={"row": 1, "chan": 1, "corr": 1}):
        assert ds.BITFLAG.dims == ("row", "chan", "corr")

        assert ds.HAS_CORRS.dims == ("row", "corr")
        assert ds.HAS_CHANS.dims == ("row", "chan")

        assert dict(ds.chunks) == {
            "uvw": (3,),
            "row": (1,) * ds.sizes["row"],
            "chan": (1,) * ds.sizes["chan"],
            "corr": (1,) * ds.sizes["corr"],
        }


# ── GPU-mock write path tests ─────────────────────────────────────────────────
#
# Inject fake cupy / torch modules so that _issubclass_fast matches the mock
# classes without requiring real GPU hardware or installed GPU libraries.
# The GPU variants raise on __array__ until to_device_cpu is called.


def _make_gpu_cupy_array(data):
    """Return a GpuArray instance backed by a fresh fake-cupy module."""
    mod = types.ModuleType("cupy")

    class GpuArray:
        # Tells numpy/dask this is an array-like object (needed for dask's is_arraylike),
        # but opts out of ufunc dispatch (correct for GPU arrays that don't speak numpy).
        __array_ufunc__ = None

        def __init__(self, d):
            self._data = np.asarray(d)

        def get(self):
            return self._data

        def __array__(self, dtype=None, copy=None):
            raise RuntimeError("GPU CuPy array: call .get() before __array__")

        @property
        def shape(self):
            return self._data.shape

        @property
        def dtype(self):
            return self._data.dtype

        @property
        def ndim(self):
            return self._data.ndim

        def __getitem__(self, item):
            return GpuArray(self._data[item])

        def __len__(self):
            return len(self._data)

    mod.ndarray = GpuArray
    return mod, GpuArray(data)


def _make_gpu_torch_tensor(data):
    """Return a GpuTensor instance backed by a fresh fake-torch module."""
    mod = types.ModuleType("torch")

    class Tensor:
        __array_ufunc__ = None

        def __init__(self, d):
            self._data = np.asarray(d)

        @property
        def shape(self):
            return self._data.shape

        @property
        def dtype(self):
            return self._data.dtype

        def __getitem__(self, item):
            return CpuTensor(self._data[item])

        def __len__(self):
            return len(self._data)

    class CpuTensor(Tensor):
        def to(self, device):
            return self

        def __array__(self, dtype=None, copy=None):
            return self._data if dtype is None else self._data.astype(dtype)

    class GpuTensor(Tensor):
        def to(self, device):
            if device == "cpu":
                return CpuTensor(self._data)
            raise RuntimeError(f"Cannot move to {device!r}")

        def __array__(self, dtype=None, copy=None):
            raise RuntimeError("GPU Torch tensor: call .to('cpu') before __array__")

        def __getitem__(self, item):
            return GpuTensor(self._data[item])

    mod.Tensor = Tensor
    return mod, GpuTensor(data)


@pytest.fixture()
def fake_cupy_ms(ms):
    """Yield (ms_path, GpuArray-of-zeros) with fake cupy injected."""
    xds = xds_from_ms(ms, columns=["DATA"], group_cols=[], chunks={"row": 2})[0]
    dims = xds.sizes
    data = _make_linear_ramp_array(
        (dims["row"], dims["chan"], dims["corr"]), dtype=np.complex64
    )

    mod, gpu_arr = _make_gpu_cupy_array(data)
    original = sys.modules.get("cupy")
    sys.modules["cupy"] = mod
    _issubclass_fast.cache_clear()
    yield ms, gpu_arr, data
    if original is None:
        sys.modules.pop("cupy", None)
    else:
        sys.modules["cupy"] = original
    _issubclass_fast.cache_clear()


@pytest.fixture()
def fake_torch_ms(ms):
    """Yield (ms_path, GpuTensor-of-zeros) with fake torch injected."""
    xds = xds_from_ms(ms, columns=["DATA"], group_cols=[], chunks={"row": 2})[0]
    dims = xds.sizes
    data = _make_linear_ramp_array(
        (dims["row"], dims["chan"], dims["corr"]), dtype=np.complex64
    )
    mod, gpu_tensor = _make_gpu_torch_tensor(data)
    original = sys.modules.get("torch")
    sys.modules["torch"] = mod
    _issubclass_fast.cache_clear()
    yield ms, gpu_tensor, data
    if original is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = original
    _issubclass_fast.cache_clear()


def _from_delayed_chunks(arr, row_chunks, shape_rest, dtype):
    """Build a dask array from per-chunk delayed objects (avoids dtype introspection)."""
    parts, row = [], 0
    for rc in row_chunks:
        chunk = arr[row : row + rc]
        parts.append(
            da.from_delayed(dask.delayed(chunk), shape=(rc,) + shape_rest, dtype=dtype)
        )
        row += rc
    return da.concatenate(parts, axis=0)


def test_fake_cupy_gpu_write(fake_cupy_ms):
    """A fake CuPy GPU array must survive the full write path via to_device_cpu/.get()."""
    ms_path, gpu_arr, expected = fake_cupy_ms
    xds = xds_from_ms(ms_path, columns=["DATA"], group_cols=[], chunks={"row": 2})[0]
    dims, chunks = xds.sizes, xds.chunks

    da_data = da.from_array(gpu_arr, chunks=(chunks["row"], dims["chan"], dims["corr"]))
    nds = xds.assign(DATA=(("row", "chan", "corr"), da_data))
    (write,) = xds_to_table(nds, ms_path, ["DATA"])
    dask.compute(write)

    result = xds_from_ms(ms_path, columns=["DATA"], group_cols=[], chunks={"row": 2})[0]
    assert_array_equal(result.DATA.values, expected)


def test_fake_torch_gpu_write(fake_torch_ms):
    """A fake PyTorch GPU tensor must survive the full write path via to_device_cpu/.to('cpu').

    PyTorch tensors must be wrapped via from_delayed to avoid dtype introspection
    at graph-construction time (torch dtypes are not numpy-compatible).
    """
    ms_path, gpu_tensor, expected = fake_torch_ms
    xds = xds_from_ms(ms_path, columns=["DATA"], group_cols=[], chunks={"row": 2})[0]
    dims, chunks = xds.sizes, xds.chunks

    da_data = _from_delayed_chunks(
        gpu_tensor, chunks["row"], (dims["chan"], dims["corr"]), expected.dtype
    )
    nds = xds.assign(DATA=(("row", "chan", "corr"), da_data))
    (write,) = xds_to_table(nds, ms_path, ["DATA"])
    dask.compute(write)

    result = xds_from_ms(ms_path, columns=["DATA"], group_cols=[], chunks={"row": 2})[0]
    assert_array_equal(result.DATA.values, expected)
