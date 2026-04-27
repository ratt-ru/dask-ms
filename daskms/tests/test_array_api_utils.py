"""
Unit tests for daskms.array_api_utils.

Uses fake module injection into sys.modules so that _issubclass_fast can
match mock classes without requiring real GPU libraries to be installed.
Each fixture restores sys.modules and clears the lru_cache afterwards.
"""
import sys
import types

import dask.array as da
import numpy as np
import pytest

from daskms.array_api_utils import _issubclass_fast, is_array_api_obj, to_device_cpu


# ── Fake module helpers ────────────────────────────────────────────────────────


def _make_fake_cupy():
    """Return (module, CpuArray, GpuArray) for a fake cupy."""
    mod = types.ModuleType("cupy")

    class GpuArray:
        """Simulates a CuPy ndarray on GPU: __array__ raises, .get() works."""

        __array_ufunc__ = None

        def __init__(self, data):
            self._data = np.asarray(data)

        def get(self):
            return self._data

        def __array__(self, dtype=None, copy=None):
            raise RuntimeError(
                "Cannot call __array__ on a GPU array — use .get() first"
            )

        @property
        def shape(self):
            return self._data.shape

        @property
        def dtype(self):
            return self._data.dtype

    mod.ndarray = GpuArray
    return mod, GpuArray


def _make_fake_torch():
    """Return (module, CpuTensor, GpuTensor) for a fake torch.

    Both CpuTensor and GpuTensor are subclasses of mod.Tensor (the base class),
    so _issubclass_fast(cls, "torch", "Tensor") matches both.  The GPU variant
    raises on __array__; .to("cpu") downcasts to CpuTensor.
    """
    mod = types.ModuleType("torch")

    class Tensor:
        """Base class registered as mod.Tensor — mirrors torch.Tensor hierarchy."""

        __array_ufunc__ = None

        def __init__(self, data):
            self._data = np.asarray(data)

        @property
        def shape(self):
            return self._data.shape

        @property
        def dtype(self):
            return self._data.dtype

    class CpuTensor(Tensor):
        """Simulates a CPU torch.Tensor: __array__ works, .to('cpu') is identity."""

        def to(self, device):
            if device == "cpu":
                return self
            raise RuntimeError(f"Fake tensor cannot move to {device!r}")

        def __array__(self, dtype=None, copy=None):
            return self._data if dtype is None else self._data.astype(dtype)

    class GpuTensor(Tensor):
        """Simulates a CUDA torch.Tensor: __array__ raises, .to('cpu') works."""

        def to(self, device):
            if device == "cpu":
                return CpuTensor(self._data)
            raise RuntimeError(f"Fake tensor cannot move to {device!r}")

        def __array__(self, dtype=None, copy=None):
            raise RuntimeError(
                "Cannot call __array__ on a CUDA tensor — use .to('cpu') first"
            )

    mod.Tensor = Tensor  # issubclass check uses mod.Tensor; both subclasses match
    return mod, CpuTensor, GpuTensor


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def fake_cupy():
    """Inject a fake cupy module; restore sys.modules and clear cache after."""
    mod, GpuArray = _make_fake_cupy()
    original = sys.modules.get("cupy")
    sys.modules["cupy"] = mod
    _issubclass_fast.cache_clear()
    yield mod, GpuArray
    if original is None:
        sys.modules.pop("cupy", None)
    else:
        sys.modules["cupy"] = original
    _issubclass_fast.cache_clear()


@pytest.fixture()
def fake_torch():
    """Inject a fake torch module; restore sys.modules and clear cache after."""
    mod, CpuTensor, GpuTensor = _make_fake_torch()
    original = sys.modules.get("torch")
    sys.modules["torch"] = mod
    _issubclass_fast.cache_clear()
    yield mod, CpuTensor, GpuTensor
    if original is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = original
    _issubclass_fast.cache_clear()


# ── is_array_api_obj ──────────────────────────────────────────────────────────


class TestIsArrayApiObj:
    def test_numpy_array(self):
        assert is_array_api_obj(np.zeros(3))

    def test_dask_array(self):
        assert is_array_api_obj(da.zeros(3))

    def test_plain_list_rejected(self):
        assert not is_array_api_obj([1, 2, 3])

    def test_scalar_rejected(self):
        assert not is_array_api_obj(42.0)

    def test_array_namespace_object(self):
        class FakeArray:
            def __array_namespace__(self, api_version=None):
                return np

        assert is_array_api_obj(FakeArray())

    def test_fake_cupy_array(self, fake_cupy):
        _, GpuArray = fake_cupy
        arr = GpuArray(np.zeros(3))
        assert is_array_api_obj(arr)

    def test_fake_torch_cpu_tensor(self, fake_torch):
        _, CpuTensor, _ = fake_torch
        arr = CpuTensor(np.zeros(3))
        assert is_array_api_obj(arr)

    def test_fake_torch_gpu_tensor(self, fake_torch):
        _, _, GpuTensor = fake_torch
        arr = GpuTensor(np.zeros(3))
        # GpuTensor is a subclass of mod.Tensor — detected as an array API object
        assert is_array_api_obj(arr)

    def test_fake_cupy_array_module_not_imported(self):
        """Without cupy in sys.modules, a cupy-like class is not detected."""
        _issubclass_fast.cache_clear()
        original = sys.modules.pop("cupy", None)
        try:
            _, GpuArray = _make_fake_cupy()
            arr = GpuArray(np.zeros(3))
            # Neither __array_namespace__ nor sys.modules["cupy"] present
            assert not is_array_api_obj(arr)
        finally:
            if original is not None:
                sys.modules["cupy"] = original
            _issubclass_fast.cache_clear()


# ── to_device_cpu ─────────────────────────────────────────────────────────────


class TestToDeviceCpu:
    def test_numpy_passthrough(self):
        arr = np.zeros((3, 4))
        result = to_device_cpu(arr)
        assert result is arr

    def test_unknown_object_passthrough(self):
        obj = object()
        assert to_device_cpu(obj) is obj

    def test_fake_cupy_calls_get(self, fake_cupy):
        _, GpuArray = fake_cupy
        data = np.array([1.0, 2.0, 3.0])
        gpu_arr = GpuArray(data)

        # Direct __array__ call must fail (simulating GPU restriction)
        with pytest.raises(RuntimeError, match="GPU"):
            np.asarray(gpu_arr)

        # to_device_cpu must call .get() and return a numpy array
        result = to_device_cpu(gpu_arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, data)

    def test_fake_cupy_result_is_numpy_castable(self, fake_cupy):
        _, GpuArray = fake_cupy
        gpu_arr = GpuArray(np.ones((2, 3), dtype=np.float32))
        cpu = to_device_cpu(gpu_arr)
        result = np.asarray(cpu)
        assert result.shape == (2, 3)
        assert result.dtype == np.float32

    def test_fake_torch_gpu_calls_to_cpu(self, fake_torch):
        _, CpuTensor, GpuTensor = fake_torch
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        gpu_tensor = GpuTensor(data)

        # Direct __array__ call must fail
        with pytest.raises(RuntimeError, match="CUDA"):
            np.asarray(gpu_tensor)

        # to_device_cpu must call .to("cpu") — result is CpuTensor, then castable
        cpu_tensor = to_device_cpu(gpu_tensor)
        assert isinstance(cpu_tensor, CpuTensor)
        result = np.asarray(cpu_tensor)
        np.testing.assert_array_equal(result, data)

    def test_fake_torch_cpu_tensor_passthrough(self, fake_torch):
        _, CpuTensor, _ = fake_torch
        data = np.array([1.0, 2.0], dtype=np.float32)
        cpu_tensor = CpuTensor(data)
        # CpuTensor is an instance of mod.Tensor, so torch path matches it
        result = to_device_cpu(cpu_tensor)
        # .to("cpu") returns self — identity preserved
        assert result is cpu_tensor

    def test_cupy_not_imported_passthrough(self):
        """If cupy is absent from sys.modules, to_device_cpu must not call .get()."""
        _issubclass_fast.cache_clear()
        original = sys.modules.pop("cupy", None)
        try:
            mod, GpuArray = _make_fake_cupy()
            arr = GpuArray(np.zeros(3))
            result = to_device_cpu(arr)
            # Falls through to the no-op return — returns the original object
            assert result is arr
        finally:
            if original is not None:
                sys.modules["cupy"] = original
            _issubclass_fast.cache_clear()
