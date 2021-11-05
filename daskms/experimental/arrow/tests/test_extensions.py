import numpy as np
from numpy.testing import assert_array_equal
import pytest

from daskms.experimental.arrow.extension_types import (
    ComplexType, ComplexArray, TensorType, TensorArray)

pa = pytest.importorskip("pyarrow")


@pytest.fixture(scope="function", params=[10])
def shape(request):
    return request.param


@pytest.fixture(scope="function", params=[np.float32])
def dtype(request):
    return request.param


@pytest.fixture(scope="function")
def test_data(shape, dtype):
    if np.issubdtype(dtype, np.complexfloating):
        data = np.random.random(shape) + np.random.random(shape)
        return data.astype(dtype)
    else:
        return np.random.random(shape).astype(dtype)


singleton_xfail = pytest.mark.xfail(reason="Singletons not handled yet")


@pytest.mark.parametrize("dtype", [
    bool,
    np.int32,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128])
@pytest.mark.parametrize("shape", [
    pytest.param((), marks=singleton_xfail),
    (10,),
    (10, 16, 4)
])
def test_arrow_numpy_conversion(test_data):
    pa_data = TensorArray.from_numpy(test_data)
    assert isinstance(pa_data,  TensorArray)
    assert isinstance(pa_data.type, TensorType)
    pa_type = pa_data.type.storage_type.value_type

    if np.issubdtype(test_data.dtype, np.complexfloating):
        assert isinstance(pa_type, ComplexType)
        expected_real_dt = pa.from_numpy_dtype(test_data.real.dtype)
        assert pa_type.storage_type.value_type == expected_real_dt
    else:
        assert pa_type == pa.from_numpy_dtype(test_data.dtype)

    assert_array_equal(test_data, pa_data.to_numpy())


@pytest.mark.parametrize("dtype", [
    np.complex64,
    np.complex128])
@pytest.mark.parametrize("shape", [
    pytest.param((), marks=singleton_xfail),
    (10,),
    (20,)
])
def test_complex_type_conversion(test_data):
    array = ComplexArray.from_numpy(test_data)
    assert_array_equal(array.to_numpy(), test_data)
