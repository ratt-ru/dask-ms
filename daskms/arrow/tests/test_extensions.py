import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")


class ComplexArray(pa.ExtensionArray):
    def to_numpy_array(self):
        return self.storage.flatten().to_numpy()


class ComplexType(pa.ExtensionType):
    def __init__(self, subtype):
        if not isinstance(subtype, pa.DataType):
            subtype = pa.type_for_alias(str(subtype))

        self._subtype = subtype
        storage_type = pa.list_(subtype, 2)
        pa.ExtensionType.__init__(self, storage_type, "daskms.complex")

    @property
    def subtype(self):
        return self._subtype

    def __arrow_ext_serialize__(self):
        return f"dtype={self.subtype}".encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        bits = serialized.decode().split("=")
        assert len(bits) == 2 and bits[0] == "dtype"
        return ComplexType(pa.type_for_alias(bits[1]))

    def __arrow_ext_class__(self):
        return ComplexArray


_complex_type = ComplexType("float32")
pa.register_extension_type(_complex_type)


def numpy_to_arrow(array):
    if np.iscomplexobj(array):
        flat_array = array.view(array.real.dtype).ravel()
        storage = pa.FixedSizeListArray.from_arrays(flat_array, 2)
        complex_type = ComplexType(storage.type.value_type)
        pa_data = pa.ExtensionArray.from_storage(complex_type, storage)
    else:
        pa_data = pa.array(array.ravel())

    for size in reversed(array.shape[1:]):
        pa_data = pa.FixedSizeListArray.from_arrays(pa_data, size)

    return pa_data


def arrow_to_numpy(array):
    pass


@pytest.mark.parametrize("dtype", [
    np.float32,
    np.float64,
    np.complex64,
    np.complex128])
@pytest.mark.parametrize("shape", [(10, 16, 4)])
def test_extensions(shape, dtype):
    if np.issubdtype(dtype, np.complexfloating):
        data = np.random.random(shape) + np.random.random(shape)*1j
        data = data.astype(dtype)
    else:
        data = np.random.random(shape).astype(dtype)

    pa_data = numpy_to_arrow(data)

    assert isinstance(pa_data, pa.FixedSizeListArray)
    assert len(pa_data) == shape[0]
    assert isinstance(pa_data.type, pa.FixedSizeListType)
    assert pa_data.type.list_size == shape[1]
    assert isinstance(pa_data.type.value_type, pa.FixedSizeListType)
    assert pa_data.type.value_type.list_size == shape[2]

    if np.issubdtype(dtype, np.complexfloating):
        assert isinstance(pa_data.type.value_type.value_type, ComplexType)
        ct = ComplexType(pa.from_numpy_dtype(data.real.dtype))
        assert pa_data.type.value_type.value_type == ct
    else:
        assert pa_data.type.value_type.value_type == pa.from_numpy_dtype(dtype)