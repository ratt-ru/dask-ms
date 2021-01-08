# Adapted from https://github.com/apache/arrow/pull/8510
# to handle Tensors of Complex Numbers. Could be removed
# if the following are addressed
# https://issues.apache.org/jira/browse/ARROW-638
# https://issues.apache.org/jira/browse/ARROW-1614


from ast import literal_eval

import numpy as np

try:
    import pyarrow as pa
except ImportError:
    pa = None
    ExtensionArray = object
    ExtensionType = object
else:
    ExtensionArray = pa.ExtensionArray
    ExtensionType = pa.ExtensionType


def _tensor_to_array(obj, pa_dtype):
    batch_size = obj.shape[0]
    element_shape = obj.shape[1:]
    total_elements = obj.size
    num_elements = 1 if len(obj.shape) == 1 else np.prod(element_shape)

    if isinstance(pa_dtype, ComplexType):
        flat_array = obj.view(obj.real.dtype).ravel()
        storage = pa.FixedSizeListArray.from_arrays(flat_array, 2)
        child_array = pa.ExtensionArray.from_storage(pa_dtype, storage)
    else:
        child_buf = pa.py_buffer(obj)
        child_array = pa.Array.from_buffers(pa_dtype, total_elements,
                                            [None, child_buf])

    offsets = np.int32([i * num_elements for i in range(batch_size + 1)])
    offset_buf = pa.py_buffer(offsets)

    storage = pa.Array.from_buffers(pa.list_(pa_dtype), batch_size,
                                    [None, offset_buf],
                                    children=[child_array])

    tensor_type = TensorType(element_shape, pa_dtype)
    return pa.ExtensionArray.from_storage(tensor_type, storage)


class TensorType(ExtensionType):
    def __init__(self, element_shape, pyarrow_dtype):
        if not isinstance(pyarrow_dtype, pa.DataType):
            pyarrow_dtype = pa.type_for_alias(str(pyarrow_dtype))

        self._element_shape = element_shape
        pa.ExtensionType.__init__(self, pa.list_(pyarrow_dtype),
                                  "daskms.tensor_type")

    def __reduce__(self):
        return TensorType, (self._element_shape, self.storage_type.value_type)

    @property
    def shape(self):
        return self._element_shape

    def __arrow_ext_serialize__(self):
        return f"shape={self._element_shape}".encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        parts = serialized.decode().split("=")
        assert parts[0] == "shape" and len(parts) == 2
        return TensorType(literal_eval(parts[1]), storage_type.value_type)

    def __arrow_ext_class__(self):
        return TensorArray


class TensorArray(ExtensionArray):
    @classmethod
    def from_numpy(cls, obj):
        assert isinstance(obj, np.ndarray)

        if not obj.flags.c_contiguous:
            obj = np.ascontiguousarray(obj)

        if np.iscomplexobj(obj):
            dtype = ComplexType(pa.from_numpy_dtype(obj.real.dtype))
        else:
            dtype = pa.from_numpy_dtype(obj.dtype)

        return _tensor_to_array(obj, dtype)

    @classmethod
    def from_tensor(cls, obj):
        assert isinstance(obj, pa.Tensor)
        assert obj.is_contiguous
        dtype = obj.type

        return _tensor_to_array(obj, dtype)

    def to_numpy(self):
        shape = (len(self),) + self.type.shape
        storage_list_type = self.storage.type
        dtype = storage_list_type.value_type.to_pandas_dtype()
        i = 4 if np.issubdtype(dtype, np.complexfloating) else 3
        buf = self.storage.buffers()[i]

        return np.ndarray(shape, buffer=buf, dtype=dtype)

    def to_tensor(self):
        return pa.Tensor.from_numpy(self.to_numpy())


class ComplexArray(ExtensionArray):
    def to_numpy_array(self):
        return self.storage.to_numpy()


class ComplexType(ExtensionType):
    def __init__(self, subtype):
        if not isinstance(subtype, pa.DataType):
            subtype = pa.type_for_alias(str(subtype))

        self._subtype = subtype
        pa.ExtensionType.__init__(self, pa.list_(subtype, 2), "daskms.complex")

    def to_pandas_dtype(self):
        return np.result_type(self._subtype.to_pandas_dtype(), np.complex64)

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return ComplexType(storage_type.value_type)

    def __arrow_ext_class__(self):
        return ComplexArray


if pa:
    _complex_type = ComplexType("float32")
    _tensor_type = TensorType((1,), "float32")
    pa.register_extension_type(_complex_type)
    pa.register_extension_type(_tensor_type)
