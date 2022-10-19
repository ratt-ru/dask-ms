import itertools
import json
import logging

import numpy as np

from daskms.constants import ARROW_METADATA, DASKMS_METADATA
from daskms.dataset_schema import DatasetSchema, ColumnSchema, unify_schemas
from daskms.utils import Frozen, merge_dicts

from daskms.experimental.arrow.extension_types import TensorType, ComplexType
from daskms.experimental.arrow.require_arrow import requires_arrow

try:
    import pyarrow as pa
except ImportError as e:
    pyarrow_import_error = e
else:
    pyarrow_import_error = None

log = logging.getLogger(__name__)


class ArrowUnificationError(ValueError):
    pass


class ArrowSchema(DatasetSchema):
    @classmethod
    def unify_column(cls, column, schema):
        it = zip(schema.dims, schema.shape, schema.attrs)
        new_dims = []
        new_shape = []
        new_attrs = []

        for i, (dims, shape, attrs) in enumerate(it):
            if len(dims) == 0:
                log.warning(
                    f"Ignoring column {column} with " f"zero-length dims {schema.dims}"
                )
                continue

            if dims[0] != "row":
                log.warning(
                    f"Ignoring column {column} without "
                    f"'row' as the starting dimension {schema.dims}"
                )
                continue

            new_dims.append(dims[1:])
            new_shape.append(shape[1:])
            new_attrs.append(attrs)

        typ = set(schema.type)
        dtype = set(schema.dtype)
        dims = set(new_dims)
        shape = set(new_shape)

        if len(typ) != 1:
            raise ArrowUnificationError(f"Inconsistent column types {typ}")

        if len(dtype) != 1:
            raise ArrowUnificationError(f"Inconsistent column dtypes {dtype}")

        if len(dims) != 1:
            raise ArrowUnificationError(f"Inconsistent column dims {dims}")

        if len(shape) != 1:
            raise ArrowUnificationError(f"Inconsistent column shapes {shape}")

        if not all(new_attrs[0] == a for a in new_attrs[1:]):
            raise ArrowUnificationError(f"Inconsistent column attributes {new_attrs}")

        chunks = None
        attrs = new_attrs[0]

        return ColumnSchema(
            typ.pop(), dims.pop(), dtype.pop(), chunks, shape.pop(), attrs
        )

    @classmethod
    def unify_attrs(cls, attrs):
        if len(attrs) == 0:
            return {}

        metadata = set(map(Frozen, (a.get(DASKMS_METADATA, {}) for a in attrs)))

        if len(metadata) != 1:
            raise ArrowUnificationError(f"Inconsistent dataset attributes {metadata}")

        return {DASKMS_METADATA: metadata.pop().value}

    @classmethod
    def from_datasets(cls, datasets):
        schemas = list(map(DatasetSchema.from_dataset, datasets))
        unified_schema = unify_schemas(schemas)

        data_vars = {
            c: cls.unify_column(c, s) for c, s in unified_schema.data_vars.items()
        }
        coords = {c: cls.unify_column(c, s) for c, s in unified_schema.coords.items()}
        attrs = cls.unify_attrs(unified_schema.attrs)

        return ArrowSchema(data_vars, coords, attrs)

    def with_attributes(self, dataset):
        data_vars = {}
        coords = {}

        for column, schema in self.data_vars.items():
            try:
                ds_column = getattr(dataset, column)
            except AttributeError:
                continue

            data_vars[column] = ColumnSchema(
                schema.type,
                schema.dims,
                schema.dtype,
                schema.chunks,
                schema.shape,
                merge_dicts(schema.attrs, ds_column.attrs),
            )

        for column, schema in self.coords.items():
            try:
                ds_column = getattr(dataset, column)
            except AttributeError:
                continue

            coords[column] = ColumnSchema(
                schema.type,
                schema.dims,
                schema.dtype,
                schema.chunks,
                schema.shape,
                merge_dicts(schema.attrs, ds_column.attrs),
            )

        attrs = merge_dicts(self.attrs, dataset.attrs)
        return ArrowSchema(data_vars, coords, attrs)

    @requires_arrow(pyarrow_import_error)
    def to_arrow_schema(self):
        fields = []

        variables = itertools.chain(self.data_vars.items(), self.coords.items())

        for column, var in variables:
            if var.dtype == np.dtype(np.complex64):
                pa_type = ComplexType(pa.float32())
            elif var.dtype == np.dtype(np.complex128):
                pa_type = ComplexType(pa.float64())
            elif var.dtype == np.dtype(object):
                # TODO(sjperkins)
                # objects contain strings.
                # Possibly replace this with a pa.binary()
                # containing pickled objects
                pa_type = pa.string()
            else:
                pa_type = pa.from_numpy_dtype(var.dtype)

            if var.ndim == 0:
                pass
            else:
                pa_type = TensorType(var.shape, pa_type)

            metadata = {
                **var.attrs,
                "coordinate": False,
                "dims": ("row",) + tuple(var.dims),
            }
            metadata = {ARROW_METADATA: json.dumps(metadata)}
            fields.append(pa.field(column, pa_type, metadata=metadata))

        metadata = {ARROW_METADATA: json.dumps(self.attrs)}
        return pa.schema(fields, metadata=metadata)
