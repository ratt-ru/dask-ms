import importlib

import numpy as np
import dask.array as da

from daskms.utils import encode_attr

_ALLOWED_PACKAGES = {"numpy", "dask"}


def encode_type(typ: type):
    return ".".join((typ.__module__, typ.__name__))


def decode_type(typ: str):
    pkg, _ = typ.split(".", 1)

    if pkg not in _ALLOWED_PACKAGES:
        raise ImportError(f"Import of '{typ}' denied")

    mod_str, typ_str = typ.rsplit(".", 1)
    mod = importlib.import_module(mod_str)

    try:
        return getattr(mod, typ_str)
    except AttributeError:
        raise ValueError(f"{typ_str} is not an "
                         f"attribute of {mod_str}")


class DatasetSchema:
    def __init__(self, data_vars, coords, attrs):
        self.data_vars = data_vars
        self.coords = coords
        self.attrs = attrs

    def __eq__(self, other):
        return (isinstance(other, DatasetSchema) and
                self.data_vars == other.data_vars and
                self.coords == other.coords and
                self.attrs == other.attrs)

    @classmethod
    def from_dataset(cls, dataset):
        cs = ColumnSchema.from_var
        data_vars = {c: cs(v) for c, v in dataset.data_vars.items()}
        coords = {c: cs(v) for c, v in dataset.coords.items()}
        return DatasetSchema(data_vars, coords, dict(dataset.attrs))

    @classmethod
    def from_dict(cls, d):
        dv = d["data_vars"]
        co = d["coords"]

        data_vars = {c: ColumnSchema.from_dict(v) for c, v in dv.items()}
        coords = {c: ColumnSchema.from_dict(v) for c, v in co.items()}
        return DatasetSchema(data_vars, coords, d["attributes"])

    def to_dict(self):
        data_vars = {c: v.to_dict() for c, v in self.data_vars.items()}
        coords = {c: v.to_dict() for c, v in self.coords.items()}

        return {
            "data_vars": data_vars,
            "coords": coords,
            "attributes": encode_attr(self.attrs)
        }


class ColumnSchema:
    def __init__(self, typ, dims, dtype, chunks, shape):
        self.type = typ
        self.dims = dims
        self.dtype = dtype
        self.chunks = chunks
        self.shape = shape

    def __eq__(self, other):
        return (isinstance(other, ColumnSchema) and
                self.type == other.type and
                self.dims == other.dims and
                self.dtype == other.dtype and
                self.chunks == other.chunks and
                self.shape == other.shape)

    @classmethod
    def from_var(cls, var):
        return ColumnSchema(
            type(var.data),
            var.dims,
            var.dtype,
            var.chunks if isinstance(var.data, da.Array) else None,
            var.shape)

    @classmethod
    def from_dict(self, d):
        chunks = d["chunks"]

        return ColumnSchema(
            decode_type(d["type"]),
            tuple(d["dims"]),
            np.dtype(d["dtype"]),
            tuple(map(tuple, chunks)) if chunks is not None else None,
            tuple(d["shape"]))

    def to_dict(self):
        return {
            "type": encode_type(self.type),
            "dims": self.dims,
            "dtype": self.dtype.name,
            "chunks": self.chunks,
            "shape": self.shape
        }
