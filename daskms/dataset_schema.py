from collections import defaultdict
from itertools import chain
import logging
import importlib

import numpy as np
import dask.array as da

from daskms.utils import freeze

log = logging.getLogger(__name__)

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


def encode_attr(arg):
    """ Convert arg into something acceptable to json """
    if isinstance(arg, tuple):
        return tuple(map(encode_attr, arg))
    elif isinstance(arg, list):
        return list(map(encode_attr, arg))
    elif isinstance(arg, set):
        return list(map(encode_attr, sorted(arg)))
    elif isinstance(arg, dict):
        return {k: encode_attr(v) for k, v in sorted(arg.items())}
    elif isinstance(arg, np.ndarray):
        return arg.tolist()
    elif isinstance(arg, np.generic):
        return arg.item()
    else:
        return arg


def decode_attr(arg):
    if isinstance(arg, (tuple, list)):
        return tuple(map(decode_attr, arg))
    elif isinstance(arg, set):
        return set(map(decode_attr, arg))
    elif isinstance(arg, dict):
        return {k: decode_attr(v) for k, v in sorted(arg.items())}
    else:
        return arg


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

    def __reduce__(self):
        return (DatasetSchema, (self.data_vars, self.coords, self.attrs))

    def __hash__(self):
        return hash(
            freeze(
                (
                    self.data_vars,
                    self.coords,
                    self.attrs
                )
            )
        )

    def drop_dim(self, dim):
        return DatasetSchema(
            {c: v.drop_dim(dim) for c, v in self.data_vars.items()},
            {c: v.drop_dim(dim) for c, v in self.coords.items()},
            self.attrs.copy(),
        )

    @classmethod
    def from_dataset(cls, dataset, columns=None):
        dv = dataset.data_vars
        co = dataset.coords

        if columns is None or columns == "ALL":
            columns = set(dv.keys()) | set(co.keys())
        elif isinstance(columns, str):
            columns = set([columns])
        else:
            columns = set(columns)

        data_vars = {
            c: ColumnSchema.from_var(v) for c, v
            in dv.items() if c in columns}
        coords = {
            c: ColumnSchema.from_var(v) for c, v
            in co.items() if c in columns}

        return DatasetSchema(data_vars, coords, dict(dataset.attrs))

    @classmethod
    def from_dict(cls, d):
        dv = d["data_vars"]
        co = d["coords"]

        data_vars = {c: ColumnSchema.from_dict(v) for c, v in dv.items()}
        coords = {c: ColumnSchema.from_dict(v) for c, v in co.items()}
        return DatasetSchema(data_vars, coords, d["attributes"].copy())

    @property
    def dims(self):
        dims = defaultdict(set)

        for v in chain(self.data_vars.values(), self.coords.values()):
            for d, s in zip(v.dims, v.shape):
                dims[d].add(s)

        ret = {}

        for d, sizes in dims.items():
            if len(sizes) > 1:
                raise ValueError(f"Inconsistent sizes {sizes} "
                                 f"for dimension {d}")

            ret[d] = next(iter(sizes))

        return ret

    @property
    def chunks(self):
        chunks = defaultdict(set)

        for v in chain(self.data_vars.values(), self.coords.values()):
            if v.chunks is None:
                continue

            for d, c in zip(v.dims, v.chunks):
                chunks[d].add(c)

        ret = {}

        for d, dim_chunks in chunks.items():
            if len(dim_chunks) > 1:
                raise ValueError(f"Inconsistent chunks {dim_chunks}"
                                 f"for dimension {d}")

            ret[d] = next(iter(dim_chunks))

        return ret

    def to_dict(self):
        data_vars = {c: v.to_dict() for c, v in self.data_vars.items()}
        coords = {c: v.to_dict() for c, v in self.coords.items()}

        return {
            "data_vars": data_vars,
            "coords": coords,
            "attributes": encode_attr(self.attrs)
        }


class ColumnSchema:
    __slots__ = ("type", "dims", "dtype", "chunks", "shape", "attrs")

    """
    Schema describing a column

    Parameters
    ----------
    typ : type
        Type of the array. e.g. `dask.array.Array` or `numpy.ndarray`
    dims : list of str
        Dimension schema of array. e.g. :code:`(:row:, :chan:, :corr:)`
    dtype : :class:`numpy.dtype`
        Array Datatype
    chunks : tuple of tuple of ints or None
        Dask dimension chunks, else None for non-dask arrays
    shape : tuple of ints
        Array shape
    attrs : dict
        Dictionary of attributes
    """

    def __init__(self, typ, dims, dtype, chunks, shape, attrs):
        self._type_check("type", typ, type)
        self._type_check("dtype", dtype, np.dtype)
        self._type_check("chunks", chunks, (tuple, type(None)))
        self._type_check("shape", shape, tuple)
        self._type_check("attrs", attrs, dict)

        self.type = typ
        self.dims = dims
        self.dtype = dtype
        self.chunks = chunks
        self.shape = shape
        self.attrs = attrs

    @classmethod
    def _type_check(cls, name, obj, typ):
        if isinstance(obj, typ):
            return

        if (isinstance(obj, (set, list, tuple)) and
                all(isinstance(o, typ) for o in obj)):
            return

        raise TypeError(f"{name} '{obj}' is not {typ.__name__} "
                        f"or a (tuple, list, set) of {typ.__name__}")

    def __eq__(self, other):
        return (isinstance(other, ColumnSchema) and
                self.type == other.type and
                self.dims == other.dims and
                self.dtype == other.dtype and
                self.chunks == other.chunks and
                self.shape == other.shape and
                self.attrs == other.attrs)

    def __reduce__(self):
        return (
            ColumnSchema,
            (
                self.type,
                self.dims,
                self.dtype,
                self.chunks,
                self.shape,
                self.attrs,
            )
        )

    def __hash__(self):
        return hash(
            (
                self.type,
                self.dims,
                self.dtype,
                self.chunks,
                self.shape,
                freeze(self.attrs),
            )
        )

    def copy(self):
        return ColumnSchema(
            self.type,
            self.dims,
            self.dtype,
            self.chunks,
            self.shape,
            self.attrs.copy()
        )

    @property
    def ndim(self):
        return len(self.dims)

    @classmethod
    def from_var(cls, var):
        return ColumnSchema(
            type(var.data),
            var.dims,
            var.dtype,
            var.chunks if isinstance(var.data, da.Array) else None,
            var.shape,
            var.attrs)

    @classmethod
    def from_dict(self, d):
        chunks = d["chunks"]

        return ColumnSchema(
            decode_type(d["type"]),
            tuple(d["dims"]),
            np.dtype(d["dtype"]),
            tuple(map(tuple, chunks)) if chunks is not None else None,
            tuple(d["shape"]),
            d["attrs"].copy())

    def to_dict(self):
        return {
            "type": encode_type(self.type),
            "dims": self.dims,
            "dtype": self.dtype.name,
            "chunks": self.chunks,
            "shape": self.shape,
            "attrs": self.attrs.copy(),
        }


def _unify_columns(columns, defs):
    for c, var in columns.items():
        defs[c]["dims"].append(var.dims)
        defs[c]["shape"].append(var.shape)
        defs[c]["chunks"].append(var.chunks)
        defs[c]["dtype"].append(var.dtype)
        defs[c]["typ"].append(var.type)
        defs[c]["attrs"].append(var.attrs)


def unify_schemas(dataset_schemas):
    if not isinstance(dataset_schemas, (tuple, list)):
        dataset_schemas = [dataset_schemas]

    if not all(isinstance(ds, DatasetSchema) for ds in dataset_schemas):
        raise TypeError("dataset_schemas must be a "
                        "DatasetSchema or list of DatasetSchema's")

    unified_data_vars = defaultdict(lambda: defaultdict(list))
    unified_coords = defaultdict(lambda: defaultdict(list))

    for ds in dataset_schemas:
        _unify_columns(ds.data_vars, unified_data_vars)
        _unify_columns(ds.coords, unified_coords)

    for column, schema_attrs in unified_data_vars.items():
        unified_data_vars[column] = ColumnSchema(**schema_attrs)

    for column, schema_attrs in unified_coords.items():
        unified_coords[column] = ColumnSchema(**schema_attrs)

    unified_attrs = [ds.attrs for ds in dataset_schemas]

    return DatasetSchema(unified_data_vars, unified_coords, unified_attrs)
