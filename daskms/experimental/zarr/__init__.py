from pathlib import Path
from threading import Lock
from weakref import WeakValueDictionary

import dask.array as da
import numpy as np

from daskms.utils import arg_hasher, requires
from daskms.experimental.utils import (encode_attr,
                                       extent_args,
                                       DATASET_TYPE,
                                       DATASET_TYPES)
from daskms.optimisation import inlined_array

DATASET_PREFIX = "__daskms_dataset__"
DASKMS_ATTR_KEY = "__daskms_zarr_attr__"

try:
    import zarr
except ImportError as e:
    zarr_import_error = e
else:
    zarr_import_error = None

_store_cache = WeakValueDictionary()
_store_lock = Lock()


class ZarrDatasetFactoryMetaClass(type):
    """
    https://en.wikipedia.org/wiki/Multiton_pattern

    """
    def __call__(cls, *args, **kwargs):
        key = arg_hasher((args, kwargs))

        try:
            return _store_cache[key]
        except KeyError:
            with _store_lock:
                try:
                    return _store_cache[key]
                except KeyError:
                    instance = type.__call__(cls, *args, **kwargs)
                    _store_cache[key] = instance
                    return instance


class ZarrDatasetFactory(metaclass=ZarrDatasetFactoryMetaClass):
    def __init__(self, store, dataset_id, schema, attrs):
        assert isinstance(store, str)
        assert isinstance(dataset_id, int)
        assert isinstance(schema, dict)
        assert isinstance(attrs, dict)

        self.store = store
        self.dataset_id = dataset_id
        self.schema = schema
        self.attrs = attrs

        self.lock = Lock()

    @property
    def group(self):
        with self.lock:
            try:
                return self._group
            except AttributeError:
                dir_store = zarr.DirectoryStore(self.store)

                try:
                    # Open in read/write, must exist
                    group = zarr.open_group(store=dir_store, mode="r+")
                except zarr.errors.GroupNotFoundError:
                    try:
                        # Create, must not exist
                        group = zarr.open_group(store=dir_store, mode="w-")
                    except zarr.errors.ContainsGroupError:
                        # Open in read/write, must exist
                        group = zarr.open_group(store=dir_store, mode="r+")

                group_name = f"{DATASET_PREFIX}{self.dataset_id:08d}"
                ds_group = group.require_group(group_name)
                ds_group.attrs.update(encode_attr(self.attrs))

                for name, (dims, shape, chunks, dtype) in self.schema.items():
                    array = ds_group.require_dataset(name, shape,
                                                     chunks=chunks,
                                                     dtype=dtype,
                                                     exact=True)
                    array.attrs[DASKMS_ATTR_KEY] = {"dims": dims}

                self._group = ds_group
                return ds_group

    def __reduce__(self):
        return (ZarrDatasetFactory,
                (self.store, self.dataset_id, self.schema, self.attrs))


def zarr_schema_factory(di, data_vars):
    schema = {}

    for name, var in data_vars.items():
        # Determine schema for backing zarr arrays
        zarr_chunks = []

        for d, dim_chunks in enumerate(var.data.chunks):
            if any(dc == np.nan for dc in dim_chunks):
                raise NotImplementedError("nan chunks not yet supported.")

            unique_chunks = set(dim_chunks[:-1])

            if len(unique_chunks) == 0:
                zarr_chunks.append(dim_chunks[-1])
            elif len(unique_chunks) == 1:
                zarr_chunks.append(unique_chunks.pop())
            else:
                raise ValueError(f"Multiple chunk sizes {unique_chunks} "
                                 f"found in dataset {di} "
                                 f"array {name} dimension {var.dims[d]}. "
                                 f"zarr requires homogenous chunk sizes "
                                 f"except for the last chunk in a "
                                 f"dimension. Rechunk your {name}.")

        schema[name] = (var.dims, var.shape, tuple(zarr_chunks), var.dtype)

    return schema


def _setter_wrapper(data, name, factory, *extents):
    zarray = getattr(factory.group, name)
    selection = tuple(slice(start, end) for start, end in extents)
    zarray[selection] = data
    return np.full((1,)*len(extents), True)


@requires("pip install dask-ms[zarr] for zarr support",
          zarr_import_error)
def xds_to_zarr(xds, store):
    if isinstance(store, Path):
        store = str(store)

    if not isinstance(store, str):
        raise TypeError(f"store '{store}' must be Path or str")

    if isinstance(xds, DATASET_TYPES):
        xds = [xds]
    elif isinstance(xds, (tuple, list)):
        if not all(isinstance(ds, DATASET_TYPES) for ds in xds):
            raise TypeError("xds must be a Dataset or list of Datasets")
    else:
        raise TypeError("xds must be a Dataset or list of Datasets")

    write_datasets = []

    for di, ds in enumerate(xds):
        schema = zarr_schema_factory(di, ds.data_vars)
        attrs = dict(ds.attrs)
        attrs[DASKMS_ATTR_KEY] = {"chunks": dict(ds.chunks)}
        factory = ZarrDatasetFactory(store, di, schema, attrs)
        data_vars = {}

        for name, var in ds.data_vars.items():
            ext_args = extent_args(var.dims, var.chunks)

            write = da.blockwise(_setter_wrapper, var.dims,
                                 var.data, var.dims,
                                 name, None,
                                 factory, None,
                                 *ext_args,
                                 adjust_chunks={d: 1 for d in var.dims},
                                 meta=np.empty((1,)*len(var.dims), np.bool))

            write = inlined_array(write, ext_args[::2])
            data_vars[name] = (var.dims, write, var.attrs)

        write_datasets.append(DATASET_TYPE(data_vars))

    return write_datasets


def _getter_wrapper(zarray, *extents):
    return zarray[tuple(slice(start, end) for start, end in extents)]


@requires("pip install dask-ms[zarr] for zarr support",
          zarr_import_error)
def xds_from_zarr(store, chunks=None):
    if isinstance(store, Path):
        store = str(store)

    if not isinstance(store, str):
        raise TypeError("store must be a Path, str")

    if chunks is None:
        pass
    elif isinstance(chunks, (tuple, list)):
        if not all(isinstance(v, dict) for v in chunks):
            raise TypeError("chunks must be None, a dict or a list of dicts")
    elif isinstance(chunks, dict):
        chunks = [chunks]
    else:
        raise TypeError("chunks must be None, a dict or a list of dicts")

    root = zarr.open(store)
    datasets = []

    for g, (group_name, group) in enumerate(sorted(root.groups())):
        assert group_name.startswith(DATASET_PREFIX)
        group_attrs = dict(group.attrs)
        natural_chunks = group_attrs.pop(DASKMS_ATTR_KEY)["chunks"]
        group_chunks = {d: tuple(dc) for d, dc in natural_chunks.items()}

        if chunks:
            # Defer to user-supplied chunking strategy
            try:
                group_chunks.update(chunks[g])
            except IndexError:
                pass

        data_vars = {}

        for name, zarray in group.items():
            attrs = dict(zarray.attrs[DASKMS_ATTR_KEY])
            dims = attrs.pop("dims")
            array_chunks = tuple(group_chunks.get(d, s) for d, s
                                 in zip(dims, zarray.shape))

            array_chunks = da.core.normalize_chunks(array_chunks, zarray.shape)
            ext_args = extent_args(dims, array_chunks)

            read = da.blockwise(_getter_wrapper, dims,
                                zarray, None,
                                *ext_args,
                                meta=np.empty((0,)*zarray.ndim, zarray.dtype))

            read = inlined_array(read, ext_args[::2])
            data_vars[name] = (dims, read, attrs)

        datasets.append(DATASET_TYPE(data_vars, attrs=group_attrs))

    return datasets
