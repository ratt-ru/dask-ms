import importlib
import itertools
from pathlib import Path
from threading import Lock
from weakref import WeakValueDictionary

import dask
import dask.array as da
import numpy as np
import numcodecs

from daskms.utils import arg_hasher, requires
from daskms.dataset import Dataset, Variable
from daskms.experimental.utils import (encode_attr,
                                       extent_args,
                                       column_iterator,
                                       promote_columns)
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

                for name, schema in self.schema.items():
                    if schema["dtype"] == np.object:
                        codec = numcodecs.Pickle()
                    else:
                        codec = None

                    array = ds_group.require_dataset(name, schema["shape"],
                                                     chunks=schema["chunks"],
                                                     dtype=schema["dtype"],
                                                     object_codec=codec,
                                                     exact=True)
                    array.attrs[DASKMS_ATTR_KEY] = {
                        "dims": schema["dims"],
                        "coordinate": schema["coordinate"],
                        "array_type": schema["type"]
                    }

                self._group = ds_group
                return ds_group

    def __reduce__(self):
        return (ZarrDatasetFactory,
                (self.store, self.dataset_id, self.schema, self.attrs))


def zarr_schema_factory(di, dataset):
    schema = {}

    data_vars = ((k, v, False) for k, v in dataset.data_vars.items())
    coords = ((k, v, True) for k, v in dataset.coords.items())
    variables = itertools.chain(data_vars, coords)
    chunks = dataset.chunks

    for name, var, is_coord in variables:
        # Determine schema for backing zarr arrays
        zarr_chunks = []

        for d in var.dims:
            dim_chunks = chunks[d]

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
                                 f"dimension. Rechunk {name}.")

        if isinstance(var.data, (da.Array, np.ndarray)):
            typ = type(var.data)
            array_type = ".".join((typ.__module__, typ.__name__))
        else:
            raise NotImplementedError(f"{type(var.data)} not supported")

        schema[name] = {
            "dims": var.dims,
            "shape": var.shape,
            "chunks": tuple(zarr_chunks),
            "dtype": var.dtype,
            "coordinate": is_coord,
            "type": array_type,
        }

    return schema


def _setter_wrapper(data, name, factory, *extents):
    zarray = getattr(factory.group, name)
    selection = tuple(slice(start, end) for start, end in extents)
    zarray[selection] = data
    return np.full((1,)*len(extents), True)


@requires("pip install dask-ms[zarr] for zarr support",
          zarr_import_error)
def xds_to_zarr(xds, store, columns=None):
    """
    Stores a dataset of list of datasets defined by `xds` in
    file location `store`.

    Parameters
    ----------
    xds : Dataset or list of Datasets
        Data
    store : str or Path
        Path to store the data
    columns : list of str or str or None
        Columns to store. `None` or `"ALL"` stores all columns on each dataset.
        Otherwise, a list of columns should be supplied.

    Returns
    -------
    writes : Dataset
        A Dataset representing the write operations
    """
    if isinstance(store, Path):
        store = str(store)

    if not isinstance(store, str):
        raise TypeError(f"store '{store}' must be Path or str")

    columns = promote_columns(columns)

    if isinstance(xds, Dataset):
        xds = [xds]
    elif isinstance(xds, (tuple, list)):
        if not all(isinstance(ds, Dataset) for ds in xds):
            raise TypeError("xds must be a Dataset or list of Datasets")
    else:
        raise TypeError("xds must be a Dataset or list of Datasets")

    write_datasets = []

    def _gen_writes(variables, chunks, columns):
        for name, var in column_iterator(variables, columns):
            if isinstance(var.data, da.Array):
                ext_args = extent_args(var.dims, var.chunks)
                var_data = var.data
            elif isinstance(var.data, np.ndarray):
                var_chunks = tuple(chunks[d] for d in var.dims)
                ext_args = extent_args(var.dims, var_chunks)
                var_data = da.from_array(var.data, chunks=var_chunks,
                                         inline_array=True, name=False,)
            else:
                raise NotImplementedError(f"Writing {type(var.data)} "
                                          f"unsupported")

            write = da.blockwise(_setter_wrapper, var.dims,
                                 var_data, var.dims,
                                 name, None,
                                 factory, None,
                                 *ext_args,
                                 adjust_chunks={d: 1 for d in var.dims},
                                 concatenate=False,
                                 meta=np.empty((1,)*len(var.dims), np.bool))
            write = inlined_array(write, ext_args[::2])

            yield name, (var.dims, write, var.attrs)

    for di, ds in enumerate(xds):
        schema = zarr_schema_factory(di, ds)
        attrs = dict(ds.attrs)
        attrs[DASKMS_ATTR_KEY] = {"chunks": dict(ds.chunks)}
        factory = ZarrDatasetFactory(store, di, schema, attrs)
        chunks = ds.chunks

        data_vars = dict(_gen_writes(ds.data_vars, chunks, columns))
        # Include coords in the write dataset so they're reified
        data_vars.update(dict(_gen_writes(ds.coords, chunks, columns)))

        write_datasets.append(Dataset(data_vars))

    return write_datasets


def _getter_wrapper(zarray, *extents):
    return zarray[tuple(slice(start, end) for start, end in extents)]


@requires("pip install dask-ms[zarr] for zarr support",
          zarr_import_error)
def xds_from_zarr(store, columns=None, chunks=None):

    """
    Reads the zarr data store in `store` and returns list of
    Dataset's containing the data.

    Parameters
    ----------
    store : str or Path
        Path containing the data
    columns : list of str or str or None
        Columns to read. `None` or `"ALL"` stores all columns on each dataset.
        Otherwise, a list of columns should be supplied.
    chunks: dict or list of dicts
        chunking schema for each dataset

    Returns
    -------
    writes : Dataset
        A Dataset representing the write operations
    """

    if isinstance(store, Path):
        store = str(store)

    if not isinstance(store, str):
        raise TypeError("store must be a Path, str")

    columns = promote_columns(columns)

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
    numpy_vars = []

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
        coords = {}

        for name, zarray in column_iterator(group, columns):
            attrs = dict(zarray.attrs[DASKMS_ATTR_KEY])
            dims = attrs.pop("dims")
            coordinate = attrs.pop("coordinate", False)
            mod, typ = attrs.pop("array_type").rsplit(".", 1)
            mod = importlib.import_module(mod)
            typ = getattr(mod, typ)
            array_chunks = tuple(group_chunks.get(d, s) for d, s
                                 in zip(dims, zarray.shape))

            array_chunks = da.core.normalize_chunks(array_chunks, zarray.shape)
            ext_args = extent_args(dims, array_chunks)

            read = da.blockwise(_getter_wrapper, dims,
                                zarray, None,
                                *ext_args,
                                concatenate=False,
                                meta=np.empty((0,)*zarray.ndim, zarray.dtype))

            read = inlined_array(read, ext_args[::2])
            var = Variable(dims, read, attrs)
            (coords if coordinate else data_vars)[name] = var

            # Save numpy arrays for reification
            if isinstance(typ, np.ndarray):
                numpy_vars.append(var)

        datasets.append(Dataset(data_vars, coords=coords, attrs=group_attrs))

    # Reify any numpy arrays directly into their variables
    for v, a in zip(numpy_vars, dask.compute(v.data for v in numpy_vars)[0]):
        v.data = a

    return datasets
