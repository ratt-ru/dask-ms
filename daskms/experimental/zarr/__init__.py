from pathlib import Path

import dask
import dask.array as da
from dask.array.core import normalize_chunks
import numcodecs
import numpy as np

from daskms.utils import requires
from daskms.dataset import Dataset, Variable
from daskms.dataset_schema import DatasetSchema, encode_type, decode_type
from daskms.experimental.utils import (extent_args,
                                       column_iterator,
                                       promote_columns)
from daskms.optimisation import inlined_array

try:
    import zarr
except ImportError as e:
    zarr_import_error = e
else:
    zarr_import_error = None


DATASET_PREFIX = "__daskms_dataset__"
DASKMS_ATTR_KEY = "__daskms_zarr_attr__"


def zarr_chunks(column, dims, chunks):
    if chunks is None:
        return None

    zchunks = []

    for dim, dim_chunks in zip(dims, chunks):
        if any(np.isnan(dc) for dc in dim_chunks):
            raise NotImplementedError(
                f"Column {column} has nan chunks "
                f"{dim_chunks} in dimension {dim} "
                f"This is not currently supported")

        unique_chunks = set(dim_chunks[:-1])

        if len(unique_chunks) == 0:
            zchunks.append(dim_chunks[-1])
        elif len(unique_chunks) == 1:
            zchunks.append(dim_chunks[0])
        else:
            raise NotImplementedError(
                f"Column {column} has heterogenous chunks "
                f"{dim_chunks} in dimension {dim} "
                f"zarr does not currently support this")

    return tuple(zchunks)


def create_array(ds_group, column, schema, coordinate=False):
    codec = numcodecs.Pickle() if schema.dtype == np.object else None

    zchunks = zarr_chunks(column, schema.dims, schema.chunks)

    array = ds_group.require_dataset(column, schema.shape,
                                     chunks=zchunks,
                                     dtype=schema.dtype,
                                     object_codec=codec,
                                     exact=True)

    if zchunks is not None:
        # Expand zarr chunks to full dask resolution
        # For comparison purposes
        zchunks = normalize_chunks(array.chunks, schema.shape)

        if zchunks != schema.chunks:
            raise ValueError(
                   f"zarr chunks {zchunks} "
                   f"don't match dask chunks {schema.chunks}. "
                   f"This can cause data corruption as described in "
                   f"https://zarr.readthedocs.io/en/stable/tutorial.html"
                   f"#parallel-computing-and-synchronization")

    array.attrs[DASKMS_ATTR_KEY] = {
        "dims": schema.dims,
        "coordinate": coordinate,
        "array_type": encode_type(schema.type),
    }


def prepare_zarr_group(dataset_id, dataset, store):
    dir_store = zarr.DirectoryStore(store)

    try:
        # Open in read/write, must exist
        group = zarr.open_group(store=dir_store, mode="r+")
    except zarr.errors.GroupNotFoundError:
        # Create, must not exist
        group = zarr.open_group(store=dir_store, mode="w-")

    group_name = f"{DATASET_PREFIX}{dataset_id:08d}"
    ds_group = group.require_group(group_name)

    schema = DatasetSchema.from_dataset(dataset)

    for column, column_schema in schema.data_vars.items():
        create_array(ds_group, column, column_schema, False)

    for column, column_schema in schema.coords.items():
        create_array(ds_group, column, column_schema, True)

    ds_group.attrs.update({
        **schema.attrs,
        DASKMS_ATTR_KEY: {"chunks": dict(dataset.chunks)}
    })

    return ds_group


def zarr_setter(data, name, group, *extents):
    try:
        zarray = getattr(group, name)
    except AttributeError:
        raise ValueError(f"{name} is not a variable of {group}")

    selection = tuple(slice(start, end) for start, end in extents)
    zarray[selection] = data
    return np.full((1,)*len(extents), True)


def _gen_writes(variables, chunks, columns, factory):
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

        write = da.blockwise(zarr_setter, var.dims,
                             var_data, var.dims,
                             name, None,
                             factory, None,
                             *ext_args,
                             adjust_chunks={d: 1 for d in var.dims},
                             concatenate=False,
                             meta=np.empty((1,)*len(var.dims), np.bool))
        write = inlined_array(write, ext_args[::2])

        yield name, (var.dims, write, var.attrs)


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

    for di, ds in enumerate(xds):
        group = prepare_zarr_group(di, ds, store)
        write_args = (ds.chunks, columns, group)

        data_vars = dict(_gen_writes(ds.data_vars, *write_args))
        # Include coords in the write dataset so they're reified
        data_vars.update(dict(_gen_writes(ds.coords, *write_args)))
        write_datasets.append(Dataset(data_vars))

    return write_datasets


def zarr_getter(zarray, *extents):
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
        dask_ms_attrs = group_attrs.pop(DASKMS_ATTR_KEY)
        natural_chunks = dask_ms_attrs["chunks"]
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
            dims = attrs["dims"]
            coordinate = attrs.get("coordinate", False)
            array_chunks = tuple(group_chunks.get(d, s) for d, s
                                 in zip(dims, zarray.shape))

            array_chunks = da.core.normalize_chunks(array_chunks, zarray.shape)
            ext_args = extent_args(dims, array_chunks)

            read = da.blockwise(zarr_getter, dims,
                                zarray, None,
                                *ext_args,
                                concatenate=False,
                                meta=np.empty((0,)*zarray.ndim, zarray.dtype))

            read = inlined_array(read, ext_args[::2])
            var = Variable(dims, read, attrs)
            (coords if coordinate else data_vars)[name] = var

            # Save numpy arrays for reification
            typ = decode_type(attrs["array_type"])

            if typ is np.ndarray:
                numpy_vars.append(var)
            elif typ is da.Array:
                pass
            else:
                raise TypeError(f"Unknown {typ}")

        datasets.append(Dataset(data_vars, coords=coords, attrs=group_attrs))

    # Reify any numpy arrays directly into their variables
    for v, a in zip(numpy_vars, dask.compute(v.data for v in numpy_vars)[0]):
        v.data = a

    return datasets
