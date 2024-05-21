from functools import reduce
from operator import mul
from pathlib import Path
import os.path
from uuid import uuid4
import warnings

import dask
import dask.array as da
from dask.array.core import normalize_chunks
from dask.base import tokenize
import numpy as np
import warnings

ARRAY_DIMENSION = "_ARRAY_DIMENSIONS"

from daskms.constants import DASKMS_PARTITION_KEY
from daskms.dataset import Dataset, Variable
from daskms.dataset_schema import DatasetSchema, encode_type, decode_type, decode_attr
from daskms.experimental.utils import (
    extent_args,
    select_vars_and_coords,
    column_iterator,
    promote_columns,
)
from daskms.optimisation import inlined_array
from daskms.utils import requires
from daskms.fsspec_store import DaskMSStore

try:
    import zarr
    import zarr.convenience as zc
except ImportError as e:
    zarr_import_error = e
else:
    zarr_import_error = None


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
                f"This is not currently supported"
            )

        unique_chunks = set(dim_chunks[:-1])

        if len(unique_chunks) == 0:
            zchunks.append(dim_chunks[-1])
        elif len(unique_chunks) == 1:
            zchunks.append(dim_chunks[0])
        else:
            raise NotImplementedError(
                f"Column {column} has heterogenous chunks "
                f"{dim_chunks} in dimension {dim} "
                f"zarr does not currently support this"
            )

    return tuple(zchunks)


def create_array(ds_group, column, column_schema, schema_chunks, coordinate=False):
    import numcodecs

    codec = numcodecs.JSON() if column_schema.dtype == object else None

    if column_schema.chunks is None:
        try:
            # No column chunking found, probably an ndarray,
            # derive column chunking from chunks on dataset
            chunks = tuple(schema_chunks[d] for d in column_schema.dims)
        except KeyError:
            # Nope, just set chunks equal to dimension size
            chunks = tuple((s,) for s in column_schema.shape)
    else:
        chunks = column_schema.chunks

    zchunks = zarr_chunks(column, column_schema.dims, chunks)

    if column_schema.dtype == object:
        if reduce(mul, zchunks, 32) >= 2 ** (32 - 1):
            raise ValueError(
                f"Column {column} has an object dtype. "
                f"Given an estimate of 32 bytes per entry "
                f"the chunk of dimensions {zchunks}"
                f"may exceed zarr's 2GiB chunk limit"
            )
    else:
        size = np.dtype(column_schema.dtype).itemsize
        if reduce(mul, zchunks, size) >= 2 ** (32 - 1):
            raise ValueError(
                f"Column {column} has a chunk of "
                f"dimension {zchunks} that will exceed "
                f"zarr's 2GiB chunk limit. Consider calling "
                f"daskms.experimental.utils.rechunk_by_size "
                f"prior to writing."
            )

    array = ds_group.require_dataset(
        column,
        column_schema.shape,
        chunks=zchunks,
        fill_value=None,
        dtype=column_schema.dtype,
        object_codec=codec,
        exact=True,
    )

    array.attrs[ARRAY_DIMENSION] = column_schema.dims

    array.attrs[DASKMS_ATTR_KEY] = {
        **column_schema.attrs,
        "dims": column_schema.dims,
        "coordinate": coordinate,
        "array_type": encode_type(column_schema.type),
    }


def prepare_zarr_group(dataset_id, dataset, store, rechunk=False):
    try:
        # Open in read/write, must exist
        group = zarr.open_group(store=store.map, mode="r+")
    except zarr.errors.GroupNotFoundError:
        # Create, must not exist
        group = zarr.open_group(store=store.map, mode="w-")

    table_path = store.table if store.table else "MAIN"

    group_name = f"{table_path}_{dataset_id}"
    ds_group = group.require_group(table_path).require_group(group_name)

    dataset, ds_group = maybe_rechunk(dataset, ds_group, rechunk=rechunk)

    schema = DatasetSchema.from_dataset(dataset)
    schema_chunks = schema.chunks

    for column, column_schema in schema.data_vars.items():
        create_array(ds_group, column, column_schema, schema_chunks, False)

    for column, column_schema in schema.coords.items():
        create_array(ds_group, column, column_schema, schema_chunks, True)

    ds_group.attrs.update(
        {**schema.attrs, DASKMS_ATTR_KEY: {"chunks": dict(dataset.chunks)}}
    )

    return dataset, ds_group


def get_group_chunks(group):
    group_chunks = {}

    for array in group.values():
        array_chunks = normalize_chunks(array.chunks, array.shape)
        array_dims = decode_attr(array.attrs[DASKMS_ATTR_KEY])["dims"]
        group_chunks.update(dict(zip(array_dims, array_chunks)))

    return group_chunks


def maybe_rechunk(dataset, group, rechunk=False):
    group_chunks = get_group_chunks(group)
    dataset_chunks = dataset.chunks

    for name, data in (*dataset.data_vars.items(), *dataset.coords.items()):
        try:
            disk_chunks = tuple(
                group_chunks.get(d, dataset_chunks[d]) for d in data.dims
            )
        except KeyError:  # Orphan coordinate (no chunks), handled elsewhere.
            continue
        dask_chunks = data.chunks

        if dask_chunks and (dask_chunks != disk_chunks):
            if rechunk:
                disk_chunks = dict(zip(data.dims, disk_chunks))
                dataset = dataset.assign({name: data.chunk(chunks=disk_chunks)})
            else:
                raise ValueError(
                    f"On disk (zarr) chunks: {disk_chunks} - don't match in "
                    f"memory (dask) chunks: {dask_chunks}. This can cause "
                    f"data corruption as described in "
                    f"https://zarr.readthedocs.io/en/stable/tutorial.html"
                    f"#parallel-computing-and-synchronization. Consider "
                    f"setting 'rechunk=True' in 'xds_to_zarr'."
                )

    try:
        dataset.chunks
    except ValueError as e:
        raise e

    # This makes the attributes consistent with the final chunking.
    group.attrs.update({DASKMS_ATTR_KEY: {"chunks": dict(dataset.chunks)}})

    return dataset, group


def zarr_setter(data, name, group, *extents):
    try:
        zarray = getattr(group, name)
    except AttributeError:
        raise ValueError(f"{name} is not a variable of {group}")

    selection = tuple(slice(start, end) for start, end in extents)
    zarray[selection] = data
    return np.full((1,) * len(extents), True)


def _gen_writes(variables, chunks, factory, epoch, indirect_dims=False):
    for name, var in variables.items():
        if isinstance(var.data, da.Array):
            ext_args = extent_args(var.dims, var.chunks)
            var_data = var.data
        elif isinstance(var.data, np.ndarray):
            try:
                var_chunks = tuple(chunks[d] for d in var.dims)
            except KeyError:
                var_chunks = tuple((s,) for s in var.shape)
            ext_args = extent_args(var.dims, var_chunks)
            var_data = da.from_array(
                var.data, chunks=var_chunks, inline_array=True, name=False
            )
        else:
            raise NotImplementedError(f"Writing {type(var.data)} " f"unsupported")

        if var_data.nbytes == 0:
            continue

        token_name = (
            f"write~{name}-" f"{tokenize(var_data, name, factory, epoch, *ext_args)}"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=da.PerformanceWarning)
            write = da.blockwise(
                zarr_setter,
                var.dims,
                var_data,
                var.dims,
                name,
                None,
                factory,
                None,
                *ext_args,
                adjust_chunks={d: 1 for d in var.dims},
                concatenate=False,
                name=token_name,
                meta=np.empty((1,) * len(var.dims), bool),
            )
        write = inlined_array(write, ext_args[::2])

        # Alter the dimension names to preserve laziness on coordinates.
        dims = [f"_{d}_" for d in var.dims] if indirect_dims else var.dims

        yield name, (dims, write, var.attrs)


@requires("pip install dask-ms[zarr] for zarr support", zarr_import_error)
def xds_to_zarr(
    xds, store, columns=None, rechunk=False, consolidated=True, epoch=None, **kwargs
):
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
        Otherwise, a list of columns should be supplied. All coordinates
        associated with a specified column will be written automatically.
    rechunk : bool
        Controls whether dask arrays should be automatically rechunked to be
        consistent with existing on-disk zarr arrays while writing to disk.
    consolidated : bool
        Controls whether metadata is consolidated
    epoch : str or None
        Uniquely identifies this instance of the returned dataset.
        Should usually be set to None.
    **kwargs : optional

    Returns
    -------
    writes : Dataset
        A Dataset representing the write operations
    """
    if isinstance(store, DaskMSStore):
        pass
    elif isinstance(store, (Path, str)):
        store = DaskMSStore(f"{store}", **kwargs.pop("storage_options", {}))
    else:
        raise TypeError(f"store '{store}' must be " f"Path, str or DaskMSStore")

    # If any kwargs are added, they should be popped prior to this check.
    if len(kwargs) > 0:
        warnings.warn(
            f"The following unsupported kwargs were ignored in "
            f"xds_to_zarr: {kwargs}",
            UserWarning,
        )

    epoch = epoch or uuid4().hex[:16]
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
        data_vars, coords = select_vars_and_coords(ds, columns)

        # Create a new ds which is consistent with what we want to write.
        ds = Dataset(data_vars, coords=coords, attrs=ds.attrs)

        ds, group = prepare_zarr_group(di, ds, store, rechunk=rechunk)

        data_vars = dict(_gen_writes(ds.data_vars, ds.chunks, group, epoch))
        # Include coords in the write dataset so they're reified
        data_vars.update(
            dict(_gen_writes(ds.coords, ds.chunks, group, epoch, indirect_dims=True))
        )

        # Transfer any partition information over to the write dataset
        partition = ds.attrs.get(DASKMS_PARTITION_KEY, False)

        if not partition:
            attrs = None
        else:
            attrs = {
                DASKMS_PARTITION_KEY: partition,
                **{k: getattr(ds, k) for k, _ in partition},
            }

        if consolidated:
            table_name = store.table if store.table else "MAIN"
            sep = store.fs.sep
            store_path = f"{store.root}{sep}{table_name}{sep}{table_name}_{di}"
            store_map = store.fs.get_mapper(store_path)
            zc.consolidate_metadata(store_map)

        write_datasets.append(Dataset(data_vars, attrs=attrs))

    return write_datasets


def zarr_getter(zarray, *extents):
    if any([start == end for start, end in extents]):  # Empty slice.
        shape = [start - end for start, end in extents]
        return np.empty(shape, dtype=zarray.dtype)
    else:
        return zarray[tuple(slice(start, end) for start, end in extents)]


def group_sortkey(element):
    return int(element[0].split("_")[-1])


@requires("pip install dask-ms[zarr] for zarr support", zarr_import_error)
def xds_from_zarr(
    store, columns=None, chunks=None, consolidated=True, epoch=None, **kwargs
):
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
    consolidated : bool
        If True, attempt to read consolidated metadata
    epoch : str or None
        Uniquely identifies this instance of the returned dataset.
        Should usually be set to None.
    **kwargs: optional

    Returns
    -------
    writes : Dataset or list of Datasets
        Dataset(s) representing write operations
    """

    if isinstance(store, DaskMSStore):
        pass
    elif isinstance(store, (Path, str)):
        store = DaskMSStore(f"{store}", **kwargs.pop("storage_options", {}))
    else:
        raise TypeError(f"store '{store}' must be " f"Path, str or DaskMSStore")

    store.assert_type("zarr")

    # If any kwargs are added, they should be popped prior to this check.
    if len(kwargs) > 0:
        warnings.warn(
            f"The following unsupported kwargs were ignored in "
            f"xds_from_zarr: {kwargs}",
            UserWarning,
        )

    epoch = epoch or uuid4().hex[:16]
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

    datasets = []
    numpy_vars = []
    table_name = store.table if store.table else "MAIN"

    store_path = f"{store.root}{store.fs.sep}{table_name}"
    store_map = store.fs.get_mapper(store_path)

    partition_ids = []

    for entry in store_map.fs.listdir(f"{store_map.root}"):
        if entry["type"] == "directory":
            _, dir_name = os.path.split(entry["name"])
            if dir_name.startswith(table_name):
                _, i = dir_name[len(table_name) :].split("_")
                partition_ids.append(int(i))

    for g in sorted(partition_ids):
        group_path = f"{store_path}{store.fs.sep}{table_name}_{g}"
        group_map = store.fs.get_mapper(group_path)

        if consolidated:
            try:
                group = zarr.open_consolidated(group_map, mode="r")
            except KeyError:
                group = zarr.open_group(group_map, mode="r")
        else:
            group = zarr.open_group(group_map, mode="r")

        group_attrs = decode_attr(dict(group.attrs))
        dask_ms_attrs = group_attrs.pop(DASKMS_ATTR_KEY)
        natural_chunks = dask_ms_attrs["chunks"]
        group_chunks = {d: tuple(dc) for d, dc in natural_chunks.items()}

        if chunks:
            # Defer to user-supplied chunking strategy
            try:
                group_chunks.update(chunks[g])
            except IndexError:
                group_chunks.update(chunks[-1])  # Reuse last chunking.
                pass

        data_vars = {}
        coords = {}

        for name, zarray in column_iterator(group, columns):
            attrs = decode_attr(dict(zarray.attrs[DASKMS_ATTR_KEY]))
            dims = attrs["dims"]
            coordinate = attrs.get("coordinate", False)
            array_chunks = tuple(
                group_chunks.get(d, s) for d, s in zip(dims, zarray.shape)
            )

            array_chunks = da.core.normalize_chunks(array_chunks, zarray.shape)
            ext_args = extent_args(dims, array_chunks)
            token_name = f"read~{name}-{tokenize(zarray, epoch, *ext_args)}"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=da.PerformanceWarning)
                read = da.blockwise(
                    zarr_getter,
                    dims,
                    zarray,
                    None,
                    *ext_args,
                    concatenate=False,
                    name=token_name,
                    meta=np.empty((0,) * zarray.ndim, zarray.dtype),
                )

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
                raise TypeError(f"Unknown array_type '{attrs['array_type']}'")

        datasets.append(Dataset(data_vars, coords=coords, attrs=group_attrs))

    # Reify any numpy arrays directly into their variables
    for v, a in zip(numpy_vars, dask.compute(v.data for v in numpy_vars)[0]):
        v.data = a

    return datasets
