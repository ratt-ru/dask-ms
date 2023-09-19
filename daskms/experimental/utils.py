import logging
from pathlib import Path
from itertools import product

import dask
import dask.array as da
from dask.highlevelgraph import HighLevelGraph
import numpy as np


log = logging.getLogger(__name__)


def extent_args(dims, chunks):
    args = []
    meta = np.empty((1,), dtype=np.int32)

    for dim, chunks in zip(dims, chunks):
        name = "-".join((dim, dask.base.tokenize(chunks)))
        layers = {}
        start = 0

        for i, c in enumerate(chunks):
            end = start + c
            layers[(name, i)] = (start, end)
            start = end

        graph = HighLevelGraph.from_collections(name, layers, [])
        args.append(da.Array(graph, name, chunks=(chunks,), meta=meta))
        args.append((dim,))

    return args


def column_iterator(variables, columns):
    if columns is None or columns == "ALL":
        for k, v in variables.items():
            yield k, v
    else:
        for c in set(columns) & set(variables.keys()):
            yield c, variables[c]


def promote_columns(columns):
    if columns is None:
        return "ALL"
    elif columns == "ALL":
        return columns
    elif isinstance(columns, (tuple, list)):
        assert all(isinstance(c, str) for c in columns)
        return columns
    elif isinstance(columns, str):
        return [columns]
    else:
        raise TypeError(
            f"'columns' must be None or str " f"or list of str. Got {columns}"
        )


def select_vars_and_coords(dataset, columns):
    """
    Returns
    -------
    data_vars : dict
        Dictionary of data variables
    coords : dict
        Dictionary of coordinates
    """
    if columns is None or columns == "ALL":
        # Return all data variables and coordinates
        ret_data_vars = dict(dataset.data_vars)
        ret_coords = dict(dataset.coords)
    else:
        # Use specified variables and coordinates
        # BUT, we also include any coordinates
        # that variables and other coordinates depend on
        column_set = set(columns)
        data_vars = dataset.data_vars
        coords = dataset.coords
        data_var_names = set(data_vars.keys())
        coord_names = set(coords.keys())

        if not column_set.issubset((data_var_names | coord_names)):
            raise ValueError(
                f"User requested writes on the following "
                f"columns: {column_set}. Some or all of these "
                f"are not present on the datasets. Aborting."
            )

        data_sel = column_set & data_var_names
        coord_sel = column_set & coord_names

        for dv in data_sel:
            coord_sel = coord_sel.union(set(data_vars[dv].coords.keys()))

        ret_data_vars = {col: data_vars[col] for col in data_sel}
        ret_coords = {c: coords[c] for c in coord_sel}

    return ret_data_vars, ret_coords


def store_path_split(store):
    if not isinstance(store, Path):
        store = Path(store)

    parts = store.name.split("::", 1)

    if len(parts) == 1:
        name = parts[0]
        subtable = "MAIN"
    elif len(parts) == 2:
        name, subtable = parts

        if subtable == "MAIN":
            raise ValueError("'MAIN' is a reserved subtable name")
    else:
        raise RuntimeError(f"len(parts) {len(parts)} not in (1, 2)")

    return store.parent / name, subtable


def largest_chunk(arr):
    return max(arr.blocks[i].nbytes for i in product(*map(range, arr.blocks.shape)))


def rechunk_by_size(
    dataset, max_chunk_mem=2**31 - 1, unchunked_dims=None, only_when_needed=False
):
    """
    Given an xarray.Dataset, rechunk it such that chunking is uniform and
    consistent in all dimensions and all chunks are smaller than a specified
    size in bytes.

    Parameters
    ----------
    dataset : xarray.Dataset
        A dataset containing datavars and coords.
    max_chunk_mem : int
        Target maximum chunk size in bytes.
    unchunked_dims: None or set
        A set of dimensions which should not be chunked.
    only_when_needed: bool
        If set, only rechunk if existing chunks violate max_chunk_mem.

    Returns
    -------
    rechunked_dataset : xarray.Dataset
        Dataset with appropriate chunking.
    """

    def _rechunk(data_array, unchunked_dims):
        dims = set(data_array.dims)
        unchunked_dims = unchunked_dims & dims
        chunked_dims = dims - unchunked_dims

        n_dim = len(dims)
        n_unchunked_dim = len(unchunked_dims)
        n_chunked_dim = n_dim - n_unchunked_dim

        dim_sizes = data_array.sizes

        # The maximum number of array elements in the chunk.
        max_n_ele = max_chunk_mem // data_array.dtype.itemsize
        # The number of elements associated with unchunkable dimensions.
        fixed_n_ele = np.product([dim_sizes[k] for k in unchunked_dims])

        if fixed_n_ele > max_n_ele:
            raise ValueError(
                f"Target chunk size could not be reached in rechunk_by_size. "
                f"Unchunkable dimensions were: {unchunked_dims}."
            )

        chunk_dict = {k: dim_sizes[k] for k in unchunked_dims}

        if n_chunked_dim == 0:  # No chunking but still less than target size.
            return chunk_dict

        ideal_chunk = int(
            np.power(max_n_ele / fixed_n_ele, 1 / (n_dim - n_unchunked_dim))
        )

        chunk_dict.update({k: ideal_chunk for k in chunked_dims})

        new_unchunked_dims = {k for k in dims if chunk_dict[k] >= dim_sizes[k]}

        if len(new_unchunked_dims) == n_dim:
            return {k: dim_sizes[k] for k in unchunked_dims}
        elif new_unchunked_dims != unchunked_dims:
            return _rechunk(data_array, new_unchunked_dims)
        else:
            return chunk_dict

    # Figure out chunking from the largest arrays to the smallest. NOTE:
    # Using nbytes may be unreliable for object arrays.
    dvs_and_coords = [*dataset.data_vars.values(), *dataset.coords.values()]
    dvs_and_coords = [d for d in dvs_and_coords if isinstance(d.data, da.Array)]
    dvs_and_coords = sorted(dvs_and_coords, key=lambda arr: arr.data.nbytes)

    if only_when_needed:
        largest_chunks = [largest_chunk(dc.data) for dc in dvs_and_coords]
        if not any(lc > max_chunk_mem for lc in largest_chunks):
            return dataset.copy()

    chunk_dims = {}

    for data_array in dvs_and_coords[::-1]:  # From largest to smallest.
        chunk_update = _rechunk(data_array, unchunked_dims or set())

        chunk_dims.update(
            {
                k: min(chunk_update[k], chunk_dims.get(k, chunk_update[k]))
                for k in chunk_update.keys()
            }
        )

    rechunked_dataset = dataset.chunk(chunk_dims)

    return rechunked_dataset
