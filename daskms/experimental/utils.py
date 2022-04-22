import logging
from pathlib import Path

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
        for c in (set(columns) & set(variables.keys())):
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
        raise TypeError(f"'columns' must be None or str "
                        f"or list of str. Got {columns}")


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
            raise ValueError(f"User requested writes on the following "
                             f"columns: {column_set}. Some or all of these "
                             f"are not present on the datasets. Aborting.")

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
