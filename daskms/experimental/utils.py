import dask
import dask.array as da
from dask.highlevelgraph import HighLevelGraph
import numpy as np


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
        return variables.items()
    else:
        column_set = set(columns)
        unknown_columns = column_set - set(variables.keys())

        if len(unknown_columns) > 0:
            raise ValueError(f"{unknown_columns} are not present "
                             f"in the dataset.")

        return ((k, variables[k]) for k in column_set)


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
