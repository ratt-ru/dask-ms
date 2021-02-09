import dask
import dask.array as da
from dask.highlevelgraph import HighLevelGraph
import numpy as np


def encode_attr(arg):
    """ Convert arg into something acceptable to json """
    if isinstance(arg, tuple):
        return tuple(map(encode_attr, arg))
    elif isinstance(arg, list):
        return list(map(encode_attr, arg))
    elif isinstance(arg, set):
        return list(map(encode_attr, sorted(arg)))
    elif isinstance(arg, dict):
        return {k: encode_attr(v) for k, v in arg.items()}
    elif isinstance(arg, np.ndarray):
        return arg.tolist()
    elif isinstance(arg, np.generic):
        return arg.item()
    else:
        return arg


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
