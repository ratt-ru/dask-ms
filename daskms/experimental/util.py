import dask
import dask.array as da
from dask.highlevelgraph import HighLevelGraph
import numpy as np


def encode_attr(a):
    """ Convert a into something acceptable to json """
    if isinstance(a, tuple):
        return tuple(encode_attr(v) for v in a)
    elif isinstance(a, list):
        return list(encode_attr(v) for v in a)
    elif isinstance(a, dict):
        return {k: encode_attr(v) for k, v in a.items()}
    elif isinstance(a, np.ndarray):
        return a.tolist()
    elif isinstance(a, np.generic):
        return a.item()
    else:
        return a


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
