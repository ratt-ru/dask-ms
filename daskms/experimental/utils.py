import dask
import dask.array as da
from dask.highlevelgraph import HighLevelGraph
import numpy as np

from daskms.dataset import Dataset

# TODO(sjperkins)
# Integrate this into the CASA Table functionality at some point
DATASET_TYPES = (Dataset,)
DATASET_TYPE = Dataset

try:
    import xarray as xr
except ImportError as e:
    xarray_import_error = e
else:
    xarray_import_error = None
    DATASET_TYPES += (xr.Dataset,)
    DATASET_TYPE = xr.Dataset


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
