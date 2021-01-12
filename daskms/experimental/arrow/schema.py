from collections import defaultdict
import json

from daskms.dataset import Variable
from daskms.utils import requires
from daskms.experimental.arrow.extension_types import TensorArray

import numpy as np

try:
    import pyarrow as pa
except ImportError as e:
    pyarrow_import_error = e
else:
    pyarrow_import_error = None

try:
    from xarray import Variable as xVariable
except ImportError:
    VariableTypes = (Variable,)
else:
    VariableTypes = (Variable, xVariable)

DASKMS_METADATA = "__daskms_metadata__"


def variable_field(column, variable):
    if isinstance(variable, VariableTypes):
        variable = [variable]
    elif not isinstance(variable, (tuple, list)):
        variable = [variable]

    if len(variable) == 0:
        return {}

    dims = set()
    dtypes = set()
    ndims = set()
    shapes = set()

    for v in variable:
        dims.add(v.dims)
        dtypes.add(v.dtype)
        ndims.add(v.ndim)
        shapes.add(v.shape[1:])

    if len(dtypes) > 1:
        raise ValueError(f"Multiple dtypes {dtypes} in {column}. "
                         f"Please cast your variables to the same dtype")
    else:
        dtype = dtypes.pop()

    if len(ndims) > 1:
        raise ValueError(f"Multiple dimensions {ndims} "
                         f"discovered for {column}. "
                         f"This is not currently supported.")
    else:
        ndim = ndims.pop()

    if len(shapes) > 1:
        raise ValueError(f"Multiple shapes {shapes} discovered for {column}. "
                         f"This is not currently supported.")

    if len(dims) > 1:
        raise ValueError(f"Multiple dimension schema {dims} "
                         f"discovered for {column}. "
                         f"This is not currently supported.")
    else:
        dims = dims.pop()

    factory = pa.array if ndim == 1 else TensorArray.from_numpy
    pa_type = factory(np.empty((0,)*ndim, dtype)).type
    metadata = {DASKMS_METADATA: json.dumps({"dims": dims})}

    return pa.field(column, pa_type, metadata=metadata, nullable=False)


@requires("pip install dask-ms[arrow] for arrow support",
          pyarrow_import_error)
def dataset_schema(datasets):
    dataset_vars = defaultdict(list)

    for ds in datasets:
        for c, v in ds.data_vars.items():
            dataset_vars[c].append(v)

    return pa.schema([variable_field(c, v) for c, v in dataset_vars.items()])
