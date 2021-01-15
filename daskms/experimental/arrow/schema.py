from collections import defaultdict
import json

from daskms.dataset import Variable
from daskms.reads import PARTITION_KEY
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
DASKMS_PARQUET_VERSION = "0.0.1"


def variable_schema(column, variable):
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
        raise ValueError(f"Multiple dtypes {dtypes} "
                         f"discovered for {column}. "
                         f"Please cast your variables to the same dtype")
    else:
        dtype = dtypes.pop()

    if len(ndims) > 1:
        raise ValueError(f"Multiple dimensions {ndims} "
                         f"discovered for {column}. "
                         f"This is not currently supported.")
    else:
        ndim = ndims.pop()

    if len(dims) > 1:
        raise ValueError(f"Multiple dimension schema {dims} "
                         f"discovered for {column}. "
                         f"This is not currently supported.")
    else:
        dims = dims.pop()

    if len(shapes) > 1:
        dim_shapes = tuple({d: s for d, s in zip(dims, shape)}
                           for shape in shapes)

        raise ValueError(f"Multiple shapes {dim_shapes} "
                         f"discovered for {column}. "
                         f"This is not currently supported.")

    return (column, ndim, dtype, {"dims": dims})


def dict_dataset_schema(datasets):
    dataset_vars = defaultdict(list)

    try:
        partition_exemplar = next(iter(datasets)).attrs[PARTITION_KEY]
    except StopIteration:
        pass
    except KeyError:
        raise ValueError("Datasets don't contain partitioning information")

    for ds in datasets:
        for c, v in ds.data_vars.items():
            dataset_vars[c].append(v)

        if partition_exemplar != ds.attrs.get(PARTITION_KEY, None):
            raise ValueError("Partitioning is not consistent across datasets")

    var_schemas = [variable_schema(c, v) for c, v in dataset_vars.items()]
    return (var_schemas, {"version": DASKMS_PARQUET_VERSION,
                          PARTITION_KEY: partition_exemplar})


@requires("pip install dask-ms[arrow] for arrow support",
          pyarrow_import_error)
def dataset_schema(schema):
    var_schemas, table_metadata = schema
    fields = []

    for (column, ndim, dtype, metadata) in var_schemas:
        factory = pa.array if ndim == 1 else TensorArray.from_numpy
        pa_type = factory(np.empty((0,)*ndim, dtype)).type
        metadata = {DASKMS_METADATA: json.dumps(metadata)}
        field = pa.field(column, pa_type, metadata=metadata, nullable=False)
        fields.append(field)

    table_metadata = {DASKMS_METADATA: json.dumps(table_metadata)}
    return pa.schema(fields, metadata=table_metadata)
