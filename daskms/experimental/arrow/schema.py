# TODO(sjperkins): remove this
# flake8: noqa

from daskms.dataset import Variable

try:
    import pyarrow
except ImportError:
    pyarrow = None

try:
    from xarray import Variable as xVariable
except ImportError:
    VariableTypes = (Variable,)
else:
    VariableTypes = (Variable, xVariable)


def variable_column_schema(column, variable):
    if isinstance(variable, VariableTypes):
        variable = [variable]
    elif not isinstance(variable, (tuple, list)):
        variable = [variable]

    if len(variable) == 0:
        return {}

    dtypes = set()
    ndims = set()
    shapes = set()

    for v in variable:
        dtypes.add(v.dtype)
        ndims.add(v.ndim)
        shapes.add(v.shape)

    if len(dtypes) == 0:
        raise ValueError(f"No dtypes discovered for {column}")
    elif len(dtypes) > 1:
        raise ValueError(f"Multiple dtypes {dtypes} in {column}. "
                         f"Please cast your variables to the same dtype")
    else:
        dtype = dtypes.pop()

    if len(ndims) == 0:
        raise ValueError(f"No ndims discovered for {column}")
    elif len(ndims) > 1:
        raise ValueError(f"Multiple ndims {ndims} discovered for {column}. "
                         f"This is not currently supported.")
    else:
        ndim = ndims.pop()

    if len(shapes) == 0:
        raise ValueError(f"No shapes discovered for {column}")
    else:
        for shape in shapes:
            if len(shape) == ndim:
                continue

            raise ValueError(f"Shape lengths don't not match ndim {ndim}.")


def dataset_schema(datasets):
    for ds in datasets:
        pass