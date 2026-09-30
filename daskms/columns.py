# -*- coding: utf-8 -*-

from collections import OrderedDict, namedtuple
import logging
from pprint import pformat

import dask
import dask.array as da
from dask.highlevelgraph import HighLevelGraph
import numpy as np

log = logging.getLogger(__name__)

# Map column string types to numpy/python types
_TABLE_TO_PY = OrderedDict(
    {
        "BOOL": "bool",
        "BOOLEAN": "bool",
        "BYTE": "uint8",
        "UCHAR": "uint8",
        "SMALLINT": "int16",
        "SHORT": "int16",
        "USMALLINT": "uint16",
        "USHORT": "uint16",
        "INT": "int32",
        "INTEGER": "int32",
        "UINTEGER": "uint32",
        "UINT": "uint32",
        "FLOAT": "float32",
        "DOUBLE": "float64",
        "FCOMPLEX": "complex64",
        "COMPLEX": "complex64",
        "DCOMPLEX": "complex128",
        "STRING": "object",
    }
)


# Map numpy/python types to column string types
_PY_TO_TABLE = OrderedDict(
    {
        "bool": "BOOLEAN",
        "uint8": "UCHAR",
        "int16": "SHORT",
        "uint16": "USHORT",
        "uint32": "UINT",
        "int32": "INTEGER",
        "float32": "FLOAT",
        "float64": "DOUBLE",
        "complex64": "COMPLEX",
        "complex128": "DCOMPLEX",
        "object": "STRING",
    }
)


def arcae_dtype(dtype):
    """The dtype arcae uses on the wire for ``dtype``.

    arcae hands casacore ``Bool`` columns back as ``uint8``. Arrow's boolean
    type is bit-packed while casacore stores a byte per value, so arcae
    exposes the buffer as the uint8 array it physically is. The layout is
    identical to numpy's ``bool_``, so this only matters when reading into a
    preallocated array, where handing arcae a ``bool_`` buffer makes it size
    the read as bit-packed and reject the eight-times-larger array we
    actually allocated.
    """
    return np.dtype(np.uint8) if np.dtype(dtype) == np.bool_ else np.dtype(dtype)


def infer_dtype(column, coldesc):
    # Extract valueType
    try:
        value_type = coldesc["valueType"]
    except KeyError:
        raise ValueError(
            "Cannot infer dtype for column '%s'. "
            "Table Column Description is missing "
            "valueType. Description is '%s'" % (column, coldesc)
        )

    # Try conversion to numpy/python type
    try:
        np_type_str = _TABLE_TO_PY[value_type.upper()]
    except KeyError:
        raise ValueError(
            "No known conversion from CASA Table type '%s' "
            "to python/numpy type. "
            "Perhaps it needs to be added "
            "to _TABLE_TO_PY?:\n"
            "%s" % (value_type, pformat(dict(_TABLE_TO_PY)))
        )
    else:
        return np.dtype(np_type_str)


def infer_casa_type(dtype):
    try:
        return _PY_TO_TABLE[np.dtype(dtype).name]
    except KeyError:
        raise ValueError(
            "No known conversion from numpy dtype '%s' "
            "to CASA Table Type. "
            "Perhaps it needs to be added "
            "to _TABLE_TO_PY?:\n"
            "%s" % (dtype, pformat(dict(_TABLE_TO_PY)))
        )


class ColumnMetadataError(Exception):
    pass


ColumnMetadata = namedtuple(
    "ColumnMetadata",
    ["shape", "dims", "chunks", "dtype", "variable"],
    defaults=(False,),
)


def _maximal_row_shape(column, table, rows, exemplar_row):
    """The per-dimension maximum of ``column``'s cell shapes over ``rows``.

    Undefined cells are ignored. If ``rows`` is empty, the
    ``exemplar_row`` cell supplies the shape instead.
    """
    if callable(rows):
        rows = rows()

    if rows is None:
        index = None
    elif len(rows) == 0:
        index = (slice(exemplar_row, exemplar_row + 1),)
    else:
        index = (np.asarray(rows),)

    try:
        shapes = table.row_shapes(column, index).drop_null()
    except Exception as e:
        raise ColumnMetadataError(f"Unable to infer shape of column '{column}'") from e

    if len(shapes) == 0:
        raise ColumnMetadataError(
            f"Unable to infer shape of column '{column}' as it has no defined rows"
        )

    ndim = shapes.type.list_size
    shapes = shapes.flatten().to_numpy().reshape(len(shapes), ndim)
    return tuple(int(s) for s in shapes.max(axis=0))


def column_metadata(
    column, table_proxy, table_schema, chunks, exemplar_row=0, rows=None
):
    """
    Infers column metadata for the purposes of creating dask arrays
    that reference their contents.

    Parameters
    ----------
    column : string
        Table column
    table_proxy : string
        CASA Table path
    table_schema : dict
        Table schema
    chunks : dict of tuple of ints
        :code:`{dim: chunks}` mapping
    exemplar_row : int, optional
        Table row whose shape is used for a variably shaped column
        if ``rows`` is empty.
    rows : :class:`numpy.ndarray` or callable, optional
        Rows of the dataset, or a callable returning them, which is only
        called for a variably shaped column. The shape of such a column is
        the maximal shape of its cells over these rows, so that every cell
        fits. Defaults to all rows in the table.


    Returns
    -------
    shape : tuple
        Shape of column (excluding the row dimension).
        For example :code:`(16, 4)`
    dims : tuple
        Dask dimension schema. For example :code:`("chan", "corr")`
    dim_chunks : list of tuples
        Dimension chunks. For example :code:`[chan_chunks, corr_chunks]`.
    dtype : :class:`numpy.dtype`
        Column data type (numpy)
    variable : bool
        True if the column is variably shaped. Cells smaller than
        ``shape`` are padded when read.


    Raises
    ------
    ColumnMetadataError
        Raised if inferring metadata failed.
    """
    try:
        coldesc = table_proxy.instance.getcoldesc(column)
    except Exception as e:
        raise ColumnMetadataError(
            f"Unable to obtain column descriptor for column '{column}'"
        ) from e
    dtype = infer_dtype(column, coldesc)
    # missing ndim implies only row dimension
    ndim = coldesc.get("ndim", "row")

    try:
        option = coldesc["option"]
    except KeyError as e:
        raise ColumnMetadataError(
            f"Column '{column}' has no 'option' in the column descriptor"
        ) from e

    # Each row is a scalar
    # TODO(sjperkins)
    # Probably could be handled by getCell/putCell calls,
    # but the effort may not be worth it
    if ndim == 0:
        raise ColumnMetadataError(
            f"Scalars in column '{column}' (ndim == {ndim}) are not currently handled"
        )
    # Only row dimensions
    elif ndim == "row":
        shape = ()
    # FixedShape
    elif option & 4:
        try:
            shape = tuple(coldesc["shape"])
        except KeyError as e:
            raise ColumnMetadataError(
                f"'{column}' column descriptor option '{option}' "
                f"specifies a FixedShape but no 'shape' "
                f"attribute was found in the "
                f"column descriptor"
            ) from e
    # Variably shaped...
    else:
        shape = _maximal_row_shape(column, table_proxy.instance, rows, exemplar_row)

        # NOTE(sjperkins)
        # -1 implies each row can be any shape whatsoever
        # Log a warning
        if ndim == -1:
            log.warning(
                "The shape of column '%s' is unconstrained "
                "(ndim == -1). Assuming shape is %s from "
                "the largest row",
                column,
                shape,
            )
        # Otherwise confirm the shape and ndim
        elif len(shape) != ndim:
            raise ColumnMetadataError(
                "'ndim=%d' in column descriptor doesn't "
                "match the row shapes %s" % (ndim, shape)
            )

    # Get the column schema, or create a default
    try:
        column_schema = table_schema[column]
    except KeyError:
        column_schema = {
            "dims": tuple("%s-%d" % (column, i) for i in range(1, len(shape) + 1))
        }

    try:
        dims = column_schema["dims"]
    except KeyError:
        raise ColumnMetadataError(
            f"Column schema {column_schema} does not contain required 'dims' attribute"
        )

    if not isinstance(dims, tuple) or not all(isinstance(d, str) for d in dims):
        raise ColumnMetadataError(f"Dimensions {dims} is not a tuple of strings")

    dim_chunks = []

    # Infer chunking for the dimension
    for s, d in zip(shape, dims):
        try:
            dc = chunks[d]
        except KeyError:
            # No chunk for this dimension, set to the full extent
            dim_chunks.append((s,))
        else:
            dc = da.core.normalize_chunks(dc, shape=(s,))
            dim_chunks.append(dc[0])

    if not (len(shape) == len(dims) == len(dim_chunks)):
        raise ColumnMetadataError(
            "The length of shape '%s' dims '%s' and "
            "dim_chunks '%s' do not agree." % (shape, dims, dim_chunks)
        )

    variable = ndim != "row" and not option & 4
    return ColumnMetadata(shape, dims, dim_chunks, dtype, variable)


def dim_extents_array(dim, chunks):
    """
    Produces a an array of chunk extents for a given dimension.

    Parameters
    ----------
    dim : str
        Name of the dimension
    chunks : tuple of ints
        Dimension chunks

    Returns
    -------
    dim_extents : :class:`dask.array.Array`
        dask array where each chunk contains a single (start, end) tuple
        defining the start and end of the chunk. The end is inclusive;
        :func:`daskms.casa_table.build_index` converts these extents into
        the half-open slices that arcae expects.

        The array chunks match ``chunks`` and are inaccurate, but
        are used to define chunk sizes of final outputs.

    Notes
    -----
    The returned array should never be computed directly, but
    rather used to produce dataset arrays.
    """

    name = "-".join((dim, dask.base.tokenize(dim, chunks)))
    layers = {}
    start = 0

    for i, c in enumerate(chunks):
        layers[(name, i)] = (start, start + c - 1)  # chunk end is inclusive
        start += c

    graph = HighLevelGraph.from_collections(name, layers, [])
    return da.Array(graph, name, chunks=(chunks,), dtype=object)
