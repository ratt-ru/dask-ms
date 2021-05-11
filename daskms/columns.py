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
_TABLE_TO_PY = OrderedDict({
    'BOOL': 'bool',
    'BOOLEAN': 'bool',
    'BYTE': 'uint8',
    'UCHAR': 'uint8',
    'SMALLINT': 'int16',
    'SHORT': 'int16',
    'USMALLINT': 'uint16',
    'USHORT': 'uint16',
    'INT': 'int32',
    'INTEGER': 'int32',
    'UINTEGER': 'uint32',
    'UINT': 'uint32',
    'FLOAT': 'float32',
    'DOUBLE': 'float64',
    'FCOMPLEX': 'complex64',
    'COMPLEX': 'complex64',
    'DCOMPLEX': 'complex128',
    'STRING': 'object',
})


# Map numpy/python types to column string types
_PY_TO_TABLE = OrderedDict({
    'bool': 'BOOLEAN',
    'uint8': 'UCHAR',
    'int16': 'SHORT',
    'uint16': 'USHORT',
    'uint32': 'UINT',
    'int32': 'INTEGER',
    'float32': 'FLOAT',
    'float64': 'DOUBLE',
    'complex64': 'COMPLEX',
    'complex128': 'DCOMPLEX',
    'object': 'STRING'
})


def infer_dtype(column, coldesc):
    # Extract valueType
    try:
        value_type = coldesc['valueType']
    except KeyError:
        raise ValueError("Cannot infer dtype for column '%s'. "
                         "Table Column Description is missing "
                         "valueType. Description is '%s'" %
                         (column, coldesc))

    # Try conversion to numpy/python type
    try:
        np_type_str = _TABLE_TO_PY[value_type.upper()]
    except KeyError:
        raise ValueError("No known conversion from CASA Table type '%s' "
                         "to python/numpy type. "
                         "Perhaps it needs to be added "
                         "to _TABLE_TO_PY?:\n"
                         "%s" % (value_type, pformat(dict(_TABLE_TO_PY))))
    else:
        return np.dtype(np_type_str)


def infer_casa_type(dtype):
    try:
        return _PY_TO_TABLE[np.dtype(dtype).name]
    except KeyError:
        raise ValueError("No known conversion from numpy dtype '%s' "
                         "to CASA Table Type. "
                         "Perhaps it needs to be added "
                         "to _TABLE_TO_PY?:\n"
                         "%s" % (dtype, pformat(dict(_TABLE_TO_PY))))


class ColumnMetadataError(Exception):
    pass


ColumnMetadata = namedtuple("ColumnMetadata",
                            ["shape", "dims", "chunks", "dtype"])


def column_metadata(column, table_proxy, table_schema, chunks, exemplar_row=0):
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
        Table row accessed when inferring a shape and dtype
        from a getcol.


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


    Raises
    ------
    ColumnMetadataError
        Raised if inferring metadata failed.
    """
    coldesc = table_proxy.getcoldesc(column).result()
    dtype = infer_dtype(column, coldesc)
    # missing ndim implies only row dimension
    ndim = coldesc.get('ndim', 'row')

    try:
        option = coldesc['option']
    except KeyError:
        raise ColumnMetadataError("Column '%s' has no option "
                                  "in the column descriptor" % column)

    # Each row is a scalar
    # TODO(sjperkins)
    # Probably could be handled by getCell/putCell calls,
    # but the effort may not be worth it
    if ndim == 0:
        raise ColumnMetadataError("Scalars in column '%s' "
                                  "(ndim == %d) are not currently handled"
                                  % (column, ndim))
    # Only row dimensions
    elif ndim == 'row':
        shape = ()
    # FixedShape
    elif option & 4:
        try:
            shape = tuple(coldesc['shape'])
        except KeyError:
            raise ColumnMetadataError("'%s' column descriptor option '%d' "
                                      "specifies a FixedShape but no 'shape' "
                                      "attribute was found in the "
                                      "column descriptor" % (column, option))
    # Variably shaped...
    else:
        try:
            # Get an exemplar row and infer the shape
            exemplar = table_proxy.getcell(column, exemplar_row).result()
        except Exception as e:
            raise ColumnMetadataError("Unable to infer shape of "
                                      "column '%s' due to:\n'%s'"
                                      % (column, str(e)))

        # Try figure out the shape
        if isinstance(exemplar, np.ndarray):
            shape = exemplar.shape

            # Double-check the dtype
            if dtype != exemplar.dtype:
                raise ColumnMetadataError("Inferred dtype '%s' does not match "
                                          "the exemplar dtype '%s'" %
                                          (dtype, exemplar.dtype))
        elif isinstance(exemplar, list):
            shape = (len(exemplar),)
            assert dtype == object
        else:
            raise ColumnMetadataError(f"Unhandled exemplar type "
                                      f"'{type(exemplar)}'")

        # NOTE(sjperkins)
        # -1 implies each row can be any shape whatsoever
        # Log a warning
        if ndim == -1:
            log.warning("The shape of column '%s' is unconstrained "
                        "(ndim == -1). Assuming shape is %s from "
                        "exemplar", column, shape)
        # Otherwise confirm the shape and ndim
        elif len(shape) != ndim:
            raise ColumnMetadataError("'ndim=%d' in column descriptor doesn't "
                                      "match shape of exemplar=%s" %
                                      (ndim, shape))

    # Extract dimension schema
    try:
        dims = table_schema[column]['dims']
    except KeyError:
        dims = tuple("%s-%d" % (column, i) for i in range(1, len(shape) + 1))

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
        raise ColumnMetadataError("The length of shape '%s' dims '%s' and "
                                  "dim_chunks '%s' do not agree." %
                                  (shape, dims, dim_chunks))

    return ColumnMetadata(shape, dims, dim_chunks, dtype)


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
        defining the start and end of the chunk. The end is inclusive
        in the python-casacore style.

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
