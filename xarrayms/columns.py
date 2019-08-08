# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict, namedtuple
from pprint import pformat

import dask.array as da
import numpy as np

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


# Map python/numpy types back to column types
# If the column type is multiply defined, OrderedDict will
# give us the last one in _TABLE_TO_PY
_PY_TO_TABLE = OrderedDict((v, k) for k, v in _TABLE_TO_PY.items())


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
                            ["shape", "dims", "dim_chunks", "dtype"])


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
    ndim = coldesc.get('ndim', 0)

    try:
        option = coldesc['option']
    except KeyError:
        raise ColumnMetadataError("Column '%s' has no option "
                                  "in the column descriptor" % column)

    # This seems to imply no other dimensions beyond row
    if ndim == 0:
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
            exemplar = table_proxy.getcol(column, exemplar_row,
                                          nrow=1).result()
        except Exception as e:
            raise ColumnMetadataError("Unable to infer shape of "
                                      "column '%s' due to:\n'%s'"
                                      % (column, str(e)))

        if isinstance(exemplar, np.ndarray):
            shape = exemplar.shape[1:]

            # Double-check the dtype
            if dtype != exemplar.dtype:
                raise ColumnMetadataError("Inferred dtype '%s' does not match "
                                          "the exemplar dtype '%s'" %
                                          (dtype, exemplar.dtype))
        elif isinstance(exemplar, list):
            shape = (len(exemplar),)
            assert dtype == object
        else:
            raise ColumnMetadataError("Unhandled exemplar "
                                      "type '%s'" % type(exemplar))

        if len(shape) != ndim:
            raise ColumnMetadataError("'ndim=%d' in column descriptor doesn't "
                                      "match shape of exemplar=%s" %
                                      (ndim, shape))

    # Extract dimension schema
    try:
        column_schema = table_schema[column]
    except KeyError:
        dims = ()
    else:
        try:
            dask_schema = column_schema['dask']
        except KeyError:
            dims = ()
        else:
            try:
                dims = dask_schema['dims']
            except KeyError:
                dims = ()

    dim_chunks = []

    # Infer chunking for the dimension
    for s, d in zip(shape, dims):
        try:
            dc = chunks[d]
        except KeyError:
            dim_chunks.append((s,))
        else:
            dc = da.core.normalize_chunks(dc, shape=(s,))
            dim_chunks.append(dc[0])

    if not (len(shape) == len(dims) == len(dim_chunks)):
        raise ColumnMetadataError("The length of shape '%s' dims '%s' and "
                                  "dim_chunks '%s' do not agree." %
                                  (shape, dims, dim_chunks))

    return ColumnMetadata(shape, dims, dim_chunks, dtype)
