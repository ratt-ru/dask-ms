# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pformat
import sys

import numpy as np

# Map  column string types to numpy/python types
_TABLE_TO_PY_TABLE = {
    'INT': np.int32,
    'FLOAT': np.float32,
    'DOUBLE': np.float64,
    'BOOLEAN': np.bool,
    'COMPLEX': np.complex64,
    'DCOMPLEX': np.complex128,
    'STRING': object,
}


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
        return _TABLE_TO_PY_TABLE[value_type.upper()]
    except KeyError:
        raise ValueError("No known conversion from CASA Table type '%s' "
                         "to python/numpy type. "
                         "Perhaps it needs to be added "
                         "to _TABLE_TO_PY_TABLE?:\n"
                         "%s" % (value_type, pformat(_TABLE_TO_PY_TABLE)))


class ColumnMetadataError(Exception):
    pass


def column_metadata(table_proxy, table_schema, column, exemplar_row=0):
    """
    Infers column configuration the following:

        1. column shape
        2. data type

    Shape is determined via the following three strategies,
    and in order of robustness:

        1. If coldesc['option'] & 4 (FixedShape) is True, use coldesc['shape']
        2. An exemplar row is read using getcol(column, exemplar_row, 1)
           to determine shape and dtype

    Parameters
    ----------
    table_proxy : string
        CASA Table path
    table_schema : dict
        Table schema
    column : string
        Table column
    exemplar_row : int, optional
        Table row accessed when inferring a shape and dtype
        from a getcol.


    Returns
    -------
    shape : tuple
        Shape of column (excluding the row dimension)
    dims : tuple
        Dask dimension schema.
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
            exemplar = table_proxy.getcol(column, exemplar_row, 1).result()
        except BaseException as e:
            ve = ColumnMetadataError("Unable to infer shape of "
                                     "column '%s' due to:\n'%s'"
                                     % (column, e))

            raise (ve, None, sys.exc_info()[2])

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

        if len(shape) - 1 != ndim:
            raise ColumnMetadataError("'ndim=%d' in column descriptor doesn't "
                                      "match shape of exemplar=%d" %
                                      (ndim, len(shape) - 1))

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

    return shape, dims, dtype
