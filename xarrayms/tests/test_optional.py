# -*- coding: utf-8 -*-

"""
Optional test cases that illustrate
CASA Table dimension and shape constraints.

Pass the --optional command line argument to py.test to enable them

$ py.test --optional
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pyrap.tables as pt
import pytest

from xarrayms.columns import infer_casa_type


@pytest.mark.optional
@pytest.mark.parametrize("column", ["BAZ"])
@pytest.mark.parametrize("dtype", [np.int32])
def test_variable_column_dimensions(tmp_path, column, dtype):
    """ ndim set to -1, we can put anything in the column """
    casa_type = infer_casa_type(dtype)

    # Column descriptor
    desc = {'desc': {'_c_order': True,
                     'comment': '%s column' % column,
                     'dataManagerGroup': '',
                     'dataManagerType': '',
                     'keywords': {},
                     # This allows any shape to go in
                     'ndim': -1,
                     'maxlen': 0,
                     'option': 0,
                     'valueType': casa_type},
            'name': column}

    table_desc = pt.maketabdesc([desc])
    fn = os.path.join(str(tmp_path), "test.table")

    with pt.table(fn, table_desc, nrow=10, ack=False) as T:
        # Put 10 rows into the table
        T.putcol(column, np.zeros((10, 20, 30), dtype=dtype))
        assert T.getcol(column).shape == (10, 20, 30)

        # Put something differently shaped in the first 5 rows
        T.putcol(column, np.zeros((5, 40), dtype=dtype), startrow=0, nrow=5)
        assert T.getcol(column, startrow=0, nrow=5).shape == (5, 40)

        # The last 5 rows have the original shape
        assert T.getcol(column, startrow=5, nrow=5).shape == (5, 20, 30)

        # We can even put a scalar in
        T.putcell(column, 8, 3)
        assert T.getcell(column, 8) == 3


@pytest.mark.optional
@pytest.mark.parametrize("column", ["BAZ"])
@pytest.mark.parametrize("dtype", [np.int32])
def test_variable_column_shapes(tmp_path, column, dtype):
    """ ndim set to 2, but shapes are variable """
    casa_type = infer_casa_type(dtype)

    # Column descriptor
    desc = {'desc': {'_c_order': True,
                     'comment': '%s column' % column,
                     'dataManagerGroup': '',
                     'dataManagerType': '',
                     'keywords': {},
                     'ndim': 2,
                     'maxlen': 0,
                     'option': 0,
                     'valueType': casa_type},
            'name': column}

    table_desc = pt.maketabdesc([desc])
    fn = os.path.join(str(tmp_path), "test.table")

    with pt.table(fn, table_desc, nrow=10, ack=False) as T:
        # Put 10 rows into the table
        T.putcol(column, np.zeros((10, 20, 30), dtype=dtype))
        assert T.getcol(column).shape == (10, 20, 30)

        # Must be ndim == 2
        err_str = "Table array conformance error"
        with pytest.raises(RuntimeError, match=err_str):
            T.putcol(column, np.zeros((5, 40), dtype=dtype))

        # Put something differently shaped in the first 5 rows
        T.putcol(column, np.zeros((5, 40, 30), dtype=dtype),
                 startrow=0, nrow=5)
        assert T.getcol(column, startrow=0, nrow=5).shape == (5, 40, 30)

        # The last 5 rows have the original shape
        assert T.getcol(column, startrow=5, nrow=5).shape == (5, 20, 30)


@pytest.mark.optional
@pytest.mark.parametrize("column", ["BAZ"])
@pytest.mark.parametrize("dtype", [np.int32])
def test_fixed_column_shapes(tmp_path, column, dtype):
    """ Fixed column, shape and ndim must be supplied """
    casa_type = infer_casa_type(dtype)

    # Column descriptor
    desc = {'desc': {'_c_order': True,
                     'comment': '%s column' % column,
                     'dataManagerGroup': '',
                     'dataManagerType': '',
                     'keywords': {},
                     'ndim': 2,
                     'shape': (20, 30),
                     'maxlen': 0,
                     'option': 0,
                     'valueType': casa_type},
            'name': column}

    table_desc = pt.maketabdesc([desc])
    fn = os.path.join(str(tmp_path), "test.table")

    with pt.table(fn, table_desc, nrow=10, ack=False) as T:
        # Put 10 rows into the table
        T.putcol(column, np.zeros((10, 20, 30), dtype=dtype))
        assert T.getcol(column).shape == (10, 20, 30)

        # Must be ndim == 2
        err_str = "Table array conformance error"
        with pytest.raises(RuntimeError, match=err_str):
            T.putcol(column, np.zeros((5, 40), dtype=dtype))

        # shape != (20, 30)
        with pytest.raises(RuntimeError, match=err_str):
            T.putcol(column, np.zeros((5, 40, 30), dtype=dtype),
                     startrow=0, nrow=5)


@pytest.mark.optional
@pytest.mark.parametrize("column", ["BAZ"])
@pytest.mark.parametrize("dtype", [np.int32])
def test_only_row_shape(tmp_path, column, dtype):
    """ Missing ndim implies row only! """
    casa_type = infer_casa_type(dtype)

    # Column descriptor
    desc = {'desc': {'_c_order': True,
                     'comment': '%s column' % column,
                     'dataManagerGroup': '',
                     'dataManagerType': '',
                     'keywords': {},
                     'maxlen': 0,
                     'option': 0,
                     'valueType': casa_type},
            'name': column}

    table_desc = pt.maketabdesc([desc])
    fn = os.path.join(str(tmp_path), "test.table")

    with pt.table(fn, table_desc, nrow=10, ack=False) as T:
        # Put 10 rows into the table
        T.putcol(column, np.zeros(10, dtype=dtype))
        assert T.getcol(column).shape == (10,)

        # Must be ndim == 2
        err_str = 'Vector<T>: ndim of other array > 1 ndim 1 differs from 2'
        with pytest.raises(RuntimeError, match=err_str):
            T.putcol(column, np.zeros((5, 40), dtype=dtype))

        # shape != (20, 30)
        err_str = 'Vector<T>: ndim of other array > 1 ndim 1 differs from 3'
        with pytest.raises(RuntimeError, match=err_str):
            T.putcol(column, np.zeros((5, 40, 30), dtype=dtype),
                     startrow=0, nrow=5)


@pytest.mark.optional
@pytest.mark.parametrize("column", ["BAZ"])
@pytest.mark.parametrize("dtype", [np.int32])
def test_scalar_ndim(tmp_path, column, dtype):
    """ ndim set to zero implies scalars """
    casa_type = infer_casa_type(dtype)

    # Column descriptor
    desc = {'desc': {'_c_order': True,
                     'comment': '%s column' % column,
                     'dataManagerGroup': '',
                     'dataManagerType': '',
                     'keywords': {},
                     'maxlen': 0,
                     'ndim': 0,
                     'option': 0,
                     'valueType': casa_type},
            'name': column}

    table_desc = pt.maketabdesc([desc])
    fn = os.path.join(str(tmp_path), "test.table")

    with pt.table(fn, table_desc, nrow=10, ack=False) as T:
        for r in range(10):
            T.putcell(column, r, r)

        for r in range(10):
            assert T.getcell(column, r) == r
