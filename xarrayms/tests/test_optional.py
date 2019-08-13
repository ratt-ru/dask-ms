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
    """ Exists to illustrate how just variably shaped CASA columns can be """
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


@pytest.mark.optional
@pytest.mark.parametrize("column", ["BAZ"])
@pytest.mark.parametrize("dtype", [np.int32])
def test_variable_column_shapes(tmp_path, column, dtype):
    """ Exists to illustrate how just variably shaped CASA columns can be """
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
    """ Exists to illustrate how just variably shaped CASA columns can be """
    casa_type = infer_casa_type(dtype)

    # Column descriptor
    desc = {'desc': {'_c_order': True,
                     'comment': '%s column' % column,
                     'dataManagerGroup': '',
                     'dataManagerType': '',
                     'keywords': {},
                     # ndim can be left out, but wierd things happen
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
