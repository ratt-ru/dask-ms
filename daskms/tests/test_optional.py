# -*- coding: utf-8 -*-

"""
Optional test cases that illustrate
CASA Table dimension and shape constraints.

Pass the --optional command line argument to py.test to enable them

$ py.test --optional
"""

import os
from pprint import pprint  # noqa

import numpy as np
from numpy.testing import assert_array_equal
import pyrap.tables as pt
import pytest

from daskms.columns import infer_casa_type


@pytest.mark.optional
@pytest.mark.parametrize("column", ["BAZ"])
@pytest.mark.parametrize("dtype", [np.int32])
def test_variable_column_dimensions(tmp_path, column, dtype):
    """ ndim set to -1, we can put anything in the column """
    casa_type = infer_casa_type(dtype)

    # Column descriptor
    desc = {'desc': {'_c_order': True,
                     'comment': f'{column} column',
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
                     'comment': f'{column} column',
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
                     'comment': f'{column} column',
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
                     'comment': f'{column} column',
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
                     'comment': f'{column} column',
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


@pytest.mark.optional
@pytest.mark.parametrize("column", ["BAZ"])
@pytest.mark.parametrize("row", [10])
@pytest.mark.parametrize("shape", [(16, 4)])
@pytest.mark.parametrize("dtype", [np.int32])
def test_tiledstman(tmp_path, column, row, shape, dtype):
    casa_type = infer_casa_type(dtype)

    # Column descriptor
    desc = {'desc': {'_c_order': True,
                     'comment': f'{column} column',
                     'dataManagerGroup': 'BAZ-GROUP',
                     'dataManagerType': 'TiledColumnStMan',
                     'keywords': {},
                     'maxlen': 0,
                     'ndim': len(shape),
                     'option': 0,
                     'shape': shape,
                     'valueType': casa_type},
            'name': column}

    table_desc = pt.maketabdesc([desc])

    tile_shape = tuple(reversed(shape)) + (row,)

    dminfo = {'*1': {
        'NAME': 'BAZ-GROUP',
        'TYPE': 'TiledColumnStMan',
        'SPEC': {'DEFAULTTILESHAPE': tile_shape},
        'COLUMNS': ['BAZ'],
    }
    }

    fn = os.path.join(str(tmp_path), "test.table")

    with pt.table(fn, table_desc, dminfo=dminfo, nrow=10, ack=False) as T:
        dmg = T.getdminfo()['*1']
        assert dmg['NAME'] == 'BAZ-GROUP'
        assert_array_equal(dmg['SPEC']['DEFAULTTILESHAPE'], tile_shape)
        assert dmg['TYPE'] == 'TiledColumnStMan'
        assert dmg['COLUMNS'] == ['BAZ']


@pytest.mark.optional
@pytest.mark.parametrize("column", ["BAZ"])
@pytest.mark.parametrize("row", [10])
@pytest.mark.parametrize("shape", [(16, 4)])
@pytest.mark.parametrize("dtype", [np.int32])
def test_tiledstman_addcols(tmp_path, column, row, shape, dtype):
    """ ndim set to zero implies scalars """
    casa_type = infer_casa_type(dtype)

    # Column descriptor
    desc = {'desc': {'_c_order': True,
                     'comment': f'{column} column',
                     'dataManagerGroup': 'BAZ_GROUP',
                     'dataManagerType': 'TiledColumnStMan',
                     'keywords': {},
                     'maxlen': 0,
                     'ndim': len(shape),
                     'option': 0,
                     'shape': shape,
                     'valueType': casa_type},
            'name': column}

    table_desc = pt.maketabdesc([desc])

    tile_shape = tuple(reversed(shape)) + (row,)

    dminfo = {'*1': {
        'NAME': 'BAZ_GROUP',
        'TYPE': 'TiledColumnStMan',
        'SPEC': {'DEFAULTTILESHAPE': tile_shape},
        'COLUMNS': ['BAZ'],
    }
    }

    fn = os.path.join(str(tmp_path), "test.table")

    with pt.table(fn, table_desc, dminfo=dminfo, nrow=10, ack=False) as T:
        # Add a new FRED column
        desc = {'FRED': {
            'dataManagerGroup': 'FRED_GROUP',
            'dataManagerType': 'TiledColumnStMan',
            'ndim': len(shape),
            'shape': shape,
            'valueType': casa_type
        }}

        dminfo = {'*1': {
            'NAME': 'FRED_GROUP',
            'TYPE': 'TiledColumnStMan',
            'SPEC': {'DEFAULTTILESHAPE': tile_shape},
            'COLUMNS': ['FRED'],
        }
        }

        T.addcols(desc, dminfo=dminfo)

        # Trying to add a new QUX column by redefining FRED_GROUP fails
        desc = {'QUX': {
            'dataManagerGroup': 'FRED_GROUP',
            'dataManagerType': 'TiledColumnStMan',
            'ndim': len(shape),
            'shape': shape,
            'valueType': casa_type
        }}

        dminfo = {'*1': {
            'NAME': 'FRED_GROUP',
            'TYPE': 'TiledColumnStMan',
            'SPEC': {'DEFAULTTILESHAPE': tile_shape},
            'COLUMNS': ['FRED', 'QUX'],
        }
        }

        with pytest.raises(RuntimeError, match="Data manager name FRED_GROUP"):
            T.addcols(desc, dminfo=dminfo)

        groups = {g['NAME']: g for g in T.getdminfo().values()}
        assert set(["BAZ_GROUP", "FRED_GROUP"]) == set(groups.keys())

        # Adding new QUX column succeeds, but can't
        # add columns to an existing TiledColumnStMan?
        # casacore creates a new group, FRED_GROUP_1
        T.addcols(desc)

        groups = {g['NAME']: g for g in T.getdminfo().values()}
        assert set(["BAZ_GROUP", "FRED_GROUP",
                    "FRED_GROUP_1"]) == set(groups.keys())

        # Add ACK and BAR to the ACKBAR_GROUP at the same time succeeds
        desc = {
            "ACK": {
                'dataManagerGroup': 'ACKBAR_GROUP',
                'dataManagerType': 'TiledColumnStMan',
                'ndim': len(shape),
                'shape': shape,
                'valueType': casa_type},
            "BAR": {
                'dataManagerGroup': 'ACKBAR_GROUP',
                'dataManagerType': 'TiledColumnStMan',
                'ndim': len(shape),
                'shape': shape,
                'valueType': casa_type},
        }

        dminfo = {'*1': {
            'NAME': 'ACKBAR_GROUP',
            'TYPE': 'TiledColumnStMan',
            'SPEC': {'DEFAULTTILESHAPE': tile_shape},
            'COLUMNS': ['ACK', 'BAR'],
        }
        }

        T.addcols(desc, dminfo=dminfo)

        groups = {g['NAME']: g for g in T.getdminfo().values()}
        assert set(["BAZ_GROUP", "FRED_GROUP",
                    "FRED_GROUP_1", "ACKBAR_GROUP"]) == set(groups.keys())

        assert set(groups["ACKBAR_GROUP"]['COLUMNS']) == set(["ACK", "BAR"])
