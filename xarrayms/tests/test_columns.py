# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pyrap.tables as pt
import pytest

from xarrayms.table_proxy import TableProxy
from xarrayms.columns import infer_dtype, column_metadata
from xarrayms.utils import assert_liveness


@pytest.mark.parametrize("casa_type, numpy_type", [
    ('int', np.int32),
    ('float', np.float32),
    ('double', np.float64),
    ('boolean', np.bool),
    ('complex', np.complex64),
    ('dcomplex', np.complex128),
    ('string', object)])
def test_infer_dtype(casa_type, numpy_type):
    assert infer_dtype('col', {'valueType': casa_type}) == numpy_type


def test_missing_casa_type():
    with pytest.raises(ValueError, match="No known conversion"):
        infer_dtype('col', {'valueType': 'qux'})


def test_missing_valuetype():
    with pytest.raises(ValueError, match="Cannot infer dtype"):
        infer_dtype('col', {})


@pytest.mark.parametrize("column, dims, shape, dtype", [
    ("DATA", ("chan", "corr"), (16, 4), np.complex128),
    ("TIME", (), (), np.float64),
    ("ANTENNA1", (), (), np.int32)])
def test_column_metadata(ms, column, dims, shape, dtype):
    table_proxy = TableProxy(pt.table, ms, readonly=True, ack=False)
    assert_liveness(1, 1)

    table_schema = {'DATA': {'dask': {'dims': dims}}}

    ishape, idims, idtype = column_metadata(table_proxy, table_schema, column)
    assert ishape == shape
    assert idims == dims
    assert idtype == dtype

    del table_proxy
    assert_liveness(0, 0)
