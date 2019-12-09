# -*- coding: utf-8 -*-

import numpy as np
import pyrap.tables as pt
import pytest

from daskms.columns import (infer_dtype, _TABLE_TO_PY,
                            infer_casa_type, _PY_TO_TABLE,
                            column_metadata)
from daskms.table_proxy import TableProxy
from daskms.utils import assert_liveness


@pytest.mark.parametrize("casa_type, numpy_type", list(_TABLE_TO_PY.items()))
def test_infer_dtype(casa_type, numpy_type):
    assert infer_dtype('col', {'valueType': casa_type}) == numpy_type


@pytest.mark.parametrize("numpy_type, casa_type", list(_PY_TO_TABLE.items()))
def test_infer_casa_type(numpy_type, casa_type):
    assert infer_casa_type(numpy_type) == casa_type


def test_missing_casa_type():
    with pytest.raises(ValueError, match="No known conversion"):
        infer_dtype('col', {'valueType': 'qux'})


def test_missing_valuetype():
    with pytest.raises(ValueError, match="Cannot infer dtype"):
        infer_dtype('col', {})


@pytest.mark.parametrize("column, shape, dtype", [
    ("DATA", (16, 4), np.complex128),
    ("TIME",  (), np.float64),
    ("ANTENNA1", (), np.int32)])
@pytest.mark.parametrize("table_schema", [
    {'DATA': {'dims': ("chan", "corr")}}
])
@pytest.mark.parametrize("chunks", [
    (("chan", (12, 4)), ("corr", (1, 1, 1, 1))),
    (("chan", (4, 4, 4, 4)), ("corr", (2, 2)))])
def test_column_metadata(ms, column, shape, chunks, table_schema, dtype):
    table_proxy = TableProxy(pt.table, ms, readonly=True, ack=False)
    assert_liveness(1, 1)

    try:
        dims = table_schema[column]['dims']
    except KeyError:
        dims = tuple("%s-%d" % (column, i) for i in range(1, len(shape) + 1))

    meta = column_metadata(column, table_proxy, table_schema, dict(chunks))

    assert meta.shape == shape
    assert meta.dims == dims
    assert meta.chunks == [c[1] for c in chunks[:len(meta.shape)]]
    assert meta.dtype == dtype

    del table_proxy
    assert_liveness(0, 0)
