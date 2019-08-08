# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal
import pyrap.tables as pt
import pytest

from xarrayms.dataset import Variable
from xarrayms.columns import (infer_dtype, _TABLE_TO_PY,
                              infer_casa_type, _PY_TO_TABLE,
                              column_metadata, dask_column_metadata)
from xarrayms.table_proxy import TableProxy
from xarrayms.utils import assert_liveness


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
    {'DATA': {'dask': {'dims': ("chan", "corr")}}}
])
@pytest.mark.parametrize("chunks", [
    (("chan", (12, 4)), ("corr", (1, 1, 1, 1))),
    (("chan", (4, 4, 4, 4)), ("corr", (2, 2)))])
def test_column_metadata(ms, column, shape, chunks, table_schema, dtype):
    table_proxy = TableProxy(pt.table, ms, readonly=True, ack=False)
    assert_liveness(1, 1)

    try:
        dims = table_schema[column]['dask']['dims']
    except KeyError:
        dims = tuple("%s-%d" % (column, i) for i in range(1, len(shape) + 1))

    meta = column_metadata(column, table_proxy, table_schema, dict(chunks))

    assert meta.shape == shape
    assert meta.dims == dims
    assert meta.chunks == [c[1] for c in chunks[:len(meta.shape)]]
    assert meta.dtype == dtype
    assert "__coldesc__" in meta.attrs

    del table_proxy
    assert_liveness(0, 0)


@pytest.mark.parametrize("chunks", [{'row': (5, 5),
                                     'chan': (4, 4, 4, 4),
                                     'corr': (4,)}])
@pytest.mark.parametrize("dtype", [np.complex128])
def test_dask_column_metadata(chunks, dtype, tmp_path):
    column_meta = []
    shapes = {k: sum(c) for k, c in chunks.items()}

    # Make some visibilities
    dims = ("row", "chan", "corr")
    shape = tuple(shapes[d] for d in dims)
    data_chunks = tuple(chunks[d] for d in dims)
    data = da.random.random(shape, chunks=data_chunks).astype(dtype)
    data_var = Variable(dims, data, {})
    meta = dask_column_metadata("DATA", data_var)
    assert_array_equal(meta.shape, shape[1:])
    assert_array_equal(data_chunks[1:], data.chunks[1:])
    assert meta.dtype == data.dtype
    column_meta.append(meta)

    # Make some string names
    dims = ("row",)
    shape = tuple(shapes[d] for d in dims)
    str_chunks = tuple(chunks[d] for d in dims)
    np_str_array = np.asarray(["BOB"] * shape[0], dtype=np.object)
    da_str_array = da.from_array(np_str_array, chunks=str_chunks)
    str_array_var = (dims, da_str_array, {})
    meta = dask_column_metadata("NAMES", str_array_var)
    assert_array_equal(meta.shape, shape[1:])
    assert_array_equal(str_chunks[1:], da_str_array.chunks[1:])
    assert meta.dtype == da_str_array.dtype
    column_meta.append(meta)

    # Create a new table with the column metadata
    fn = os.path.join(str(tmp_path), "test.ms")
    tabdesc = pt.maketabdesc([m.attrs["__coldesc__"] for m in column_meta])

    with pt.table(fn, tabdesc, readonly=False, ack=False) as T:
        # Add rows
        T.addrows(shapes['row'])

        str_list = np_str_array.tolist()

        # Put data
        T.putcol("DATA", data.compute())
        T.putcol("NAMES", str_list)

        # We get out what we put in
        assert_array_equal(T.getcol("NAMES"), str_list)
        assert_array_equal(T.getcol("DATA"), data)
