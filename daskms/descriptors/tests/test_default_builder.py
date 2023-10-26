# -*- coding: utf-8 -*-

import os

import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from daskms.dataset import Variable
from daskms.descriptors.builder import (
    DefaultDescriptorBuilder,
    variable_column_descriptor,
)
from daskms.patterns import lazy_import

ct = lazy_import("casacore.tables")


@pytest.mark.parametrize(
    "chunks", [{"row": (2, 2, 2, 2, 2), "chan": (4, 4, 4, 4), "corr": (2, 2)}]
)
def test_default_plugin(tmp_path, chunks):
    filename = str(tmp_path / "test_default_plugin.table")

    def _variable_factory(dims, dtype):
        shape = tuple(sum(chunks[d]) for d in dims)
        achunks = tuple(chunks[d] for d in dims)
        dask_array = da.random.random(shape, chunks=achunks).astype(dtype)
        return Variable(dims, dask_array, {})

    variables = {
        "ANTENNA1": _variable_factory(("row",), np.int32),
        "DATA": _variable_factory(("row", "chan", "corr"), np.complex128),
        "IMAGING_WEIGHT": _variable_factory(("row", "chan"), np.float64),
    }

    builder = DefaultDescriptorBuilder()
    default_desc = builder.default_descriptor()
    tab_desc = builder.descriptor(variables, default_desc)
    dminfo = builder.dminfo(tab_desc)

    with ct.table(filename, tab_desc, dminfo=dminfo, ack=False) as T:
        T.addrows(10)
        assert set(variables.keys()) == set(T.colnames())


@pytest.mark.parametrize(
    "chunks", [{"row": (5, 5), "chan": (4, 4, 4, 4), "corr": (4,)}]
)
@pytest.mark.parametrize("dtype", [np.complex128, np.float32])
def test_variable_column_descriptor(chunks, dtype, tmp_path):
    column_meta = []
    shapes = {k: sum(c) for k, c in chunks.items()}

    # Make some visibilities
    dims = ("row", "chan", "corr")
    shape = tuple(shapes[d] for d in dims)
    data_chunks = tuple(chunks[d] for d in dims)
    data = da.random.random(shape, chunks=data_chunks).astype(dtype)
    data_var = Variable(dims, data, {})
    meta = variable_column_descriptor("DATA", data_var)
    column_meta.append({"name": "DATA", "desc": meta})

    # Make some string names
    dims = ("row",)
    shape = tuple(shapes[d] for d in dims)
    str_chunks = tuple(chunks[d] for d in dims)
    np_str_array = np.asarray(["BOB"] * shape[0], dtype=object)
    da_str_array = da.from_array(np_str_array, chunks=str_chunks)
    str_array_var = Variable(dims, da_str_array, {})
    meta = variable_column_descriptor("NAMES", str_array_var)
    column_meta.append({"name": "NAMES", "desc": meta})

    # Create a new table with the column metadata
    fn = os.path.join(str(tmp_path), "test.ms")
    tabdesc = ct.maketabdesc(column_meta)

    with ct.table(fn, tabdesc, readonly=False, ack=False) as T:
        # Add rows
        T.addrows(shapes["row"])

        str_list = np_str_array.tolist()

        # Put data
        T.putcol("DATA", data.compute())
        T.putcol("NAMES", str_list)

        # We get out what we put in
        assert_array_equal(T.getcol("NAMES"), str_list)
        assert_array_equal(T.getcol("DATA"), data)
