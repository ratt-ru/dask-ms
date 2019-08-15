# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import dask.array as da
import numpy as np
import pyrap.tables as pt
import pytest

from xarrayms.dataset import Variable
from xarrayms.descriptors.plugin import DefaultPlugin


@pytest.mark.parametrize("chunks", [
    {"row": (2, 2, 2, 2, 2),
     "chan": (4, 4, 4, 4),
     "corr": (2, 2)}])
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

    plugin = DefaultPlugin()
    default_desc = plugin.default_descriptor()
    tab_desc = plugin.descriptor(variables, default_desc)
    dminfo = plugin.dminfo(tab_desc)

    with pt.table(filename, tab_desc, dminfo=dminfo, ack=False) as T:
        T.addrows(10)
        assert set(variables.keys()) == set(T.colnames())
