# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal
import pyrap.tables as pt
import pytest

from daskms.dataset import DataArray
from daskms.descriptors.ms import MSDescriptorBuilder


@pytest.mark.parametrize("variables", [
    [("DATA", ("row", "chan", "corr"), np.complex64)],
    [("DATA", ("row", "chan", "corr"), np.complex128),
     ("MODEL_DATA", ("row", "chan", "corr"), np.complex128)],
    [("IMAGING_WEIGHT", ("row", "chan"), np.float32),
     ("SIGMA_SPECTRUM", ("row", "chan", "corr"), np.float)],
], ids=lambda v: "variables=%s" % v)
@pytest.mark.parametrize("chunks", [
    {"row": (2, 2, 2, 2, 2),
     "chan": (4, 4, 4, 4),
     "corr": (2, 2)}
])
@pytest.mark.parametrize("fixed", [
    True,
    False
])
def test_ms_builder(tmp_path, variables, chunks, fixed):
    def _variable_factory(dims, dtype):
        shape = tuple(sum(chunks[d]) for d in dims)
        achunks = tuple(chunks[d] for d in dims)
        dask_array = da.random.random(shape, chunks=achunks).astype(dtype)
        return [DataArray(dims, dask_array, {})]

    variables = {n: _variable_factory(dims, dtype)
                 for n, dims, dtype in variables}
    var_names = set(variables.keys())

    builder = MSDescriptorBuilder(fixed)
    default_desc = builder.default_descriptor()
    tab_desc = builder.descriptor(variables, default_desc)
    dminfo = builder.dminfo(tab_desc)

    # These columns must always be present on an MS
    required_cols = {k for k in pt.required_ms_desc().keys()
                     if not k.startswith('_')}

    filename = str(tmp_path / "test_plugin.ms")

    with pt.table(filename, tab_desc, dminfo=dminfo, ack=False, nrow=10) as T:
        # We got required + the extra columns we asked for
        assert set(T.colnames()) == set.union(var_names, required_cols)

        if fixed:
            original_dminfo = {v['NAME']: v for v in dminfo.values()}
            table_dminfo = {v['NAME']: v for v in T.getdminfo().values()}

            assert len(original_dminfo) == len(table_dminfo)

            for dm_name, dm_group in table_dminfo.items():
                odm_group = original_dminfo[dm_name]
                assert odm_group['TYPE'] == dm_group['TYPE']
                assert set(odm_group['COLUMNS']) == set(dm_group['COLUMNS'])

                if dm_group['TYPE'] == 'TiledColumnStMan':
                    original_tile = odm_group['SPEC']['DEFAULTTILESHAPE']
                    table_tile = dm_group['SPEC']['DEFAULTTILESHAPE']
                    assert_array_equal(original_tile, table_tile)
