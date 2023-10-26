# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from daskms.dataset import Variable
from daskms.descriptors.ms import MSDescriptorBuilder
from daskms.patterns import lazy_import

ct = lazy_import("casacore.tables")


@pytest.fixture(
    params=[
        [
            {"row": (2, 3, 2, 2), "chan": (4, 4, 4, 4), "corr": (2, 2)},
            {"row": (4, 3), "chan": (4, 4, 4, 4), "corr": (2, 2)},
        ],
    ]
)
def dataset_chunks(request):
    return request.param


def _variable_factory(dims, dtype, chunks):
    shape = tuple(sum(chunks[d]) for d in dims)
    achunks = tuple(chunks[d] for d in dims)
    dask_array = da.random.random(shape, chunks=achunks).astype(dtype)
    return Variable(dims, dask_array, {})


@pytest.fixture(
    params=[
        [("DATA", ("row", "chan", "corr"), np.complex64)],
        [
            ("DATA", ("row", "chan", "corr"), np.complex128),
            ("MODEL_DATA", ("row", "chan", "corr"), np.complex128),
        ],
        [
            ("IMAGING_WEIGHT", ("row", "chan"), np.float32),
            ("SIGMA_SPECTRUM", ("row", "chan", "corr"), float),
        ],
    ],
    ids=lambda v: f"variables={v}",
)
def column_schema(request):
    return request.param


@pytest.fixture
def variables(column_schema, dataset_chunks):
    # We want channel and correlation chunks
    # to be consistent across datasets
    assert len(set(c["chan"] for c in dataset_chunks)) == 1
    assert len(set(c["corr"] for c in dataset_chunks)) == 1

    return {
        column: [_variable_factory(dims, dtype, chunks) for chunks in dataset_chunks]
        for column, dims, dtype in column_schema
    }


@pytest.mark.parametrize("fixed", [True, False])
def test_ms_builder(tmp_path, variables, fixed):
    var_names = set(variables.keys())

    builder = MSDescriptorBuilder(fixed)
    default_desc = builder.default_descriptor()
    tab_desc = builder.descriptor(variables, default_desc)
    dminfo = builder.dminfo(tab_desc)

    # These columns must always be present on an MS
    required_cols = {k for k in ct.required_ms_desc().keys() if not k.startswith("_")}

    filename = str(tmp_path / "test_plugin.ms")

    with ct.table(filename, tab_desc, dminfo=dminfo, ack=False, nrow=10) as T:
        # We got required + the extra columns we asked for

        assert set(T.colnames()) == set.union(var_names, required_cols)

        if fixed:
            original_dminfo = {v["NAME"]: v for v in dminfo.values()}
            table_dminfo = {v["NAME"]: v for v in T.getdminfo().values()}

            for column in variables.keys():
                try:
                    column_group = table_dminfo[column + "_GROUP"]
                except KeyError:
                    raise ValueError(
                        f"{column} should be fixed but no "
                        f"Data Manager Group was created"
                    )

                assert column in column_group["COLUMNS"]
                assert column_group["TYPE"] == "TiledColumnStMan"

            assert len(original_dminfo) == len(table_dminfo)

            for dm_name, dm_group in table_dminfo.items():
                odm_group = original_dminfo[dm_name]
                assert odm_group["TYPE"] == dm_group["TYPE"]
                assert set(odm_group["COLUMNS"]) == set(dm_group["COLUMNS"])

                if dm_group["TYPE"] == "TiledColumnStMan":
                    original_tile = odm_group["SPEC"]["DEFAULTTILESHAPE"]
                    table_tile = dm_group["SPEC"]["DEFAULTTILESHAPE"]
                    assert_array_equal(original_tile, table_tile)
