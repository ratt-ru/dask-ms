# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import pytest

from daskms.dataset import Variable
from daskms.descriptors.ms_subtable import MSSubTableDescriptorBuilder
from daskms.patterns import lazy_import
from daskms.table_schemas import SUBTABLES

ct = lazy_import("casacore.tables")


@pytest.mark.parametrize("table", SUBTABLES)
def test_ms_subtable_builder(tmp_path, table):
    A = da.zeros((10, 20, 30), chunks=(2, 20, 30), dtype=np.int32)
    variables = {"FOO": Variable(("row", "chan", "corr"), A, {})}
    var_names = set(variables.keys())

    builder = MSSubTableDescriptorBuilder(table)
    default_desc = builder.default_descriptor()
    tab_desc = builder.descriptor(variables, default_desc)
    dminfo = builder.dminfo(tab_desc)

    # These columns must always be present on an MS
    required_cols = {
        k for k in ct.required_ms_desc(table).keys() if not k.startswith("_")
    }

    filename = str(tmp_path / (f"{table}.table"))

    with ct.table(filename, tab_desc, dminfo=dminfo, ack=False) as T:
        T.addrows(10)

        # We got required + the extra columns we asked for
        assert set(T.colnames()) == set.union(var_names, required_cols)
