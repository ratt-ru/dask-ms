# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import dask.array as da
import numpy as np
import pyrap.tables as pt
import pytest

from xarrayms.dataset import Variable
from xarrayms.descriptors.ms_subtable import MSSubTableDescriptorBuilder


@pytest.mark.parametrize("table", MSSubTableDescriptorBuilder.SUBTABLES)
def test_ms_subtable_builder(tmp_path, table):
    A = da.zeros((10, 20, 30), chunks=(2, 20, 30), dtype=np.int32)
    variables = {"FOO": Variable(("row", "chan", "corr"), A, {})}
    var_names = set(variables.keys())

    builder = MSSubTableDescriptorBuilder(table)
    default_desc = builder.default_descriptor()
    tab_desc = builder.descriptor(variables, default_desc)
    dminfo = builder.dminfo(tab_desc)

    # These columns must always be present on an MS
    required_cols = {k for k in pt.required_ms_desc(table).keys()
                     if not k.startswith('_')}

    filename = str(tmp_path / ("%s.table" % table))

    with pt.table(filename, tab_desc, dminfo=dminfo, ack=False) as T:
        T.addrows(10)

        # We got required + the extra columns we asked for
        assert set(T.colnames()) == set.union(var_names, required_cols)
