# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import dask.array as da
import numpy as np
import pyrap.tables as pt
import pytest

from xarrayms.dataset import Variable
from xarrayms.descriptors.ms_subtable import MSSubTablePlugin


@pytest.mark.parametrize("table", MSSubTablePlugin.SUBTABLES)
def test_ms_subtable_plugin(tmp_path, table):
    A = da.zeros((10, 20, 30), chunks=(2, 20, 30), dtype=np.int32)
    variables = {"FOO": Variable(("row", "chan", "corr"), A, {})}
    var_names = set(variables.keys())

    plugin = MSSubTablePlugin(table)
    default_desc = plugin.default_descriptor()
    tab_desc = plugin.descriptor(variables, default_desc)
    dminfo = plugin.dminfo(tab_desc)

    # These columns must always be present on an MS
    required_cols = {k for k in pt.required_ms_desc(table).keys()
                     if not k.startswith('_')}

    filename = str(tmp_path / ("%s.table" % table))

    from pprint import pprint
    pprint(tab_desc)

    with pt.table(filename, tab_desc, dminfo=dminfo, ack=False) as T:
        T.addrows(10)

        # We got required + the extra columns we asked for
        assert set(T.colnames()) == set.union(var_names, required_cols)
