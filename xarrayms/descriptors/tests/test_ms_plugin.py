# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pyrap.tables as pt
import pytest

from xarrayms.descriptors.ms import MeasurementSetPlugin


@pytest.mark.parametrize("variables", [
    ["DATA"],
    ["DATA", "MODEL_DATA"],
    ["IMAGING_WEIGHT", "SIGMA_SPECTRUM"]
], ids=lambda v: "variables=%s" % v)
def test_ms_plugin(tmp_path, variables):
    var_names = set(variables)
    variables = {v: None for v in variables}

    plugin = MeasurementSetPlugin()
    default_desc = plugin.default_descriptor()
    tab_desc = plugin.descriptor(variables, default_desc)
    dminfo = plugin.dminfo(tab_desc)

    # These columns must always be present on an MS
    required_cols = {k for k in pt.required_ms_desc().keys()
                     if not k.startswith('_')}

    filename = str(tmp_path / "test_plugin.ms")

    with pt.table(filename, tab_desc, dminfo=dminfo, ack=False) as T:
        T.addrows(10)

        # We got required + the extra columns we asked for
        assert set(T.colnames()) == set.union(var_names, required_cols)
