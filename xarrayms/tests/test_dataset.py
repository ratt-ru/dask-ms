# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask
import numpy as np
import pytest

from xarrayms.dataset import dataset, write_columns
from xarrayms.utils import group_cols_str, index_cols_str, assert_liveness


@pytest.mark.parametrize("group_cols", [
    ["FIELD_ID", "SCAN_NUMBER"]],
    ids=group_cols_str)
@pytest.mark.parametrize("index_cols", [
    ["TIME", "ANTENNA1", "ANTENNA2"]],
    ids=index_cols_str)
@pytest.mark.parametrize("chunks", [{"rows": 2}])
def test_dataset(ms, group_cols, index_cols, chunks):
    datasets = dataset(ms, None, group_cols, index_cols, chunks)
    assert_liveness(2, 1)

    for ds in datasets:
        dask.compute(ds.variables)

    del ds
    del datasets
    assert_liveness(0, 0)
