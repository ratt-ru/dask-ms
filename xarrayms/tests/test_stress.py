# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask
import pytest


from xarrayms.dataset import dataset, write_columns


@pytest.mark.stress
@pytest.mark.parametrize("big_ms", [1000], indirect=True)
@pytest.mark.parametrize("iteration", range(10))
def test_stress(big_ms, iteration):
    datasets = dataset(big_ms, ["TIME", "DATA"],
                       ["FIELD_ID", "DATA_DESC_ID"], [],
                       {"row": 10})

    assert len(datasets) == 1
    ds = datasets[0]

    writes = write_columns(big_ms, ds, ["TIME", "DATA"])

    dask.compute(writes)
