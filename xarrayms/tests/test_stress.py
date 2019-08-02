# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask
import pytest


from xarrayms.dataset import dataset, write_columns


@pytest.mark.stress
@pytest.mark.parametrize("big_ms", [1000], indirect=True)
@pytest.mark.parametrize("iterations", [10])
def test_stress(big_ms, iterations):
    datasets = dataset(big_ms, ["TIME", "DATA"],
                       ["FIELD_ID", "DATA_DESC_ID"], [],
                       {"row": 10})

    assert len(datasets) == 1
    ds = datasets[0]

    writes = []

    for i in range(iterations):
        time = ds.TIME + i
        data = ds.DATA + i

        nds = ds.assign(TIME=(ds.TIME + i, ("row",)),
                        DATA=(ds.DATA + i, ("row", "chan", "corr")))

        writes.append(write_columns(big_ms, nds, ["TIME", "DATA"]))

    dask.compute(writes)
