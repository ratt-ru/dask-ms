# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask
import pytest


from xarrayms.dataset import dataset, write_dataset


@pytest.mark.stress
@pytest.mark.parametrize("big_ms", [1000], indirect=True)
@pytest.mark.parametrize("iterations", [10])
@pytest.mark.parametrize("chunks", [{"row": 100}])
def test_stress(big_ms, iterations, chunks):
    datasets = dataset(big_ms, ["TIME", "DATA"],
                       ["FIELD_ID", "DATA_DESC_ID"], [],
                       chunks=chunks)

    assert len(datasets) == 1
    ds = datasets[0]

    writes = []

    for i in range(iterations):
        nds = ds.assign(TIME=ds.TIME + i, DATA=ds.DATA + i)
        writes.append(write_dataset(big_ms, nds, ["TIME", "DATA"]))

    dask.compute(writes)
