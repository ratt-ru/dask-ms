# -*- coding: utf-8 -*-

import dask
import pytest


from daskms.reads import read_datasets
from daskms.writes import write_datasets


@pytest.mark.stress
@pytest.mark.parametrize("big_ms", [1000], indirect=True)
@pytest.mark.parametrize("iterations", [10])
@pytest.mark.parametrize("chunks", [{"row": 100}])
def test_stress(big_ms, iterations, chunks):
    datasets = read_datasets(big_ms, ["TIME", "DATA"],
                             ["FIELD_ID", "DATA_DESC_ID"], [],
                             chunks=chunks)

    assert len(datasets) == 1
    ds = datasets[0]

    writes = []

    for i in range(iterations):
        nds = ds.assign(TIME=(("row",), ds.TIME.data + i),
                        DATA=(("row", "chan", "corr"), ds.DATA.data + i))
        writes.append(write_datasets(big_ms, nds, ["TIME", "DATA"]))

    dask.compute(writes)
