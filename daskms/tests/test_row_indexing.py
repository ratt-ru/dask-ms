# -*- coding: utf-8 -*-
"""arcae receives dataset row ids directly, in dataset order"""

import dask
import dask.array as da
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from daskms import xds_from_table, xds_to_table
from daskms.casa_table import build_index, open_table
from daskms.dataset import Dataset


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([3, 4, 5], slice(3, 6)),
        ([7], slice(7, 8)),
        ([5, 4, 3], [5, 4, 3]),
        ([0, 2, 3], [0, 2, 3]),
        ([], []),
    ],
)
def test_build_index_rows(rows, expected):
    (index,) = build_index(np.asarray(rows, dtype=np.int32))

    if isinstance(expected, slice):
        assert index == expected
    else:
        assert index.dtype == np.int64
        assert_array_equal(index, expected)


@pytest.mark.parametrize(
    "chunks",
    [{"row": 3}, {"row": 3, "chan": 4, "corr": 2}],
    ids=lambda c: f"chunks={c}",
)
def test_unsorted_rowid_roundtrip(ms, chunks):
    rs = np.random.RandomState(42)
    rowid = rs.permutation(10)
    data = rs.random_sample((10, 16, 4)) + rs.random_sample((10, 16, 4)) * 1j
    row_chunks, chan_chunks, corr_chunks = (
        chunks.get(d, -1) for d in ("row", "chan", "corr")
    )
    data_chunks = (row_chunks, chan_chunks, corr_chunks)

    ds = Dataset(
        {"DATA": (("row", "chan", "corr"), da.from_array(data, chunks=data_chunks))},
        coords={"ROWID": (("row",), da.from_array(rowid, chunks=row_chunks))},
    )
    dask.compute(xds_to_table(ds, ms, ["DATA"]))

    # Row i of the dataset was written to table row rowid[i]
    on_disk = open_table(ms).getcol("DATA")
    assert_array_equal(on_disk[rowid], data)

    # Reading the rows back in the same order recovers the dataset
    (rds,) = xds_from_table(ms, columns=["DATA"], chunks=chunks)
    assert_array_equal(rds.DATA.data[rowid].compute(), data)


def test_duplicate_rowid_write_raises(ms):
    ds = Dataset(
        {"STATE_ID": (("row",), da.arange(3, chunks=3, dtype=np.int32))},
        coords={"ROWID": (("row",), da.from_array(np.array([1, 1, 2]), chunks=3))},
    )

    with pytest.raises(Exception, match="Duplicate"):
        dask.compute(xds_to_table(ds, ms, ["STATE_ID"]))


def test_append_with_empty_row_chunk(tmp_path):
    table = str(tmp_path / "append.table")
    data = np.arange(5 * 3, dtype=np.float64).reshape(5, 3)

    ds = Dataset(
        {"VALUE": (("row", "comp"), da.from_array(data, chunks=((3, 0, 2), 3)))}
    )
    dask.compute(xds_to_table(ds, table, ["VALUE"]))

    t = open_table(table)
    assert t.nrow() == 5
    assert_array_equal(t.getcol("VALUE"), data)

    # Appending a second dataset continues after the existing rows
    dask.compute(xds_to_table(ds, table, ["VALUE"]))
    t = open_table(table)
    assert t.nrow() == 10
    assert_array_equal(t.getcol("VALUE"), np.concatenate([data, data]))


def test_read_empty_table(tmp_path):
    table = str(tmp_path / "empty.table")
    data = np.zeros((1, 3), dtype=np.float64)
    ds = Dataset({"VALUE": (("row", "comp"), da.from_array(data, chunks=(1, 3)))})
    dask.compute(xds_to_table(ds, table, ["VALUE"]))

    # Select no rows so the only row chunk is empty
    (rds,) = xds_from_table(table, taql_where="ROWID() < 0", columns=["VALUE"])
    assert rds.VALUE.shape == (0, 3)
    assert rds.VALUE.data.compute().shape == (0, 3)


def test_write_graph(ms):
    """A write is one putcol per block, fed by the data and its row ids.
    Nothing is cached or inlined in the graph"""
    (ds,) = xds_from_table(ms, columns=["DATA"], chunks={"row": 3, "chan": 4})
    (write,) = xds_to_table(ds, ms, ["DATA"])
    graph = write.DATA.data.__dask_graph__()
    prefixes = {name.split("~")[0].split("-")[0] for name in graph.layers}

    # Read the data and its rows, give the rows the data's rank, then write
    assert prefixes == {"read", "rowid", "getitem", "write"}
    assert write.DATA.data.numblocks == ds.DATA.data.numblocks
    dask.compute(write)
