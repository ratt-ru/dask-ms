# -*- coding: utf-8 -*-

import concurrent.futures as cf

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from daskms.partition import TablePartitioner

pytest.importorskip("arcae")


def _reference(index, partitionby, sortby):
    """Partition and sort with a single lexsort over the whole index"""
    rows = np.arange(len(next(iter(index.values()))))
    keys = [index[c] for c in partitionby + sortby] + [rows]
    order = np.lexsort(tuple(reversed(keys)))
    partitions = {}

    for row in order:
        key = tuple((c, index[c][row].item()) for c in partitionby)
        partitions.setdefault(key, []).append(row)

    return {k: np.array(v) for k, v in partitions.items()}


def _index(nrow, rs):
    return {
        "FIELD_ID": rs.randint(0, 3, nrow).astype(np.int32),
        "DATA_DESC_ID": rs.randint(0, 2, nrow).astype(np.int32),
        "TIME": rs.randint(0, 5, nrow).astype(np.float64),
        "ANTENNA1": rs.randint(0, 4, nrow).astype(np.int32),
    }


@pytest.mark.parametrize("nworkers", [1, 3, 8])
@pytest.mark.parametrize("nrow", [1, 2, 7, 100, 1001])
def test_partition_matches_lexsort(nworkers, nrow):
    rs = np.random.RandomState(nrow)
    index = _index(nrow, rs)
    partitionby = ["FIELD_ID", "DATA_DESC_ID"]
    sortby = ["TIME", "ANTENNA1"]
    partitioner = TablePartitioner(partitionby, sortby, ["row"])

    with cf.ThreadPoolExecutor(nworkers) as pool:
        partitions = partitioner.partition(index, pool)

    expected = _reference(index, partitionby, sortby)

    # Keys come out in ascending order
    assert list(partitions) == sorted(expected)

    for key, rows in expected.items():
        partition = partitions[key]
        assert_array_equal(partition["row"], rows)

        for column, values in index.items():
            assert partition[column].dtype == values.dtype
            assert_array_equal(partition[column], values[rows])


@pytest.mark.parametrize("nworkers", [1, 4])
def test_partition_unmergeable_dtypes(nworkers):
    """Columns that merge_np_partitions rejects are still sorted correctly"""
    rs = np.random.RandomState(42)
    nrow = 50
    index = {
        "OBS_MODE": rs.choice(["TARGET", "CALIBRATE", "SCAN"], nrow),
        "FLAG_ROW": rs.randint(0, 2, nrow).astype(np.bool_),
        "SMALL": rs.randint(-3, 3, nrow).astype(np.int16),
        "BIG": rs.randint(0, 4, nrow).astype(np.uint64) + np.uint64(2**63),
        "TIME": rs.random_sample(nrow).astype(np.float32),
    }
    partitionby = ["OBS_MODE", "FLAG_ROW"]
    sortby = ["SMALL", "BIG", "TIME"]
    partitioner = TablePartitioner(partitionby, sortby, ["row"])

    with cf.ThreadPoolExecutor(nworkers) as pool:
        partitions = partitioner.partition(index, pool)

    expected = _reference(index, partitionby, sortby)
    assert list(partitions) == sorted(expected)

    for key, rows in expected.items():
        assert all(type(v) in (str, bool) for _, v in key)
        assert_array_equal(partitions[key]["row"], rows)

        for column, values in index.items():
            assert partitions[key][column].dtype == values.dtype
            assert_array_equal(partitions[key][column], values[rows])


def test_partition_ungrouped():
    """Without partitioning columns, all rows form one sorted partition"""
    time = np.array([3.0, 1.0, 2.0, 1.0])
    partitioner = TablePartitioner([], ["TIME"], ["row"])

    with cf.ThreadPoolExecutor(2) as pool:
        partitions = partitioner.partition({"TIME": time}, pool)

    assert list(partitions) == [()]
    # Ties are broken by row
    assert_array_equal(partitions[()]["row"], [1, 3, 2, 0])


def test_partition_without_row():
    partitioner = TablePartitioner(["G"], ["T"])

    with cf.ThreadPoolExecutor(2) as pool:
        partitions = partitioner.partition(
            {"G": np.int32([1, 0, 1]), "T": np.float64([2.0, 5.0, 1.0])}, pool
        )

    assert list(partitions) == [(("G", 0),), (("G", 1),)]
    assert "row" not in partitions[(("G", 1),)]
    assert_array_equal(partitions[(("G", 1),)]["T"], [1.0, 2.0])


def test_partition_empty():
    partitioner = TablePartitioner(["G"], ["T"], ["row"])

    with cf.ThreadPoolExecutor(2) as pool:
        empty = {"G": np.int32([]), "T": np.float64([])}
        assert partitioner.partition(empty, pool) == {}


def test_partition_invalid_index():
    partitioner = TablePartitioner(["G"], ["T"])

    with cf.ThreadPoolExecutor(2) as pool:
        with pytest.raises(ValueError, match="length mismatch"):
            partitioner.partition({"G": np.int32([0]), "T": np.float64([])}, pool)

        with pytest.raises(ValueError, match="missing"):
            partitioner.partition({"G": np.int32([0])}, pool)

        with pytest.raises(ValueError, match="Empty index"):
            partitioner.partition({}, pool)
