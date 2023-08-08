import pytest
import dask
import dask.array as da
import numpy.testing as npt
from daskms import xds_from_storage_ms
from daskms.experimental.fragments import (
    xds_to_table_fragment,
    xds_from_ms_fragment,
    xds_from_table_fragment,
)

# Prevent warning pollution generated by all calls to xds_from_zarr with
# unsupported kwargs.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The following unsupported kwargs were ignored in xds_from_zarr"
)


@pytest.fixture(
    scope="module",
    params=[
        ("DATA_DESC_ID", "FIELD_ID", "SCAN_NUMBER"),
        ("DATA_DESC_ID", "FIELD_ID"),
        ("DATA_DESC_ID",),
    ],
)
def group_cols(request):
    return request.param


def test_fragment_with_noop(ms, tmp_path_factory, group_cols):
    """Unchanged data_vars must remain the same when read from a fragment."""
    reads = xds_from_storage_ms(
        ms,
        index_cols=("TIME",),
        group_cols=group_cols,
    )

    fragment_path = tmp_path_factory.mktemp("fragment0.ms")

    writes = xds_to_table_fragment(reads, fragment_path, ms, columns=("DATA",))

    dask.compute(writes)

    fragment_reads = xds_from_ms_fragment(
        fragment_path,
        index_cols=("TIME",),
        group_cols=group_cols,
    )

    for rxds, frxds in zip(reads, fragment_reads):
        for dv in rxds.data_vars.keys():
            npt.assert_array_equal(rxds[dv].data, frxds[dv].data)


def test_fragment_with_update(ms, tmp_path_factory, group_cols):
    """Updated data_vars must change when read from a fragment."""
    reads = xds_from_storage_ms(
        ms,
        index_cols=("TIME",),
        group_cols=group_cols,
    )

    fragment_path = tmp_path_factory.mktemp("fragment0.ms")

    updates = [
        xds.assign({"DATA": (xds.DATA.dims, da.ones_like(xds.DATA.data))})
        for xds in reads
    ]

    writes = xds_to_table_fragment(updates, fragment_path, ms, columns=("DATA",))

    dask.compute(writes)

    fragment_reads = xds_from_ms_fragment(
        fragment_path,
        index_cols=("TIME",),
        group_cols=group_cols,
    )

    for frxds in fragment_reads:
        npt.assert_array_equal(1, frxds.DATA.data)


def test_nonoverlapping_parents(ms, tmp_path_factory, group_cols):
    """All updated data_vars must change when read from a fragment."""
    reads = xds_from_storage_ms(
        ms,
        index_cols=("TIME",),
        group_cols=group_cols,
    )

    fragment0_path = tmp_path_factory.mktemp("fragment0.ms")

    updates = [
        xds.assign({"DATA": (xds.DATA.dims, da.zeros_like(xds.DATA.data))})
        for xds in reads
    ]

    writes = xds_to_table_fragment(updates, fragment0_path, ms, columns=("DATA",))

    dask.compute(writes)

    fragment0_reads = xds_from_ms_fragment(
        fragment0_path,
        index_cols=("TIME",),
        group_cols=group_cols,
    )

    fragment1_path = tmp_path_factory.mktemp("fragment1.ms")

    updates = [
        xds.assign({"UVW": (xds.UVW.dims, da.zeros_like(xds.UVW.data))})
        for xds in fragment0_reads
    ]

    writes = xds_to_table_fragment(
        updates, fragment1_path, fragment0_path, columns=("UVW",)
    )

    dask.compute(writes)

    fragment1_reads = xds_from_ms_fragment(
        fragment1_path,
        index_cols=("TIME",),
        group_cols=group_cols,
    )

    for frxds in fragment1_reads:
        npt.assert_array_equal(0, frxds.DATA.data)
        npt.assert_array_equal(0, frxds.UVW.data)


def test_overlapping_parents(ms, tmp_path_factory, group_cols):
    """Youngest child takes priority if updated data_vars overlap."""
    reads = xds_from_storage_ms(
        ms,
        index_cols=("TIME",),
        group_cols=group_cols,
    )

    fragment0_path = tmp_path_factory.mktemp("fragment0.ms")

    updates = [
        xds.assign({"DATA": (xds.DATA.dims, da.ones_like(xds.DATA.data))})
        for xds in reads
    ]

    writes = xds_to_table_fragment(updates, fragment0_path, ms, columns=("DATA",))

    dask.compute(writes)

    fragment0_reads = xds_from_ms_fragment(
        fragment0_path,
        index_cols=("TIME",),
        group_cols=group_cols,
    )

    fragment1_path = tmp_path_factory.mktemp("fragment1.ms")

    updates = [
        xds.assign({"DATA": (xds.DATA.dims, da.zeros_like(xds.DATA.data))})
        for xds in fragment0_reads
    ]

    writes = xds_to_table_fragment(
        updates, fragment1_path, fragment0_path, columns=("DATA",)
    )

    dask.compute(writes)

    fragment1_reads = xds_from_ms_fragment(
        fragment1_path,
        index_cols=("TIME",),
        group_cols=group_cols,
    )

    for frxds in fragment1_reads:
        npt.assert_array_equal(0, frxds.DATA.data)


def test_inconsistent_partitioning(ms, tmp_path_factory, group_cols):
    """Raises a ValueError when parititoning would be inconsistent."""
    reads = xds_from_storage_ms(
        ms,
        index_cols=("TIME",),
        group_cols=group_cols,
    )

    fragment_path = tmp_path_factory.mktemp("fragment0.ms")

    writes = xds_to_table_fragment(reads, fragment_path, ms, columns=("DATA",))

    dask.compute(writes)

    with pytest.raises(ValueError, match="consolidate failed"):
        xds_from_ms_fragment(
            fragment_path,
            index_cols=("TIME",),
            group_cols=(),
        )
