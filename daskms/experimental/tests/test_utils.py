import pytest
import numpy as np
import dask.array as da
from itertools import product, combinations
from daskms.experimental.zarr import xds_to_zarr
from daskms.experimental.utils import rechunk_by_size

xarray = pytest.importorskip("xarray")

ZARR_MAX_CHUNK = 2 ** (32 - 1)


@pytest.fixture(scope="function")
def dataset():
    ndim = 3

    def get_large_shape(ndim, dtype, max_size=2**31, exceed=1.2):
        dim_size = ((exceed * max_size) / dtype().itemsize) ** (1 / ndim)
        return (int(np.ceil(dim_size)),) * ndim

    large_shape = get_large_shape(ndim, np.complex64)

    dv0 = da.zeros(large_shape, dtype=np.complex64, chunks=-1)
    dv1 = da.zeros(large_shape, dtype=np.float32, chunks=-1)
    dv2 = da.zeros(large_shape[0], dtype=int, chunks=-1)

    coord_names = [f"coord{i}" for i in range(ndim)]

    xds = xarray.Dataset(
        {
            "dv0": (coord_names[: dv0.ndim], dv0),
            "dv1": (coord_names[: dv1.ndim], dv1),
            "dv2": (coord_names[: dv2.ndim], dv2),
        },
        coords={cn: (cn, range(ds)) for cn, ds in zip(coord_names, large_shape)},
    )

    return xds


def test_error_before_rechunk(dataset, tmp_path_factory):
    """Original motivating case - chunks too large for zarr compressor."""

    tmp_dir = tmp_path_factory.mktemp("datasets")
    zarr_path = tmp_dir / "dataset.zarr"

    with pytest.raises(ValueError, match=r"Column .* has a chunk of"):
        xds_to_zarr(dataset, zarr_path)


def test_error_after_rechunk(dataset, tmp_path_factory):
    """Check that rechunking solves the original morivating case."""

    tmp_dir = tmp_path_factory.mktemp("datasets")
    zarr_path = tmp_dir / "dataset.zarr"

    xds_to_zarr(rechunk_by_size(dataset), zarr_path)  # No error.


@pytest.mark.parametrize("max_chunk_mem", [2**28, 2**29, 2**30])
def test_rechunk(dataset, max_chunk_mem):
    """Check that rechunking works for a range of target sizes."""

    dataset = rechunk_by_size(dataset, max_chunk_mem=max_chunk_mem)

    for dv in dataset.data_vars.values():
        itr = product(*map(range, dv.data.blocks.shape))
        assert all(dv.data.blocks[i].nbytes < max_chunk_mem for i in itr), (
            f"Data variable {dv.name} contains chunks which exceed the "
            f"maximum per chunk memory size of {max_chunk_mem}."
        )


@pytest.mark.parametrize(
    "unchunked_dims",
    [*combinations(["coord0", "coord1", "coord2"], 2), *["coord0", "coord1", "coord2"]],
)
def test_rechunk_with_unchunkable_axis(dataset, unchunked_dims):
    """Check that rechunking works when some dimensions must not be chunked."""

    dataset = rechunk_by_size(
        dataset, max_chunk_mem=ZARR_MAX_CHUNK, unchunked_dims={unchunked_dims}
    )

    for dv in dataset.data_vars.values():
        itr = product(*map(range, dv.data.blocks.shape))
        assert all(dv.data.blocks[i].nbytes < ZARR_MAX_CHUNK for i in itr), (
            f"Data variable {dv.name} contains chunks which exceed the "
            f"maximum per chunk memory size of {ZARR_MAX_CHUNK}."
        )


def test_rechunk_impossible(dataset):
    """Check that rechunking raises a sensible error in impossible cases."""

    with pytest.raises(ValueError, match="Target chunk size could not be"):
        rechunk_by_size(
            dataset,
            max_chunk_mem=ZARR_MAX_CHUNK,
            unchunked_dims={"coord0", "coord1", "coord2"},
        )


def test_rechunk_if_required(dataset):
    dataset = dataset.chunk({c: 100 for c in dataset.coords.keys()})

    rechunked_dataset = rechunk_by_size(dataset, only_when_needed=True)

    assert rechunked_dataset.chunks == dataset.chunks, (
        "rechunk_by_size has altered chunk sizes even though input dataset "
        "did not require rechunking."
    )
