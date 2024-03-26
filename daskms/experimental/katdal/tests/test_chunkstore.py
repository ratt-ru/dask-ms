from functools import partial

import pytest

dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")
xarray = pytest.importorskip("xarray")
katdal = pytest.importorskip("katdal")
katpoint = pytest.importorskip("katpoint")

import numba
import numpy as np
from numpy.testing import assert_array_equal

from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr

from katdal.lazy_indexer import DaskLazyIndexer
from katpoint import Timestamp

from daskms.experimental.katdal.transpose import transpose
from daskms.experimental.katdal.corr_products import corrprod_index
from daskms.experimental.katdal.uvw import uvw_coords


@pytest.mark.parametrize(
    "dataset", [{"ntime": 20, "nchan": 16, "nant": 4}], indirect=True
)
@pytest.mark.parametrize("include_auto_corrs", [True])
@pytest.mark.parametrize("row_dim", [True, False])
@pytest.mark.parametrize("out_store", ["output.zarr"])
def test_chunkstore(tmp_path_factory, dataset, include_auto_corrs, row_dim, out_store):
    # Example using an actual mvf4 dataset, make sure to replace the token with a real one
    # url = "https://archive-gw-1.kat.ac.za/1711249692/1711249692_sdp_l0.full.rdb?token=abcdef1234567890"
    # import katdal
    # dataset = katdal.open(url, applycal="l1")
    cp_info = corrprod_index(dataset, ["HH", "HV", "VH", "VV"], include_auto_corrs)
    all_antennas = dataset.ants

    xds = []

    for scan_index, scan_state, target in dataset.scans():
        # Extract numpy and dask products
        time_utc = dataset.timestamps
        t_chunks, chan_chunks, cp_chunks = dataset.vis.dataset.chunks

        # Modified Julian Date in Seconds
        time_mjds = np.asarray(
            [t.to_mjd() * 24 * 60 * 60 for t in map(Timestamp, time_utc)]
        )

        # Create a dask chunking transform
        rechunk = partial(da.rechunk, chunks=(t_chunks, chan_chunks, cp_chunks))

        # Transpose from (time, chan, corrprod) to (time, bl, chan, corr)
        cpi = cp_info.cp_index
        flag_transpose = partial(
            transpose,
            cp_index=cpi,
            data_type=numba.literally("flags"),
            row=row_dim,
        )
        weight_transpose = partial(
            transpose,
            cp_index=cpi,
            data_type=numba.literally("weights"),
            row=row_dim,
        )
        vis_transpose = partial(
            transpose, cp_index=cpi, data_type=numba.literally("vis"), row=row_dim
        )

        flags = DaskLazyIndexer(dataset.flags, (), (rechunk, flag_transpose))
        weights = DaskLazyIndexer(dataset.weights, (), (rechunk, weight_transpose))
        vis = DaskLazyIndexer(dataset.vis, (), transforms=(vis_transpose,))

        time = da.from_array(time_mjds[:, None], chunks=(t_chunks, 1))
        ant1 = da.from_array(cp_info.ant1_index[None, :], chunks=(1, cpi.shape[0]))
        ant2 = da.from_array(cp_info.ant2_index[None, :], chunks=(1, cpi.shape[0]))

        uvw = uvw_coords(
            target,
            da.from_array(time_utc, chunks=t_chunks),
            all_antennas,
            cp_info,
            row=row_dim,
        )

        time, ant1, ant2 = da.broadcast_arrays(time, ant1, ant2)

        if row_dim:
            primary_dims = ("row",)
            time = time.ravel().rechunk({0: vis.dataset.chunks[0]})
            ant1 = ant1.ravel().rechunk({0: vis.dataset.chunks[0]})
            ant2 = ant2.ravel().rechunk({0: vis.dataset.chunks[0]})
        else:
            primary_dims = ("time", "baseline")

        xds.append(
            xarray.Dataset(
                {
                    # Primary indexing columns
                    "TIME": (primary_dims, time),
                    "ANTENNA1": (primary_dims, ant1),
                    "ANTENNA2": (primary_dims, ant2),
                    "FEED1": (primary_dims, da.zeros_like(ant1)),
                    "FEED2": (primary_dims, da.zeros_like(ant1)),
                    # TODO(sjperkins)
                    # Fill these in with real values
                    "DATA_DESC_ID": (primary_dims, da.zeros_like(ant1)),
                    "FIELD_ID": (primary_dims, da.zeros_like(ant1)),
                    "STATE_ID": (primary_dims, da.zeros_like(ant1)),
                    "ARRAY_ID": (primary_dims, da.zeros_like(ant1)),
                    "OBSERVATION_ID": (primary_dims, da.zeros_like(ant1)),
                    "PROCESSOR_ID": (primary_dims, da.ones_like(ant1)),
                    "SCAN_NUMBER": (primary_dims, da.full_like(ant1, scan_index)),
                    "TIME_CENTROID": (primary_dims, time),
                    "INTERVAL": (primary_dims, da.full_like(time, dataset.dump_period)),
                    "EXPOSURE": (primary_dims, da.full_like(time, dataset.dump_period)),
                    "UVW": (primary_dims + ("uvw",), uvw),
                    "DATA": (primary_dims + ("chan", "corr"), vis.dataset),
                    "FLAG": (primary_dims + ("chan", "corr"), flags.dataset),
                    "WEIGHT_SPECTRUM": (
                        primary_dims + ("chan", "corr"),
                        weights.dataset,
                    ),
                    # Estimated RMS noise per frequency channel
                    # note this column is used when computing calibration weights
                    # in CASA - WEIGHT_SPECTRUM may be modified based on the
                    # values in this column. See
                    # https://casadocs.readthedocs.io/en/stable/notebooks/data_weights.html
                    # for further details
                    "SIGMA_SPECTRUM": (
                        primary_dims + ("chan", "corr"),
                        weights.dataset**-0.5,
                    ),
                }
            )
        )

    xds = xarray.concat(xds, dim=primary_dims[0])

    ant_xds = [
        xarray.Dataset(
            {
                "NAME": (("row",), np.asarray([a.name for a in all_antennas])),
                "OFFSET": (
                    ("row", "xyz"),
                    np.asarray(np.zeros((len(all_antennas), 3))),
                ),
                "POSITION": (
                    ("row", "xyz"),
                    np.asarray([a.position_ecef for a in all_antennas]),
                ),
                "DISH_DIAMETER": (
                    ("row",),
                    np.asarray([a.diameter for a in all_antennas]),
                ),
                # "FLAG_ROW": (("row","xyz"),
                #             np.zeros([a.flags for a in all_antennas],np.uint8)
                # )
            }
        )
    ]

    spw = dataset.spectral_windows[dataset.spw]

    spw_xds = [
        xarray.Dataset(
            {
                "CHAN_FREQ": (("row", "chan"), dataset.channel_freqs[np.newaxis, :]),
                "CHAN_WIDTH": (
                    ("row", "chan"),
                    np.full_like(dataset.channel_freqs, dataset.channel_width)[
                        np.newaxis, :
                    ],
                ),
                "EFFECTIVE_BW": (
                    ("row", "chan"),
                    np.full_like(dataset.channel_freqs, dataset.channel_width)[
                        np.newaxis, :
                    ],
                ),
                "FLAG_ROW": (("row",), np.zeros(1, dtype=np.int32)),
                "NUM_CHAN": (("row",), np.array([spw.num_chans], dtype=np.int32)),
            }
        )
    ]

    print(spw_xds)
    print(ant_xds)
    print(xds)

    # Reintroduce the shutil.rmtree and remote the tmp_path_factory
    # to test in the local directory
    # shutil.rmtree(out_store, ignore_errors=True)
    out_store = tmp_path_factory.mktemp("output") / out_store

    dask.compute(xds_to_zarr(xds, out_store))
    dask.compute(xds_to_zarr(ant_xds, f"{out_store}::ANTENNA"))

    # Compare visibilities, weights and flags
    (read_xds,) = dask.compute(xds_from_zarr(out_store))
    (read_xds,) = read_xds

    test_data = dataset._test_data["correlator_data"]
    # Defer to ChunkStoreVisWeights application of weight scaling
    test_weights = dataset._vfw.weights
    assert test_weights.shape == test_data.shape
    # Clamp test data to [0, 1]
    test_flags = np.where(dataset._test_data["flags"] != 0, 1, 0)
    ntime, nchan, _ = test_data.shape
    (nbl,) = cp_info.ant1_index.shape
    ncorr = read_xds.sizes["corr"]

    # This must hold for test_tranpose to work
    assert_array_equal(cp_info.cp_index.ravel(), np.arange(nbl * ncorr))

    def assert_transposed_equal(a, e):
        """Simple transpose of katdal (time, chan, corrprod) to
        (time, bl, chan, corr)."""
        t = a.reshape(ntime, nchan, nbl, ncorr).transpose(0, 2, 1, 3)
        t = t.reshape(-1, nchan, ncorr) if row_dim else t
        return assert_array_equal(t, e)

    assert_transposed_equal(test_data, read_xds.DATA.values)
    assert_transposed_equal(test_weights, read_xds.WEIGHT_SPECTRUM.values)
    assert_transposed_equal(test_flags, read_xds.FLAG.values)
