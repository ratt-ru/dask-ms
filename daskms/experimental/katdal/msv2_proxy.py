from functools import partial

import dask.array as da
import numpy as np

from katdal.dataset import DataSet
from katdal.lazy_indexer import DaskLazyIndexer
from katpoint import Timestamp
import numba
import xarray

from daskms.experimental.katdal.corr_products import corrprod_index
from daskms.experimental.katdal.transpose import transpose
from daskms.experimental.katdal.uvw import uvw_coords


PROXIED_PROPERTIES = [
    "ants",
    "size",
    "shape",
    "catalogue",
    "scan_indices",
    "target_indices" "name",
    "experiment_id",
    "observer",
    "description",
    "version",
]


def property_factory(name: str):
    def impl(self):
        return getattr(self._dataset, name)

    impl.__name__ = name
    impl.__doc__ = f"Proxies :attr:`katdal.Dataset.{name}"

    return property(impl)


class MSv2DataProxyMetaclass(type):
    def __new__(cls, name, bases, dct):
        for p in PROXIED_PROPERTIES:
            dct[p] = property_factory(p)

        return type.__new__(cls, name, bases, dct)


class MSv2DatasetProxy(metaclass=MSv2DataProxyMetaclass):
    """Proxies a katdal dataset to present an MSv2 view over archive data"""

    def __init__(
        self, dataset: DataSet, auto_corrs: bool = True, row_view: bool = True
    ):
        # Reset the dataset selection
        self._dataset = dataset
        self._auto_corrs = auto_corrs
        self._row_view = row_view
        self._pols_to_use = ["HH", "HV", "VH", "VV"]
        self.select(reset="")
        self._antennas = dataset.ants.copy()

    @property
    def cp_info(self):
        return self._cp_info

    def select(self, **kwargs):
        """Proxies :meth:`katdal.select`"""
        result = self._dataset.select(**kwargs)
        self._cp_info = corrprod_index(
            self._dataset, self._pols_to_use, self._auto_corrs
        )
        return result

    def _main_xarray_factory(self, scan_index, scan_state, target):
        # Extract numpy and dask products
        dataset = self._dataset
        cp_info = self._cp_info
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
            row=self._row_view,
        )
        weight_transpose = partial(
            transpose,
            cp_index=cpi,
            data_type=numba.literally("weights"),
            row=self._row_view,
        )
        vis_transpose = partial(
            transpose,
            cp_index=cpi,
            data_type=numba.literally("vis"),
            row=self._row_view,
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
            self._antennas,
            cp_info,
            row=self._row_view,
        )

        time, ant1, ant2 = da.broadcast_arrays(time, ant1, ant2)

        if self._row_view:
            primary_dims = ("row",)
            time = time.ravel().rechunk({0: vis.dataset.chunks[0]})
            ant1 = ant1.ravel().rechunk({0: vis.dataset.chunks[0]})
            ant2 = ant2.ravel().rechunk({0: vis.dataset.chunks[0]})
        else:
            primary_dims = ("time", "baseline")

        data_vars = {
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

        return xarray.Dataset(data_vars)

    def scans(self):
        """Proxies :meth:`katdal.scans`"""
        xds = [
            self._main_xarray_factory(*scan_data) for scan_data in self._dataset.scans()
        ]
        yield xarray.concat(xds, dim="row" if self._row_view else "time")

    def _antenna_xarray_factory(self):
        antennas = self._dataset.ants
        nant = len(antennas)
        return xarray.Dataset(
            {
                "NAME": ("row", np.asarray([a.name for a in antennas], dtype=object)),
                "STATION": (
                    "row",
                    np.asarray([a.name for a in antennas], dtype=object),
                ),
                "POSITION": (
                    ("row", "xyz"),
                    np.asarray([a.position_ecef for a in antennas]),
                ),
                "OFFSET": (("row", "xyz"), np.zeros((nant, 3))),
                "DISH_DIAMETER": ("row", np.asarray([a.diameter for a in antennas])),
                "MOUNT": ("row", np.array(["ALT-AZ"] * nant, dtype=object)),
                "TYPE": ("row", np.array(["GROUND-BASED"] * nant, dtype=object)),
                "FLAG_ROW": ("row", np.zeros(nant, dtype=np.int32)),
            }
        )

    def _spw_xarray_factory(self):
        def ref_freq(chan_freqs):
            return chan_freqs[len(chan_freqs) // 2].astype(np.float64)

        return [
            xarray.Dataset(
                {
                    "NUM_CHAN": (("row",), np.array([spw.num_chans], dtype=np.int32)),
                    "CHAN_FREQ": (("row", "chan"), spw.channel_freqs[np.newaxis, :]),
                    "RESOLUTION": (("row", "chan"), spw.channel_freqs[np.newaxis, :]),
                    "CHAN_WIDTH": (
                        ("row", "chan"),
                        np.full_like(spw.channel_freqs, spw.channel_width)[
                            np.newaxis, :
                        ],
                    ),
                    "EFFECTIVE_BW": (
                        ("row", "chan"),
                        np.full_like(spw.channel_freqs, spw.channel_width)[
                            np.newaxis, :
                        ],
                    ),
                    "MEAS_FREQ_REF": ("row", np.array([5], dtype=np.int32)),
                    "REF_FREQUENCY": ("row", [ref_freq(spw.channel_freqs)]),
                    "NAME": ("row", np.asarray([f"{spw.band}-band"], dtype=object)),
                    "FREQ_GROUP_NAME": (
                        "row",
                        np.asarray([f"{spw.band}-band"], dtype=object),
                    ),
                    "FREQ_GROUP": ("row", np.zeros(1, dtype=np.int32)),
                    "IF_CONV_CHAN": ("row", np.zeros(1, dtype=np.int32)),
                    "NET_SIDEBAND": ("row", np.ones(1, dtype=np.int32)),
                    "TOTAL_BANDWIDTH": ("row", np.asarray([spw.channel_freqs.sum()])),
                    "FLAG_ROW": ("row", np.zeros(1, dtype=np.int32)),
                }
            )
            for spw in self._dataset.spectral_windows
        ]

    def _pol_xarray_factory(self):
        pol_num = {"H": 0, "V": 1}
        # MeerKAT only has linear feeds, these map to
        # CASA ["XX", "XY", "YX", "YY"]
        pol_types = {"HH": 9, "HV": 10, "VH": 11, "VV": 12}
        return xarray.Dataset(
            {
                "NUM_CORR": ("row", np.array([len(self._pols_to_use)], dtype=np.int32)),
                "CORR_PRODUCT": (
                    ("row", "corr", "corrprod_idx"),
                    np.array(
                        [[[pol_num[p[0]], pol_num[p[1]]] for p in self._pols_to_use]],
                        dtype=np.int32,
                    ),
                ),
                "CORR_TYPE": (
                    ("row", "corr"),
                    np.asarray(
                        [[pol_types[p] for p in self._pols_to_use]], dtype=np.int32
                    ),
                ),
                "FLAG_ROW": ("row", np.zeros(1, dtype=np.int32)),
            }
        )

    def _ddid_xarray_factory(self):
        return xarray.Dataset(
            {
                "SPECTRAL_WINDOW_ID": np.zeros(1, dtype=np.int32),
                "POLARIZATION_ID": np.zeros(1, dtype=np.int32),
                "FLAG_ROW": np.zeros(1, dtype=np.int32),
            }
        )

    def _feed_xarray_factory(self):
        nfeeds = len(self._dataset.ants)
        NRECEPTORS = 2

        return xarray.Dataset(
            {
                # ID of antenna in this array (integer)
                "ANTENNA_ID": ("row", np.arange(nfeeds, dtype=np.int32)),
                # Id for BEAM model (integer)
                "BEAM_ID": ("row", np.ones(nfeeds, dtype=np.int32)),
                # Beam position offset (on sky but in antenna reference frame): (double, 2-dim)
                "BEAM_OFFSET": (
                    ("row", "receptors", "radec"),
                    np.zeros((nfeeds, 2, 2), dtype=np.float64),
                ),
                # Feed id (integer)
                "FEED_ID": ("row", np.zeros(nfeeds, dtype=np.int32)),
                # Interval for which this set of parameters is accurate (double)
                "INTERVAL": ("row", np.zeros(nfeeds, dtype=np.float64)),
                # Number of receptors on this feed (probably 1 or 2) (integer)
                "NUM_RECEPTORS": ("row", np.full(nfeeds, NRECEPTORS, dtype=np.int32)),
                # Type of polarisation to which a given RECEPTOR responds (string, 1-dim)
                "POLARIZATION_TYPE": (
                    ("row", "receptors"),
                    np.array([["X", "Y"]] * nfeeds, dtype=object),
                ),
                # D-matrix i.e. leakage between two receptors (complex, 2-dim)
                "POL_RESPONSE": (
                    ("row", "receptors", "receptors-2"),
                    np.array([np.eye(2, dtype=np.complex64) for _ in range(nfeeds)]),
                ),
                # Position of feed relative to feed reference position (double, 1-dim, shape=(3,))
                "POSITION": (("row", "xyz"), np.zeros((nfeeds, 3), np.float64)),
                # The reference angle for polarisation (double, 1-dim). A parallactic angle of
                # 0 means that V is aligned to x (celestial North), but we are mapping H to x
                # so we have to correct with a -90 degree rotation.
                "RECEPTOR_ANGLE": (
                    ("row", "receptors"),
                    np.full((nfeeds, NRECEPTORS), -np.pi / 2, dtype=np.float64),
                ),
                # ID for this spectral window setup (integer)
                "SPECTRAL_WINDOW_ID": ("row", np.full(nfeeds, -1, dtype=np.int32)),
                # Midpoint of time for which this set of parameters is accurate (double)
                "TIME": ("row", np.zeros(nfeeds, dtype=np.float64)),
            }
        )

    def subtables(self):
        self.select(reset="")

        return {
            "ANTENNA": self._antenna_xarray_factory(),
            "DATA_DESCRIPTION": self._ddid_xarray_factory(),
            "SPECTRAL_WINDOW": self._spw_xarray_factory(),
            "POLARIZATION": self._pol_xarray_factory(),
            "FEED": self._feed_xarray_factory(),
        }
