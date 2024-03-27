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

TAG_TO_INTENT = {
    "gaincal": "CALIBRATE_PHASE,CALIBRATE_AMPLI",
    "bpcal": "CALIBRATE_BANDPASS,CALIBRATE_FLUX",
    "target": "TARGET",
}


def to_mjds(timestamp: Timestamp):
    """Converts a katpoint Timestamp to Modified Julian Date Seconds"""
    return timestamp.to_mjd() * 24 * 60 * 60


class XarrayMSV2Facade:
    """Provides a simplified xarray Dataset view over a katdal dataset"""

    def __init__(
        self, dataset: DataSet, auto_corrs: bool = True, row_view: bool = True
    ):
        self._dataset = dataset
        self._auto_corrs = auto_corrs
        self._row_view = row_view
        self._pols_to_use = ["HH", "HV", "VH", "VV"]
        # Reset the dataset selection
        self._dataset.select(reset="")
        self._cp_info = corrprod_index(dataset, self._pols_to_use, auto_corrs)

    @property
    def cp_info(self):
        return self._cp_info

    def _main_xarray_factory(self, state_id, scan_index, scan_state, target):
        # Extract numpy and dask products
        dataset = self._dataset
        cp_info = self._cp_info
        time_utc = dataset.timestamps
        t_chunks, chan_chunks, cp_chunks = dataset.vis.dataset.chunks

        # Modified Julian Date in Seconds
        time_mjds = np.asarray([to_mjds(t) for t in map(Timestamp, time_utc)])

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
            dataset.ants,
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
            "STATE_ID": (primary_dims, da.full_like(ant1, state_id)),
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
                "SPECTRAL_WINDOW_ID": ("row", np.zeros(1, dtype=np.int32)),
                "POLARIZATION_ID": ("row", np.zeros(1, dtype=np.int32)),
                "FLAG_ROW": ("row", np.zeros(1, dtype=np.int32)),
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

    def _field_xarray_factory(self, field_data):
        fields = [
            xarray.Dataset(
                {
                    "NAME": ("row", np.array([target.name], object)),
                    "CODE": ("row", np.array(["T"], object)),
                    "SOURCE_ID": ("row", np.array([s], dtype=np.int32)),
                    "NUM_POLY": ("row", np.zeros(1, dtype=np.int32)),
                    "TIME": ("row", np.array([time])),
                    "DELAY_DIR": (
                        ("row", "field-poly", "field-dir"),
                        np.array([[radec]], dtype=np.float64),
                    ),
                    "PHASE_DIR": (
                        ("row", "field-poly", "field-dir"),
                        np.array([[radec]], dtype=np.float64),
                    ),
                    "REFERENCE_DIR": (
                        ("row", "field-poly", "field-dir"),
                        np.array([[radec]], dtype=np.float64),
                    ),
                    "FLAG_ROW": ("row", np.zeros(1, dtype=np.int32)),
                }
            )
            for s, (time, target, radec) in enumerate(field_data)
        ]

        return xarray.concat(fields, dim="row")

    def _source_xarray_factory(self, field_data):
        times, targets, radecs = zip(*field_data)
        times = np.array(times, dtype=np.float64)
        nfields = len(times)
        return xarray.Dataset(
            {
                "NAME": ("row", np.array([t.name for t in targets], dtype=object)),
                "SOURCE_ID": ("row", np.arange(nfields, dtype=np.int32)),
                "PROPER_MOTION": (
                    ("row", "radec-per-sec"),
                    np.zeros((nfields, 2), dtype=np.float32),
                ),
                "CALIBRATION_GROUP": ("row", np.full(nfields, -1, dtype=np.int32)),
                "DIRECTION": (("row", "radec"), np.array(radecs)),
                "TIME": ("row", times),
                "NUM_LINES": ("row", np.ones(nfields, dtype=np.int32)),
                "REST_FREQUENCY": (
                    ("row", "lines"),
                    np.zeros((nfields, 1), dtype=np.float64),
                ),
            }
        )

    def _state_xarray_factory(self, state_modes):
        state_ids, modes = zip(*sorted((i, m) for m, i in state_modes.items()))
        nstates = len(state_ids)
        return xarray.Dataset(
            {
                "SIG": np.ones(nstates, dtype=np.uint8),
                "REF": np.zeros(nstates, dtype=np.uint8),
                "CAL": np.zeros(nstates, dtype=np.float64),
                "LOAD": np.zeros(nstates, dtype=np.float64),
                "SUB_SCAN": np.zeros(nstates, dtype=np.int32),
                "OBS_MODE": np.array(modes, dtype=object),
                "FLAG_ROW": np.zeros(nstates, dtype=np.int32),
            }
        )

    def _observation_xarray_factory(self):
        ds = self._dataset
        start, end = [to_mjds(t) for t in [ds.start_time, ds.end_time]]
        return xarray.Dataset(
            {
                "OBSERVER": ("row", np.array([ds.observer], dtype=object)),
                "PROJECT": ("row", np.array([ds.experiment_id], dtype=object)),
                "LOG": (("row", "extra"), np.array([["unavailable"]], dtype=object)),
                "SCHEDULE": (
                    ("row", "extra"),
                    np.array([["unavailable"]], dtype=object),
                ),
                "SCHEDULE_TYPE": ("row", np.array(["unknown"], dtype=object)),
                "TELESCOPE": ("row", np.array(["MeerKAT"], dtype=object)),
                "TIME_RANGE": (("row", "extent"), np.array([[start, end]])),
                "FLAG_ROW": ("row", np.zeros(1, np.uint8)),
            }
        )

    def xarray_datasets(self):
        """Generates partitions of the main MSv2 table, as well as the subtables.

        Returns
        -------
        main_xds: list of :code:`xarray.Dataset`
            A list of xarray datasets corresponding to Measurement Set 2
            partitions
        subtable_xds: dict of :code:`xarray.Dataset`
            A dictionary of datasets keyed on subtable names
        """
        main_xds = []
        field_data = []
        state_modes = {"UNKNOWN": 0}

        # Generate MAIN table xarray partition datasets
        for scan_index, scan_state, target in self._dataset.scans():
            # Create per-scan field and source data
            time_origin = Timestamp(self._dataset.timestamps[0])
            field_data.append((to_mjds(time_origin), target, target.radec(time_origin)))

            # Create or retrieve the state_id associated
            # with the tags of the current source
            state_tag = ",".join(
                TAG_TO_INTENT[tag] for tag in target.tags if tag in TAG_TO_INTENT
            )
            if state_tag and state_tag not in state_modes:
                state_modes[state_tag] = len(state_modes)
            state_id = state_modes.get(state_tag, state_modes["UNKNOWN"])

            main_xds.append(
                self._main_xarray_factory(state_id, scan_index, scan_state, target)
            )

        # Generate subtable xarray datasets
        subtables = {
            "ANTENNA": self._antenna_xarray_factory(),
            "DATA_DESCRIPTION": self._ddid_xarray_factory(),
            "SPECTRAL_WINDOW": self._spw_xarray_factory(),
            "POLARIZATION": self._pol_xarray_factory(),
            "FEED": self._feed_xarray_factory(),
            "FIELD": self._field_xarray_factory(field_data),
            "SOURCE": self._source_xarray_factory(field_data),
            "OBSERVATION": self._observation_xarray_factory(),
            "STATE": self._state_xarray_factory(state_modes),
        }

        return main_xds, subtables
