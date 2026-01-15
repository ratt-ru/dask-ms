# Much of the subtable generation code is derived from
# https://github.com/ska-sa/katdal/blob/v0.22/katdal/ms_extra.py
# under the following license
#
# ################################################################################
# Copyright (c) 2011-2023, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

from functools import partial
from operator import getitem
import warnings
import dask.array as da
import numpy as np

from katdal.dataset import DataSet
from katdal.lazy_indexer import DaskLazyIndexer
from katpoint import Timestamp
import xarray

from daskms.constants import DASKMS_PARTITION_KEY
from daskms.experimental.katdal.constants import (
    GROUP_COLS,
    DATA_DESC_ID,
    TAG_TO_INTENT,
    EMPTY_PARTITION_SCHEMA,
)
from daskms.experimental.katdal.corr_products import corrprod_index
from daskms.experimental.katdal.transpose import transpose
from daskms.experimental.katdal.uvw import uvw_coords


def to_mjds(timestamp: Timestamp):
    """Converts a katpoint Timestamp to Modified Julian Date Seconds"""
    return timestamp.to_mjd() * 24 * 60 * 60


DEFAULT_TIME_CHUNKS = 100


class XArrayMSv2Facade:
    """Provides a simplified xarray Dataset view over a katdal dataset"""

    def __init__(
        self,
        dataset: DataSet,
        no_auto: bool = True,
        row_view: bool = True,
    ):
        self._dataset = dataset
        self._no_auto = no_auto
        self._row_view = row_view
        self._pols_to_use = ["HH", "HV", "VH", "VV"]
        # Reset the dataset selection
        self._dataset.select(reset="")
        self._cp_info = corrprod_index(dataset, self._pols_to_use, not no_auto)

    def transform_chunks(self, chunks: dict | list[dict] | None = None) -> list[dict]:
        if chunks is None:
            chunks = [{"time": DEFAULT_TIME_CHUNKS}]
        elif isinstance(chunks, dict):
            chunks = [chunks]
        elif not isinstance(chunks, list) and not all(
            isinstance(c, dict) for c in chunks
        ):
            raise TypeError(f"{chunks} must a dictionary or list of dictionaries")
        elif len(chunks) == 0:
            chunks.append([{"time": DEFAULT_TIME_CHUNKS}])

        xformed_chunks = []

        for ds_chunks in chunks:
            if not self._row_view:
                xformed_chunks.append(ds_chunks)
            else:
                # katdal's internal data shape is (time, chan, baseline*pol)
                # If chunking reasoning is row-based it's necessary to
                # derive a time based chunking from the row dimension
                # We cannot always exactly supply the requested number of rows,
                # as we always have to supply a multiple of the number of baselines
                row = ds_chunks.pop("row", DEFAULT_TIME_CHUNKS * self.nbl)
                # We need at least one timestamps worth of rows
                row = max(row, self.nbl)
                time = row // self.nbl
                ds_chunks["time"] = min(time, len(self._dataset.timestamps))
                xformed_chunks.append(ds_chunks)

        return xformed_chunks

    @property
    def cp_info(self):
        return self._cp_info

    @property
    def ntime(self):
        return len(self._dataset.timestamps)

    @property
    def na(self):
        return len(self._dataset.ants)

    @property
    def nbl(self):
        return self._cp_info.cp_index.shape[0]

    @property
    def npol(self):
        return self._cp_info.cp_index.shape[1]

    def _main_xarray_factory(
        self, field_id, state_id, scan_index, scan_state, target, chunks
    ):
        # Extract numpy and dask products
        dataset = self._dataset
        cp_info = self._cp_info
        time_utc = dataset.timestamps
        t_chunks, chan_chunks, cp_chunks = dataset.vis.dataset.chunks

        # Override time and channel chunking
        t_chunks = chunks.get("time", t_chunks)
        chan_chunks = chunks.get("chan", chan_chunks)

        # Modified Julian Date in Seconds
        time_mjds = np.asarray([to_mjds(t) for t in map(Timestamp, time_utc)])

        # Create a dask chunking transform
        rechunk = partial(da.rechunk, chunks=(t_chunks, chan_chunks, cp_chunks))

        # Transpose from (time, chan, corrprod) to (time, bl, chan, corr)
        cpi = cp_info.cp_index
        flag_transpose = partial(
            transpose,
            cp_index=cpi,
            data_type="flags",
            row=self._row_view,
        )
        weight_transpose = partial(
            transpose,
            cp_index=cpi,
            data_type="weights",
            row=self._row_view,
        )
        vis_transpose = partial(
            transpose,
            cp_index=cpi,
            data_type="vis",
            row=self._row_view,
        )

        flags = DaskLazyIndexer(dataset.flags, (), (rechunk, flag_transpose))
        weights = DaskLazyIndexer(dataset.weights, (), (rechunk, weight_transpose))
        vis = DaskLazyIndexer(dataset.vis, (), (rechunk, vis_transpose))

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

        # Better graph than da.broadcast_arrays
        bcast = da.blockwise(
            np.broadcast_arrays,
            ("time", "bl"),
            time,
            ("time", "bl"),
            ant1,
            ("time", "bl"),
            ant2,
            ("time", "bl"),
            align_arrays=False,
            adjust_chunks={"time": time.chunks[0], "bl": ant1.chunks[1]},
            meta=np.empty((0,) * 2, dtype=np.int32),
        )

        time = da.blockwise(
            getitem, ("time", "bl"), bcast, ("time", "bl"), 0, None, dtype=time.dtype
        )

        ant1 = da.blockwise(
            getitem, ("time", "bl"), bcast, ("time", "bl"), 1, None, dtype=ant1.dtype
        )

        ant2 = da.blockwise(
            getitem, ("time", "bl"), bcast, ("time", "bl"), 2, None, dtype=ant2.dtype
        )

        if self._row_view:
            primary_dims = ("row",)
            time = time.ravel()
            ant1 = ant1.ravel()
            ant2 = ant2.ravel()
        else:
            primary_dims = ("time", "baseline")

        data_vars = {
            # Primary indexing columns
            "TIME": (primary_dims, time),
            "ANTENNA1": (primary_dims, ant1),
            "ANTENNA2": (primary_dims, ant2),
            "FEED1": (primary_dims, da.zeros_like(ant1)),
            "FEED2": (primary_dims, da.zeros_like(ant1)),
            "STATE_ID": (primary_dims, da.full_like(ant1, state_id)),
            "ARRAY_ID": (primary_dims, da.zeros_like(ant1)),
            "OBSERVATION_ID": (primary_dims, da.zeros_like(ant1)),
            "PROCESSOR_ID": (primary_dims, da.ones_like(ant1)),
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
            "WEIGHT": (
                primary_dims + ("corr",),
                da.mean(weights.dataset, axis=len(primary_dims)),
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
            "SIGMA": (
                primary_dims + ("corr",),
                da.mean(weights.dataset**-0.5, axis=len(primary_dims)),
            ),
        }

        attrs = {
            DASKMS_PARTITION_KEY: tuple((c, "int32") for c in GROUP_COLS),
            "FIELD_ID": field_id,
            "DATA_DESC_ID": DATA_DESC_ID,
            "SCAN_NUMBER": scan_index,
        }

        assert (set(GROUP_COLS) & set(attrs)) == set(GROUP_COLS)

        return xarray.Dataset(data_vars, attrs=attrs)

    def _apply_subtable_grouping(self, datasets, group_cols, subtable):
        # Assign empty partition schemas to subtables
        if group_cols == ["__row__"]:
            return [ds.assign_attrs(EMPTY_PARTITION_SCHEMA) for ds in datasets]
        elif len(group_cols) == 0:
            return [
                xarray.concat(datasets, dim="row").assign_attrs(EMPTY_PARTITION_SCHEMA)
            ]
        else:
            raise ValueError(
                f"group_cols {group_cols} not supported "
                f"for subtable {subtable}. "
                f'Only [] or "__row__" is supported'
            )

    def _antenna_xarray_factory(self, group_cols):
        antennas = self._dataset.ants
        datasets = [
            xarray.Dataset(
                {
                    "NAME": (
                        "row",
                        np.asarray([a.name], dtype=object),
                    ),
                    "STATION": (
                        "row",
                        np.asarray([a.name], dtype=object),
                    ),
                    "POSITION": (
                        ("row", "xyz"),
                        np.asarray([a.position_ecef]),
                    ),
                    "OFFSET": (("row", "xyz"), np.zeros((1, 3))),
                    "DISH_DIAMETER": (
                        "row",
                        np.asarray([a.diameter]),
                    ),
                    "MOUNT": ("row", np.array(["ALT-AZ"], dtype=object)),
                    "TYPE": ("row", np.array(["GROUND-BASED"], dtype=object)),
                    "FLAG_ROW": ("row", np.zeros(1, dtype=np.int32)),
                }
            )
            for a in antennas
        ]

        return self._apply_subtable_grouping(datasets, group_cols, "ANTENNA")

    def _spw_xarray_factory(self, group_cols):
        def ref_freq(chan_freqs):
            return chan_freqs[len(chan_freqs) // 2].astype(np.float64)

        datasets = [
            xarray.Dataset(
                {
                    "NUM_CHAN": (("row",), np.array([spw.num_chans], dtype=np.int32)),
                    "CHAN_FREQ": (("row", "chan"), spw.channel_freqs[np.newaxis, :]),
                    "RESOLUTION": (("row", "chan"), spw.channel_freqs[np.newaxis, :]),
                    "CHAN_WIDTH": (
                        ("row", "chan"),
                        np.full_like(
                            spw.channel_freqs[np.newaxis, :], spw.channel_width
                        ),
                    ),
                    "EFFECTIVE_BW": (
                        ("row", "chan"),
                        np.full_like(
                            spw.channel_freqs[np.newaxis, :], spw.channel_width
                        ),
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

        return self._apply_subtable_grouping(datasets, group_cols, "SPECTRAL_WINDOW")

    def _pol_xarray_factory(self, group_cols):
        pol_num = {"H": 0, "V": 1}
        # MeerKAT only has linear feeds, these map to
        # CASA ["XX", "XY", "YX", "YY"]
        pol_types = {"HH": 9, "HV": 10, "VH": 11, "VV": 12}
        datasets = [
            xarray.Dataset(
                {
                    "NUM_CORR": (
                        "row",
                        np.array([len(self._pols_to_use)], dtype=np.int32),
                    ),
                    "CORR_PRODUCT": (
                        ("row", "corr", "corrprod_idx"),
                        np.array(
                            [
                                [
                                    [pol_num[p[0]], pol_num[p[1]]]
                                    for p in self._pols_to_use
                                ]
                            ],
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
        ]

        return self._apply_subtable_grouping(datasets, group_cols, "SPECTRAL_WINDOW")

    def _ddid_xarray_factory(self, group_cols):
        datasets = [
            xarray.Dataset(
                {
                    "SPECTRAL_WINDOW_ID": ("row", np.zeros(1, dtype=np.int32)),
                    "POLARIZATION_ID": ("row", np.zeros(1, dtype=np.int32)),
                    "FLAG_ROW": ("row", np.zeros(1, dtype=np.int32)),
                }
            )
        ]
        return self._apply_subtable_grouping(datasets, group_cols, "DATA_DESCRIPTION")

    def _feed_xarray_factory(self, group_cols):
        NRECEPTORS = 2

        datasets = [
            xarray.Dataset(
                {
                    # ID of antenna in this array (integer)
                    "ANTENNA_ID": ("row", np.array([f], dtype=np.int32)),
                    # Id for BEAM model (integer)
                    "BEAM_ID": ("row", np.ones(1, dtype=np.int32)),
                    # Beam position offset (on sky but in antenna reference frame): (double, 2-dim)
                    "BEAM_OFFSET": (
                        ("row", "receptors", "radec"),
                        np.zeros((1, 2, 2), dtype=np.float64),
                    ),
                    # Feed id (integer)
                    "FEED_ID": ("row", np.zeros(1, dtype=np.int32)),
                    # Interval for which this set of parameters is accurate (double)
                    "INTERVAL": ("row", np.zeros(1, dtype=np.float64)),
                    # Number of receptors on this feed (probably 1 or 2) (integer)
                    "NUM_RECEPTORS": (
                        "row",
                        np.full(1, NRECEPTORS, dtype=np.int32),
                    ),
                    # Type of polarisation to which a given RECEPTOR responds (string, 1-dim)
                    "POLARIZATION_TYPE": (
                        ("row", "receptors"),
                        np.array([["X", "Y"]], dtype=object),
                    ),
                    # D-matrix i.e. leakage between two receptors (complex, 2-dim)
                    "POL_RESPONSE": (
                        ("row", "receptors", "receptors-2"),
                        np.array([np.eye(2, dtype=np.complex64)]),
                    ),
                    # Position of feed relative to feed reference position (double, 1-dim, shape=(3,))
                    "POSITION": (("row", "xyz"), np.zeros((1, 3), np.float64)),
                    # The reference angle for polarisation (double, 1-dim). A parallactic angle of
                    # 0 means that V is aligned to x (celestial North), but we are mapping H to x
                    # so we have to correct with a -90 degree rotation.
                    "RECEPTOR_ANGLE": (
                        ("row", "receptors"),
                        np.full((1, NRECEPTORS), -np.pi / 2, dtype=np.float64),
                    ),
                    # ID for this spectral window setup (integer)
                    "SPECTRAL_WINDOW_ID": ("row", np.full(1, -1, dtype=np.int32)),
                    # Midpoint of time for which this set of parameters is accurate (double)
                    "TIME": ("row", np.zeros(1, dtype=np.float64)),
                }
            )
            for f in range(len(self._dataset.ants))
        ]

        return self._apply_subtable_grouping(datasets, group_cols, "FEED")

    def _field_xarray_factory(self, field_data, group_cols):
        datasets = [
            xarray.Dataset(
                {
                    "NAME": ("row", np.array([target.name], object)),
                    "CODE": ("row", np.array(["T"], object)),
                    "SOURCE_ID": ("row", np.array([field_id], dtype=np.int32)),
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
            for field_id, time, target, radec in field_data.values()
        ]

        return self._apply_subtable_grouping(datasets, group_cols, "FIELD")

    def _source_xarray_factory(self, field_data, group_cols):
        field_ids, times, targets, radecs = zip(*(field_data.values()))
        times = np.array(times, dtype=np.float64)
        datasets = [
            xarray.Dataset(
                {
                    "NAME": ("row", np.array([target.name], dtype=object)),
                    "SOURCE_ID": ("row", np.array([f], dtype=np.int32)),
                    "PROPER_MOTION": (
                        ("row", "radec-per-sec"),
                        np.zeros((1, 2), dtype=np.float32),
                    ),
                    "CALIBRATION_GROUP": ("row", np.full(1, -1, dtype=np.int32)),
                    "DIRECTION": (("row", "radec"), np.array([radec])),
                    "TIME": ("row", np.array([time])),
                    "NUM_LINES": ("row", np.ones(1, dtype=np.int32)),
                    "REST_FREQUENCY": (
                        ("row", "lines"),
                        np.zeros((1, 1), dtype=np.float64),
                    ),
                }
            )
            for f, time, target, radec in zip(field_ids, times, targets, radecs)
        ]
        return self._apply_subtable_grouping(datasets, group_cols, "SOURCE")

    def _state_xarray_factory(self, state_modes, group_cols):
        state_ids, modes = zip(*sorted((i, m) for m, i in state_modes.items()))
        datasets = [
            xarray.Dataset(
                {
                    "SIG": (("row",), np.ones(1, dtype=np.uint8)),
                    "REF": (("row",), np.zeros(1, dtype=np.uint8)),
                    "CAL": (("row",), np.zeros(1, dtype=np.float64)),
                    "LOAD": (("row",), np.zeros(1, dtype=np.float64)),
                    "SUB_SCAN": (("row",), np.zeros(1, dtype=np.int32)),
                    "OBS_MODE": (("row",), np.array([mode], dtype=object)),
                    "FLAG_ROW": (("row",), np.zeros(1, dtype=np.int32)),
                }
            )
            for s, mode in zip(state_ids, modes)
        ]

        return self._apply_subtable_grouping(datasets, group_cols, "STATE")

    def _observation_xarray_factory(self, group_cols):
        ds = self._dataset
        start, end = [to_mjds(t) for t in [ds.start_time, ds.end_time]]
        datasets = [
            xarray.Dataset(
                {
                    "OBSERVER": ("row", np.array([ds.observer], dtype=object)),
                    "PROJECT": ("row", np.array([ds.experiment_id], dtype=object)),
                    "LOG": (
                        ("row", "extra"),
                        np.array([["unavailable"]], dtype=object),
                    ),
                    "SCHEDULE": (
                        ("row", "extra"),
                        np.array([["unavailable"]], dtype=object),
                    ),
                    "SCHEDULE_TYPE": ("row", np.array(["unknown"], dtype=object)),
                    "TELESCOPE_NAME": ("row", np.array(["MeerKAT"], dtype=object)),
                    "TIME_RANGE": (("row", "extent"), np.array([[start, end]])),
                    "FLAG_ROW": ("row", np.zeros(1, np.uint8)),
                }
            )
        ]
        return self._apply_subtable_grouping(datasets, group_cols, "OBSERVATION")

    def xarray_datasets(self, subtable_kw=None, **kw):
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
        field_data = {}
        UNKNOWN_STATE_ID = 0
        state_modes = {"UNKNOWN": UNKNOWN_STATE_ID}

        main_chunks = self.transform_chunks(kw.get("chunks"))

        if (main_group_cols := kw.get("group_cols", GROUP_COLS)) != GROUP_COLS:
            warnings.warn(
                f"Supplied {main_group_cols} does not match "
                f"the hard-coded katdal grouping columns "
                f"{GROUP_COLS} for the main dataset  and will be ignored"
            )

        # Generate MAIN table xarray partition datasets
        for i, (scan_index, scan_state, target) in enumerate(self._dataset.scans()):
            if scan_state == "slew":
                continue

            # Retrieve existing field data, or create
            try:
                field_id, _, _, _ = field_data[target.name]
            except KeyError:
                field_id = len(field_data)
                time_origin = Timestamp(self._dataset.timestamps[0])
                field_data[target.name] = (
                    field_id,
                    to_mjds(time_origin),
                    target,
                    target.radec(time_origin),
                )

            # Create or retrieve the state_id associated
            # with the tags of the current source
            state_tag = ",".join(
                TAG_TO_INTENT[tag] for tag in target.tags if tag in TAG_TO_INTENT
            )
            if state_tag and state_tag not in state_modes:
                state_modes[state_tag] = len(state_modes)
            state_id = state_modes.get(state_tag, UNKNOWN_STATE_ID)

            try:
                chunks = main_chunks[i]
            except IndexError:
                chunks = main_chunks[-1]

            main_xds.append(
                self._main_xarray_factory(
                    field_id, state_id, scan_index, scan_state, target, chunks
                )
            )

        subtable_factory_map = {
            "ANTENNA": self._antenna_xarray_factory,
            "DATA_DESCRIPTION": self._ddid_xarray_factory,
            "SPECTRAL_WINDOW": self._spw_xarray_factory,
            "POLARIZATION": self._pol_xarray_factory,
            "FEED": self._feed_xarray_factory,
            "FIELD": partial(self._field_xarray_factory, field_data),
            "SOURCE": partial(self._source_xarray_factory, field_data),
            "OBSERVATION": self._observation_xarray_factory,
            "STATE": partial(self._state_xarray_factory, state_modes),
        }

        # Generate subtable xarray datasets
        s_kw = subtable_kw if subtable_kw is not None else {}

        subtables = {
            subtable: factory(**{"group_cols": []} | s_kw.get(subtable, {}))
            for subtable, factory in subtable_factory_map.items()
        }

        return main_xds, subtables
