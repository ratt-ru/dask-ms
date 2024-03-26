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

    def _xarray_factory(self, scan_index, scan_state, target):
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
        xds = [self._xarray_factory(*scan_data) for scan_data in self._dataset.scans()]
        yield xarray.concat(xds, dim="row" if self._row_view else "time")
