import dask
import katdal
from katdal.dataset import DataSet

from daskms.experimental.katdal.msv2_facade import XarrayMSV2Facade
from daskms.experimental.zarr import xds_to_zarr


def katdal_import(url: str, out_store: str, auto_corrs: bool = True):
    if isinstance(url, str):
        dataset = katdal.open(url)
    elif isinstance(url, DataSet):
        dataset = url
    else:
        raise TypeError(f"{url} must be a string or a katdal DataSet")

    facade = XarrayMSV2Facade(dataset, auto_corrs=auto_corrs)
    main_xds, subtable_xds = facade.xarray_datasets()

    writes = [
        xds_to_zarr(main_xds, out_store),
        *(xds_to_zarr(ds, f"{out_store}::{k}") for k, ds in subtable_xds.items()),
    ]

    dask.compute(writes)
