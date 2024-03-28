import logging
import os
import urllib

import dask

from daskms.fsspec_store import DaskMSStore
from daskms.utils import requires

log = logging.getLogger(__file__)

try:
    import katdal
    from katdal.dataset import DataSet

    from daskms.experimental.katdal.msv2_facade import XarrayMSV2Facade
    from daskms.experimental.zarr import xds_to_zarr
except ImportError as e:
    import_error = e
else:
    import_error = None


def default_output_name(url):
    url_parts = urllib.parse.urlparse(url, scheme="file")
    # Create zarr dataset in current working directory (strip off directories)
    dataset_filename = os.path.basename(url_parts.path)
    # Get rid of the ".full" bit on RDB files (it's the same dataset)
    full_rdb_ext = ".full.rdb"
    if dataset_filename.endswith(full_rdb_ext):
        dataset_basename = dataset_filename[: -len(full_rdb_ext)]
    else:
        dataset_basename = os.path.splitext(dataset_filename)[0]
    return f"{dataset_basename}.zarr"


@requires("pip install dask-ms[katdal]", import_error)
def katdal_import(url: str, out_store: str, no_auto: bool, applycal: str, chunks: dict):
    if isinstance(url, str):
        dataset = katdal.open(url, appycal=applycal)
    elif isinstance(url, DataSet):
        dataset = url
    else:
        raise TypeError(f"{url} must be a string or a katdal DataSet")

    facade = XarrayMSV2Facade(dataset, no_auto=no_auto, chunks=chunks)
    main_xds, subtable_xds = facade.xarray_datasets()

    if not out_store:
        out_store = default_output_name(url)

    out_store = DaskMSStore(out_store)
    if out_store.exists():
        log.warn("Removing previously existing %s", out_store)
        out_store.rm("", recursive=True)

    writes = [
        xds_to_zarr(main_xds, out_store),
        *(xds_to_zarr(ds, f"{out_store}::{k}") for k, ds in subtable_xds.items()),
    ]

    dask.compute(writes)
