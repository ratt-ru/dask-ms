from __future__ import annotations
import logging
import os
import urllib
import warnings

import dask

from daskms.experimental.katdal.constants import GROUP_COLS
from daskms.fsspec_store import DaskMSStore
from daskms.utils import requires, promote_columns
from daskms.multiton import MultitonMetaclass

log = logging.getLogger(__file__)

try:
    import katdal
    from katdal.dataset import DataSet

    from daskms.experimental.katdal.msv2_facade import XArrayMSv2Facade
    from daskms.experimental.zarr import xds_to_zarr
except ImportError as e:
    import_error = e
else:
    import_error = None


class FacadeMultiton(metaclass=MultitonMetaclass):
    """Apply some caching to facades"""

    @staticmethod
    def from_args(url: str, applycal: str = "", no_auto: bool = True, **kw):
        katdal_dataset = katdal.open(url, applycal=applycal, **kw)
        return XArrayMSv2Facade(katdal_dataset, no_auto=no_auto)


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
def xds_from_katdal(
    url_or_dataset: str | DataSet,
    applycal: str = "",
    no_auto: bool = True,
    group_cols: list | None = None,
    chunks: list[dict] | dict | None = None,
    **kwargs,
):
    if isinstance(url_or_dataset, DataSet):
        base_url = url_or_dataset
    elif isinstance(url_or_dataset, str):
        try:
            base_url, subtable = url_or_dataset.split("::", 1)
        except ValueError:
            base_url = url_or_dataset
            subtable = ""
    else:
        raise TypeError(
            f"url_or_dataset {type(url_or_dataset)} must be a str or Dataset"
        )

    # Ignore these arguments
    for ignored_argument in ["columns", "index_cols", "table_schema", "taql_where"]:
        if (arg := kwargs.pop(ignored_argument, None)) is not None:
            warnings.warn(
                f"{ignored_argument} {arg} argument to xds_from_katdal ignored",
                UserWarning,
            )

    # This changes the return signature -- fail
    for failed_argument in ["table_keywords", "column_keywords", "table_proxy"]:
        if (arg := kwargs.pop(failed_argument, None)) is not None:
            raise NotImplementedError(
                f"{failed_argument} {arg} argument to xds_from_katdal is not supported"
            )

    if subtable:
        main_kw = {}

        if chunks:
            raise NotImplementedError(f"chunking {subtable} subtable")

        subtable_kw = {subtable: {"group_cols": promote_columns(group_cols, [])}}
    else:
        main_kw = {
            "chunks": chunks,
            "group_cols": promote_columns(group_cols, GROUP_COLS),
        }
        subtable_kw = {}

    facade = FacadeMultiton(
        FacadeMultiton.from_args, base_url, applycal, no_auto, **kwargs
    )
    main_xds, subtable_xds = facade.instance.xarray_datasets(
        subtable_kw=subtable_kw, **main_kw
    )

    if subtable:
        return subtable_xds[subtable]

    return main_xds


@requires("pip install dask-ms[katdal]", import_error)
def katdal_import(url: str, out_store: str, no_auto: bool, applycal: str, chunks: dict):
    if isinstance(url, str):
        dataset = katdal.open(url, appycal=applycal)
    elif isinstance(url, DataSet):
        dataset = url
    else:
        raise TypeError(f"{url} must be a string or a katdal DataSet")

    facade = FacadeMultiton(FacadeMultiton.from_args, dataset, applycal, no_auto)
    main_xds, subtable_xds = facade.instance.xarray_datasets(chunks=chunks)

    if not out_store:
        out_store = default_output_name(url)

    out_store = DaskMSStore(out_store)
    if out_store.exists():
        warnings.warn(f"Removing previously existing {out_store}", UserWarning)
        out_store.rm("", recursive=True)

    writes = [
        xds_to_zarr(main_xds, out_store),
        *(xds_to_zarr(ds, f"{out_store}::{k}") for k, ds in subtable_xds.items()),
    ]

    dask.compute(writes)
