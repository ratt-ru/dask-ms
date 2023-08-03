from daskms import xds_from_storage_ms, xds_to_storage_table
from daskms.fsspec_store import DaskMSStore
from daskms.utils import requires
from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr

from collections.abc import Iterable

try:
    import xarray
except ImportError as e:
    xarray_import_error = e
else:
    xarray_import_error = None


@requires("pip install dask-ms[xarray] for xarray support", xarray_import_error)
def xds_from_merged_storage(stores, **kwargs):

    if not isinstance(stores, Iterable):
        stores = [stores]

    storage_options = kwargs.pop("storage_options", {})

    lxdsl = []

    for store in stores:
        lxdsl.append(xds_from_storage_ms(store, storage_options=storage_options, **kwargs))

    assert len({len(xdsl) for xdsl in lxdsl}) == 1, (
        "xds_from_merged_storage was unable to merge datasets dynamically "
        "due to conflicting lengths on the intermediary lists of datasets."
    )

    return [xarray.merge(xdss) for xdss in zip(*lxdsl)]


def xds_from_proxy(store, **kwargs):

    xdsl = xds_from_storage_ms(store, **kwargs)

    parent_urls = {xds.attrs.get("__dask_ms_parent_url__", None) for xds in xdsl}

    assert len(parent_urls) == 1, (
        "Proxy has more than one parent - this is not supported."
    )

    parent_url = parent_urls.pop()

    if parent_url:

        if not isinstance(parent_url, DaskMSStore):
            store = DaskMSStore(parent_url)

        xdsl_nested = xds_from_proxy(store, **kwargs)
    else:
        return [xdsl]

    return [xdsl, *xdsl_nested]



def xds_to_proxy(xds, store, parent, **kwargs):

    if not isinstance(parent, DaskMSStore):
        parent = DaskMSStore(parent)

    xds = [x.assign_attrs({"__dask_ms_parent_url__": parent.url}) for x in xds]

    return xds_to_zarr(xds, store, **kwargs)
