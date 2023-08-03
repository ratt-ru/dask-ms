from daskms import xds_from_storage_ms
from daskms.utils import requires

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
