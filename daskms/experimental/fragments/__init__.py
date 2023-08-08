from daskms import xds_from_storage_ms, xds_from_storage_table
from daskms.fsspec_store import DaskMSStore
from daskms.utils import requires
from daskms.experimental.zarr import xds_to_zarr
from daskms.fsspec_store import UnknownStoreTypeError

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
        lxdsl.append(
            xds_from_storage_ms(store, storage_options=storage_options, **kwargs)
        )

    assert len({len(xdsl) for xdsl in lxdsl}) == 1, (
        "xds_from_merged_storage was unable to merge datasets dynamically "
        "due to conflicting lengths on the intermediary lists of datasets."
    )

    return [xarray.merge(xdss) for xdss in zip(*lxdsl)]


def _xds_from_ms_fragment(store, **kwargs):
    xdsl = xds_from_storage_ms(store, **kwargs)

    parent_urls = {xds.attrs.get("__dask_ms_parent_url__", None) for xds in xdsl}

    assert (
        len(parent_urls) == 1
    ), "Proxy has more than one parent - this is not supported."

    parent_url = parent_urls.pop()

    if parent_url:
        if not isinstance(parent_url, DaskMSStore):
            store = DaskMSStore(parent_url)

        xdsl_nested = _xds_from_ms_fragment(store, **kwargs)
    else:
        return [xdsl]

    return [*xdsl_nested, xdsl]


def merge_via_assign(xdsl):
    composite_xds = xdsl[0]

    partition_keys = [p[0] for p in composite_xds.__daskms_partition_schema__]

    for xds in xdsl[1:]:
        if not all(xds.attrs[k] == composite_xds.attrs[k] for k in partition_keys):
            raise ValueError(
                "merge_via_assign failed due to conflicting partition keys."
                "This usually means you are attempting to merge datasets "
                "which were constructed with different group_cols arguments."
            )

        composite_xds = composite_xds.assign(xds.data_vars)
        composite_xds = composite_xds.assign_attrs(xds.attrs)

    return composite_xds


def xds_from_ms_fragment(store, **kwargs):
    """
    Creates a list of xarray datasets representing the contents a composite
    Measurement Set. The resulting list of datasets will consist of some root
    dataset with any newer variables populated from the child fragments. It
    defers to :func:`xds_from_storage_ms`, which should be consulted
    for more information.

    Parameters
    ----------
    store : str or DaskMSStore
        Store or string of the child fragment of interest.
    columns : tuple or list, optional
        Columns present on the resulting dataset.
        Defaults to all if ``None``.
    index_cols  : tuple or list, optional
        Sequence of indexing columns.
        Defaults to :code:`%(indices)s`
    group_cols  : tuple or list, optional
        Sequence of grouping columns.
        Defaults to :code:`%(groups)s`
    **kwargs : optional

    Returns
    -------
    datasets : list of :class:`xarray.Dataset`
        xarray datasets for each group
    """

    lxdsl = _xds_from_ms_fragment(store, **kwargs)

    return [merge_via_assign(xdss) for xdss in zip(*lxdsl)]


def xds_to_ms_fragment(xds, store, parent, **kwargs):
    """
    Generates a list of Datasets representing write operations from the
    specified arrays in :class:`xarray.Dataset`'s into a child fragment
    dataset.

    Parameters
    ----------
    xds : :class:`xarray.Dataset` or list of :class:`xarray.Dataset`
        dataset(s) containing the specified columns. If a list of datasets
        is provided, the concatenation of the columns in
        sequential datasets will be written.
    store : str or DaskMSStore
        Store or string which determines the location to which the child
        fragment will be written.
    parent : str or DaskMSStore
        Store or sting corresponding to the parent dataset. Can be either
        point to either a root dataset or another child fragment.

    **kwargs : optional arguments. See :func:`xds_to_table`.

    Returns
    -------
    write_datasets : list of :class:`xarray.Dataset`
        Datasets containing arrays representing write operations
        into a CASA Table
    table_proxy : :class:`daskms.TableProxy`, optional
        The Table Proxy associated with the datasets
    """

    if not isinstance(parent, DaskMSStore):
        parent = DaskMSStore(parent)

    xds = [x.assign_attrs({"__dask_ms_parent_url__": parent.url}) for x in xds]

    return xds_to_zarr(xds, store, **kwargs)


def _xds_from_table_fragment(store, **kwargs):
    try:
        # Try to open the store. However, as we are reading from a fragment,
        # the subtable may not exist in the child.
        xdsl = xds_from_storage_table(store, **kwargs)
        required = True
    except UnknownStoreTypeError:
        # NOTE: We don't pass kwargs - the only purpose of this read is to
        # grab the parent urls (if they exist).
        xdsl = xds_from_storage_table(DaskMSStore(store.root))
        required = False

    subtable = store.table

    parent_urls = {xds.attrs.get("__dask_ms_parent_url__", None) for xds in xdsl}

    assert (
        len(parent_urls) == 1
    ), "Proxy has more than one parent - this is not supported."

    parent_url = parent_urls.pop()

    if parent_url:
        if not isinstance(parent_url, DaskMSStore):
            store = DaskMSStore(parent_url).subtable_store(subtable)

        xdsl_nested = _xds_from_table_fragment(store, **kwargs)
    else:
        return [xdsl]

    if required:
        return [*xdsl_nested, xdsl]
    else:
        return [*xdsl_nested]


def xds_from_table_fragment(store, **kwargs):
    """
    Creates a list of xarray datasets representing the contents a composite
    Measurement Set. The resulting list of datasets will consist of some root
    dataset with any newer variables populated from the child fragments. It
    defers to :func:`xds_from_storage_ms`, which should be consulted
    for more information.

    Parameters
    ----------
    store : str or DaskMSStore
        Store or string of the child fragment of interest.
    columns : tuple or list, optional
        Columns present on the resulting dataset.
        Defaults to all if ``None``.
    index_cols  : tuple or list, optional
        Sequence of indexing columns.
        Defaults to :code:`%(indices)s`
    group_cols  : tuple or list, optional
        Sequence of grouping columns.
        Defaults to :code:`%(groups)s`
    **kwargs : optional

    Returns
    -------
    datasets : list of :class:`xarray.Dataset`
        xarray datasets for each group
    """

    if not isinstance(store, DaskMSStore):
        store = DaskMSStore(store)

    lxdsl = _xds_from_table_fragment(store, **kwargs)

    return [merge_via_assign(xdss) for xdss in zip(*lxdsl)]


def xds_to_table_fragment(xds, store, parent, **kwargs):
    """
    Generates a list of Datasets representing write operations from the
    specified arrays in :class:`xarray.Dataset`'s into a child fragment
    dataset.

    Parameters
    ----------
    xds : :class:`xarray.Dataset` or list of :class:`xarray.Dataset`
        dataset(s) containing the specified columns. If a list of datasets
        is provided, the concatenation of the columns in
        sequential datasets will be written.
    store : str or DaskMSStore
        Store or string which determines the location to which the child
        fragment will be written.
    parent : str or DaskMSStore
        Store or sting corresponding to the parent dataset. Can be either
        point to either a root dataset or another child fragment.

    **kwargs : optional arguments. See :func:`xds_to_table`.

    Returns
    -------
    write_datasets : list of :class:`xarray.Dataset`
        Datasets containing arrays representing write operations
        into a CASA Table
    table_proxy : :class:`daskms.TableProxy`, optional
        The Table Proxy associated with the datasets
    """

    if not isinstance(parent, DaskMSStore):
        parent = DaskMSStore(parent)

    xds = [x.assign_attrs({"__dask_ms_parent_url__": parent.url}) for x in xds]

    return xds_to_zarr(xds, store, **kwargs)
