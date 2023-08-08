from daskms import xds_from_storage_ms, xds_from_storage_table
from daskms.fsspec_store import DaskMSStore
from daskms.utils import requires
from daskms.experimental.zarr import xds_to_zarr
from daskms.fsspec_store import UnknownStoreTypeError

try:
    import xarray  # noqa
except ImportError as e:
    xarray_import_error = e
else:
    xarray_import_error = None

xarray_import_msg = "pip install dask-ms[xarray] for xarray support"


@requires(xarray_import_msg, xarray_import_error)
def consolidate(xdsl):
    """
    Consolidates a list of xarray datasets by assigning data variables.
    Priority is determined by the position within the list, with elements at
    the end of the list having higher priority than those at the start. The
    primary purpose of this function is the construction of a consolidated
    dataset from a root and deltas (fragments).

    Parameters
    ----------
    xdsl : tuple or list
        Tuple or list of :class:`xarray.Dataset` objects to consolidate.

    Returns
    -------
    consolidated_xds : :class:`xarray.Dataset`
        A single :class:`xarray.Dataset`.
    """

    root_xds = xdsl[0]  # First element is the root for this operation.

    root_schema = root_xds.__daskms_partition_schema__

    partition_keys = [p[0] for p in root_schema]

    consolidated_xds = root_xds  # Will be replaced in the loop.

    for xds in xdsl[1:]:
        if not all(xds.attrs[k] == root_xds.attrs[k] for k in partition_keys):
            raise ValueError(
                "consolidate failed due to conflicting partition keys."
                "This usually means you are attempting to merge datasets "
                "which were constructed with different group_cols arguments."
            )

        consolidated_xds = consolidated_xds.assign(xds.data_vars)
        # NOTE: Assigning the fragment's attributes may be unnecessary.
        consolidated_xds = consolidated_xds.assign_attrs(xds.attrs)

    return consolidated_xds


@requires(xarray_import_msg, xarray_import_error)
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


@requires(xarray_import_msg, xarray_import_error)
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

    return [consolidate(xdss) for xdss in zip(*lxdsl)]


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


@requires(xarray_import_msg, xarray_import_error)
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

    # TODO: Where, when and how should we pass storage options?
    if not isinstance(store, DaskMSStore):
        store = DaskMSStore(store)

    lxdsl = _xds_from_table_fragment(store, **kwargs)

    return [consolidate(xdss) for xdss in zip(*lxdsl)]


@requires(xarray_import_msg, xarray_import_error)
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

    # TODO: Where, when and how should we pass storage options?
    if not isinstance(parent, DaskMSStore):
        parent = DaskMSStore(parent)

    xds = [x.assign_attrs({"__dask_ms_parent_url__": parent.url}) for x in xds]

    return xds_to_zarr(xds, store, **kwargs)
