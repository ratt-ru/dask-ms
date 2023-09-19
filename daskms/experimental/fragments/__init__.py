from daskms import xds_from_storage_table
from daskms.fsspec_store import DaskMSStore
from daskms.utils import requires
from daskms.experimental.zarr import xds_to_zarr
from zarr.errors import GroupNotFoundError

try:
    import xarray  # noqa
except ImportError as e:
    xarray_import_error = e
else:
    xarray_import_error = None

xarray_import_msg = "pip install dask-ms[xarray] for xarray support"


def get_ancestry(store, only_required=True):
    """Produces a list of stores needed to reconstruct the dataset at store."""

    fragments = []

    if not isinstance(store, DaskMSStore):
        # TODO: Where, when and how should we pass storage options?
        store = DaskMSStore(store)

    while True:
        root_store = DaskMSStore(store.root_url)

        if store.exists():
            try:
                # Store exists and can be read.
                xdsl = xds_from_storage_table(store, columns=[])
                fragments += [store]
            except GroupNotFoundError:
                # Store exists, but cannot be read. We may be dealing with
                # a subtable only fragment. NOTE: This assumes that all
                # subtables in a fragment have the same parent, so we don't
                # care which subtable we read.
                subtable_name = store.subdirectories()[0].rsplit("/")[-1]
                subtable_store = store.subtable_store(subtable_name)
                xdsl = xds_from_storage_table(subtable_store, columns=[])
                fragments += [] if only_required else [subtable_store]
        elif root_store.exists():
            # Root store exists and can be read.
            xdsl = xds_from_storage_table(root_store, columns=[])
            fragments += [] if only_required else [root_store]
        else:
            raise FileNotFoundError(f"No root/fragment found at {store}.")

        subtable = store.table

        parent_urls = {xds.attrs.get("__dask_ms_parent_url__", None) for xds in xdsl}

        assert (
            len(parent_urls) == 1
        ), "Fragment has more than one parent - this is not supported."

        parent_url = parent_urls.pop()

        if parent_url:
            if not isinstance(parent_url, DaskMSStore):
                # TODO: Where, when and how should we pass storage options?
                store = DaskMSStore(parent_url).subtable_store(subtable or "")
        else:
            if store.table and not any(f.table for f in fragments):
                # If we are attempting to open a subtable, we don't know if
                # it exists until we have traversed the entire ancestry.
                raise FileNotFoundError(
                    f"{store.table} subtable was not found in parents."
                )

            return fragments[::-1]  # Flip so that the root appears first.


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
    root_partition_keys = {p[0] for p in root_schema}

    consolidated_xds = root_xds  # Will be replaced in the loop.

    for xds in xdsl[1:]:
        xds_schema = xds.__daskms_partition_schema__
        xds_partition_keys = {p[0] for p in xds_schema}

        if root_partition_keys.symmetric_difference(xds_partition_keys):
            raise ValueError(
                f"consolidate failed due to conflicting partition keys. "
                f"This usually means the partition keys of the fragments "
                f"are inconsistent with the current group_cols argument. "
                f"Current group_cols produces {root_partition_keys} but "
                f"the fragment has {xds_partition_keys}."
            )

        consolidated_xds = consolidated_xds.assign(xds.data_vars)

    return consolidated_xds


@requires(xarray_import_msg, xarray_import_error)
def xds_from_ms_fragment(store, **kwargs):
    """
    Creates a list of xarray datasets representing the contents a composite
    Measurement Set. The resulting list of datasets will consist of some root
    dataset with any newer variables populated from the child fragments. It
    defers to :func:`xds_from_table_fragment`, which should be consulted
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

    return xds_from_table_fragment(store, **kwargs)


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

    ancestors = get_ancestry(store)

    lxdsl = [xds_from_storage_table(s, **kwargs) for s in ancestors]

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

    # TODO: Where, when and how should we pass storage options?
    if not isinstance(store, DaskMSStore):
        store = DaskMSStore(store)

    if parent == store:
        raise ValueError(
            "store and parent arguments identical in xds_to_table_fragment. "
            "This is unsupported i.e. a fragment cannot be its own parent. "
        )

    xds = [x.assign_attrs({"__dask_ms_parent_url__": parent.url}) for x in xds]

    return xds_to_zarr(xds, store, **kwargs)
