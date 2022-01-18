# -*- coding: utf-8 -*-

import logging

from daskms.fsspec_store import DaskMSStore
from daskms.table_proxy import TableProxy
from daskms.reads import DatasetFactory
from daskms.writes import write_datasets
from daskms.utils import promote_columns

_DEFAULT_INDEX_COLUMNS = []
_DEFAULT_GROUP_COLUMNS = ["FIELD_ID", "DATA_DESC_ID"]

log = logging.getLogger(__name__)


def xds_to_table(xds, table_name, columns="ALL", descriptor=None,
                 table_keywords=None, column_keywords=None,
                 table_proxy=False):
    """
    Generates a list of Datasets representing a write operations from the
    specified arrays in :class:`xarray.Dataset`'s into
    the CASA table columns specified by ``table_name`` and ``columns``.
    This is lazy operation -- it is only execute when a :meth:`dask.compute`
    or :meth:`xarray.Dataset.compute` method is called.

    Parameters
    ----------
    xds : :class:`xarray.Dataset` or list of :class:`xarray.Dataset`
        dataset(s) containing the specified columns. If a list of datasets
        is provided, the concatenation of the columns in
        sequential datasets will be written.

    table_name : str
        CASA table path

    columns : tuple or list or "ALL"
        list of column names to write to the table.

        "ALL" is a special marker which specifies that all columns
        should be written. If you wish to write an "ALL" array to
        a column, use :code:`columns=['ALL']`

    descriptor : None or \
        :class:`~daskms.descriptors.builder.AbstractBuilderFactory` or \
        str

        A class describing how CASA table descriptors and data managers
        are constructors. Some defaults are available such
        as `ms` and `ms_subtable`.

        If None, defaults are used.

    table_keywords : dict, optional
        Dictionary of table keywords to add to existing keywords.
        The operation is performed immediately, not lazily.

    column_keywords : dict, optional
        Dictionary of :code:`{column: keywords}` to add to existing
        column keywords.
        The operation is performed immediately, not lazily.

    table_proxy : {False, True}
        If True returns the table_proxy

    Returns
    -------
    write_datasets : list of :class:`xarray.Dataset`
        Datasets containing arrays representing write operations
        into a CASA Table
    table_proxy : :class:`daskms.TableProxy`, optional
        The Table Proxy associated with the datasets
    """
    if isinstance(table_name, DaskMSStore):
        table_name = table_name.casa_path()
    else:
        table_name = DaskMSStore(table_name).casa_path()

    # Promote dataset to a list
    if not isinstance(xds, (tuple, list)):
        xds = [xds]

    if not isinstance(columns, (tuple, list)):
        if columns != "ALL":
            columns = [columns]

    # Write the datasets
    out_ds = write_datasets(table_name, xds, columns,
                            descriptor=descriptor,
                            table_keywords=table_keywords,
                            column_keywords=column_keywords,
                            table_proxy=table_proxy)

    # Unpack table proxy if it was requested
    if table_proxy is True:
        assert isinstance(out_ds, tuple)
        out_ds, tp = out_ds
        assert isinstance(tp, TableProxy)
    else:
        tp = None

    # Repack the Table Proxy
    if table_proxy is True:
        return out_ds, tp

    return out_ds


def xds_from_table(table_name, columns=None,
                   index_cols=None, group_cols=None,
                   **kwargs):
    """
    Create multiple :class:`xarray.Dataset` objects
    from CASA table ``table_name`` with the rows lexicographically
    sorted according to the columns in ``index_cols``.
    If ``group_cols`` is supplied, the table data is grouped into
    multiple :class:`xarray.Dataset` objects, each associated with a
    permutation of the unique values for the columns in ``group_cols``.

    Notes
    -----
    Both ``group_cols`` and ``index_cols`` should consist of
    columns that are part of the table index.

    However, this may not always be possible as CASA tables
    may not always contain indexing columns.
    The ``ANTENNA`` or ``SPECTRAL_WINDOW`` Measurement Set subtables
    are examples in which the ``row id`` serves as the index.

    Generally, calling

    .. code-block:: python

        antds = list(xds_from_table("WSRT.MS::ANTENNA"))

    is fine, since the data associated with each row of the ``ANTENNA``
    table has the same shape and so a dask or numpy array can be
    constructed around the contents of the table.

    This may not be the case for the ``SPECTRAL_WINDOW`` subtable.
    Here, each row defines a separate spectral window, but each
    spectral window may contain different numbers of frequencies.
    In this case, it is probably better to group the subtable
    by ``row``.

    There is a *special* group column :code:`"__row__"`
    that can be used to group the table by row.

    .. code-block:: python

        for spwds in xds_from_table("WSRT.MS::SPECTRAL_WINDOW",
                                            group_cols="__row__"):
            ...

    If :code:`"__row__"` is used for grouping, then no other
    column may be used. It should also only be used for *small*
    tables, as the number of datasets produced, may be prohibitively
    large.

    Parameters
    ----------
    table_name : str
        CASA table
    columns : list or tuple, optional
        Columns present on the returned dataset.
        Defaults to all if ``None``
    index_cols  : list or tuple, optional
        List of CASA table indexing columns. Defaults to :code:`()`.
    group_cols : list or tuple, optional
        List of columns on which to group the CASA table.
        Defaults to :code:`()`
    table_schema : dict or str or list of dict or str, optional
        A schema dictionary defining the dimension naming scheme for
        each column in the table. For example:

        .. code-block:: python

            {
                "UVW": {'dims': ('uvw',)},
                "DATA": {'dims': ('chan', 'corr')},
            }

        will result in the UVW and DATA arrays having dimensions
        :code:`('row', 'uvw')` and :code:`('row', 'chan', 'corr')`
        respectively.

        A string can be supplied, which will be matched
        against existing default schemas. Examples here include
        ``MS``, ``ANTENNA`` and ``SPECTRAL_WINDOW``
        corresponding to ``Measurement Sets`` the ``ANTENNA`` subtable
        and the ``SPECTRAL_WINDOW`` subtable, respectively.

        By default, the end of ``table_name`` will be
        inspected to see if it matches any default schemas.

        It is also possible to supply a list of strings or dicts defining
        a sequence of schemas which are combined. Later elements in the
        list override previous elements. In the following
        example, the standard UVW MS component name scheme is overridden
        with "my-uvw".

        .. code-block:: python

            ["MS", {"UVW": {'dims': ('my-uvw',)}}]

    table_keywords : {False, True}, optional
        If True, returns table keywords.
        Changes return type of the function into a tuple

    column_keywords : {False, True}, optional
        If True return keywords for each column on the table
        Changes return type of the function into a tuple

    table_proxy : {False, True}, optional
        If True returns the Table Proxy associated with the Dataset

    taql_where : str, optional
        TAQL where clause. For example, to exclude auto-correlations

        .. code-block:: python

            xds_from_table("WSRT.MS", taql_where="ANTENNA1 != ANTENNA2")

    chunks : list of dicts or dict, optional
        A :code:`{dim: chunk}` dictionary, specifying the chunking
        strategy of each dimension in the schema.
        Defaults to :code:`{'row': 100000 }` which will partition
        the row dimension into chunks of 100000.

        * If a dict, the chunking strategy is applied to each group.
        * If a list of dicts, each element is applied
          to the associated group. The last element is
          extended over the remaining groups if there
          are insufficient elements.

        It's also possible to specify the individual chunks for
        multiple dimensions:

        .. code-block:: python

            {'row': (40000, 60000, 40000, 60000),
             'chan': (16, 16, 16, 16),
             'corr': (1, 2, 1)}

        The above chunks a 200,000 row, 64 channel and 4 correlation
        space into 4 x 4 x 3 = 48 chunks, but requires prior
        knowledge of dimensionality, probably obtained with an
        initial call to :func:`xds_from_table`.

    Returns
    -------
    datasets : list of :class:`xarray.Dataset`
        datasets for each group, each ordered by indexing columns
    table_keywords : dict, optional
        Returned if ``table_keywords is True``
    column_keywords : dict, optional
        Returned if ``column_keywords is True``
    table_proxy : :class:`daskms.TableProxy`, optional
        Returned if ``table_proxy is True``


    """
    if isinstance(table_name, DaskMSStore):
        table_name = table_name.casa_path()
    else:
        table_name = DaskMSStore(table_name).casa_path()

    columns = promote_columns(columns, [])
    index_cols = promote_columns(index_cols, [])
    group_cols = promote_columns(group_cols, [])

    return DatasetFactory(table_name, columns,
                          group_cols, index_cols,
                          **kwargs).datasets()


def xds_from_ms(ms, columns=None, index_cols=None, group_cols=None, **kwargs):
    """
    Generator yielding a series of xarray datasets representing
    the contents a Measurement Set.
    It defers to :func:`xds_from_table`, which should be consulted
    for more information.

    Parameters
    ----------
    ms : str
        Measurement Set filename
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

    columns = promote_columns(columns, [])
    index_cols = promote_columns(index_cols, _DEFAULT_INDEX_COLUMNS)
    group_cols = promote_columns(group_cols, _DEFAULT_GROUP_COLUMNS)

    kwargs.setdefault("table_schema", "MS")

    return xds_from_table(ms, columns=columns,
                          index_cols=index_cols,
                          group_cols=group_cols,
                          **kwargs)


def xds_from_storage_table(store, **kwargs):
    if not isinstance(store, DaskMSStore):
        store = DaskMSStore(store)

    typ = store.type()

    if typ == "casa":
        return xds_from_table(store, **kwargs)
    elif typ == "zarr":
        from daskms.experimental.zarr import xds_from_zarr
        return xds_from_zarr(store, **kwargs)
    elif typ == "parquet":
        from daskms.experimental.arrow import xds_from_parquet
        return xds_from_parquet(store, **kwargs)
    else:
        raise TypeError(f"Unknown dataset {typ}")


def xds_from_storage_ms(store, **kwargs):
    if not isinstance(store, DaskMSStore):
        store = DaskMSStore(store)

    typ = store.type()

    if typ == "casa":
        return xds_from_ms(store, **kwargs)
    elif typ == "zarr":
        from daskms.experimental.zarr import xds_from_zarr
        return xds_from_zarr(store, **kwargs)
    elif typ == "parquet":
        from daskms.experimental.arrow import xds_from_parquet
        return xds_from_parquet(store, **kwargs)
    else:
        raise TypeError(f"Unknown dataset {typ}")


# Set docstring variables in try/except
# ``__doc__`` may not be present as
# ``python -OO`` strips docstrings
try:
    xds_from_ms.__doc__ %= {
        'indices': _DEFAULT_INDEX_COLUMNS,
        'groups': _DEFAULT_GROUP_COLUMNS}
except AttributeError:
    pass
