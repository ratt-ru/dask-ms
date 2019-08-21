# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging

try:
    import xarray as xr
except ImportError:
    xr = None

from daskms.dataset import Dataset
from daskms.reads import DatasetFactory
from daskms.writes import write_datasets
from daskms.utils import promote_columns

_DEFAULT_GROUP_COLUMNS = ["FIELD_ID", "DATA_DESC_ID"]
_DEFAULT_INDEX_COLUMNS = ["TIME"]

log = logging.getLogger(__name__)


def xds_to_table(xds, table_name, columns=None, **kwargs):
    """
    Generates a dask array which writes the
    specified columns from :class:`xarray.Dataset`'s into
    the CASA table specified by ``table_name`` when
    the :meth:`dask.array.Array.compute` method is called.

    Parameters
    ----------
    xds : :class:`xarray.Dataset` or list of :class:`xarray.Dataset`
        dataset(s) containing the specified columns. If a list of datasets
        is provided, the concatenation of the columns in
        sequential datasets will be written.
    table_name : str
        CASA table path
    columns : tuple or list, optional
        list of column names to write to the table.
        If ``None`` all columns will be written.

    Returns
    -------
    writes : :class:`dask.array.Array`
        dask array representing the write to the
        dataset.
    """

    # Promote dataset to a list
    if not isinstance(xds, (tuple, list)):
        xds = [xds]

    columns = promote_columns(columns, [])

    datasets = []

    # No xarray available, assume dask datasets
    if xr is None:
        datasets = xds
    else:
        for ds in xds:
            if isinstance(ds, Dataset):
                # Already a dask dataset
                datasets.append(ds)
            elif isinstance(ds, xr.Dataset):
                # Produce a list of internal variable and dataset types
                # from the xarray Dataset
                variables = {k: (v.dims, v.data, v.attrs) for k, v
                             in ds.data_vars.items()}

                coords = {k: (v.dims, v.data, v.attrs) for k, v
                          in ds.coords.items()}

                dds = Dataset(variables, attrs=ds.attrs, coords=coords)
                datasets.append(dds)
            else:
                raise TypeError("Invalid Dataset type '%s'" % type(ds))

    # Write the datasets
    return write_datasets(table_name, datasets, columns)


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
                "UVW": {'dask': {'dims': ('uvw',)}},
                "DATA": {'dask': {'dims': ('chan', 'corr')}},
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

            ["MS", {"UVW": {'dask': {'dims': ('my-uvw',)}}}]

    taql_where : str, optional
        TAQL where clause. For example, to exclude auto-correlations

        .. code-block:: python

            xds_from_table("WSRT.MS", taql_where="ANTENNA1 != ANTENNA2")

    chunks : list of dicts or dict, optional
        A :code:`{dim: chunk}` dictionary, specifying the chunking
        strategy of each dimension in the schema.
        Defaults to :code:`{'row': 100000 }`.

        * If a dict, the chunking strategy is applied to each group.
        * If a list of dicts, each element is applied
          to the associated group. The last element is
          extended over the remaining groups if there
          are insufficient elements.

    Returns
    -------
    datasets : list of :class:`xarray.Dataset`
        datasets for each group, each ordered by indexing columns
    """
    columns = promote_columns(columns, [])
    index_cols = promote_columns(index_cols, [])
    group_cols = promote_columns(group_cols, [])

    dask_datasets = DatasetFactory(table_name, columns,
                                   group_cols, index_cols,
                                   **kwargs).datasets()

    # Return dask datasets if xarray is not available
    if xr is None:
        return dask_datasets

    xarray_datasets = []

    for ds in dask_datasets:
        data_vars = collections.OrderedDict()
        coords = collections.OrderedDict()

        for k, v in ds.data_vars.items():
            data_vars[k] = xr.DataArray(v.data, dims=v.dims, attrs=v.attrs)

        for k, v in ds.coords.items():
            coords[k] = xr.DataArray(v.data, dims=v.dims, attrs=v.attrs)

        xarray_datasets.append(xr.Dataset(data_vars,
                                          attrs=dict(ds.attrs),
                                          coords=coords))

    return xarray_datasets


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
        Defaults to :code:`%(index)s`
    group_cols  : tuple or list, optional
        Sequence of grouping columns.
        Defaults to :code:`%(parts)s`
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


# Set docstring variables in try/except
# ``__doc__`` may not be present as
# ``python -OO`` strips docstrings
try:
    xds_from_ms.__doc__ %= {
        'index': _DEFAULT_INDEX_COLUMNS,
        'parts': _DEFAULT_GROUP_COLUMNS}
except AttributeError:
    pass
