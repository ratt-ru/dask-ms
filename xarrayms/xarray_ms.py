from __future__ import print_function

try:
    from future_builtins import zip
except ImportError:
    pass

import collections
from functools import partial
import logging
import os
import os.path

import dask
import dask.array as da
import numpy as np
import pyrap.tables as pt
from six import string_types
from six.moves import range

try:
    from cytoolz import merge
except ImportError:
    from toolz import merge

import xarray as xr

from xarrayms.table_proxy import TableProxy
from xarrayms.known_table_schemas import registered_schemas

_DEFAULT_PARTITION_COLUMNS = ["FIELD_ID", "DATA_DESC_ID"]
_DEFAULT_INDEX_COLUMNS = ["TIME"]
_DEFAULT_ROWCHUNKS = 100000

log = logging.getLogger(__name__)


def select_clause(select_cols):
    if select_cols is None or len(select_cols) == 0:
        return "SELECT * "

    return " ".join(("SELECT", ", ".join(select_cols)))


def orderby_clause(index_cols):
    if len(index_cols) == 0:
        return ""

    return " ".join(("ORDERBY", ", ".join(index_cols)))


def groupby_clause(group_cols):
    if len(group_cols) == 0:
        return ""

    return " ".join(("GROUPBY", ", ".join(group_cols)))


def where_clause(group_cols, group_vals):
    if len(group_cols) == 0:
        return ""

    assign_str = ["%s=%s" % (c, v) for c, v in zip(group_cols, group_vals)]
    return " ".join(("WHERE", " AND ".join(assign_str)))


def short_table_name(table_name):
    """
    Returns the last part

    Parameters
    ----------
    table_name : str
        CASA table path

    Returns
    -------
    str
        Shortened path

    """
    return os.path.split(table_name.rstrip(os.sep))[1]


def table_open_graph(table_name, **kwargs):
    """
    Generate a dask graph containing table open commands

    Parameters
    ----------
    table_name : str
        CASA table name
    **kwargs:
        Keywords arguments passed to the :meth:`casacore.tables.table`
        constructor, for e.g. :code:`readonly=False`

    Returns
    -------
    tuple
        Graph key associated with the opened table
    dict
        Dask graph containing the graph open command

    """
    token = dask.base.tokenize(table_name, kwargs)
    table_key = ('open', short_table_name(table_name), token)
    table_graph = {table_key: (partial(TableProxy, **kwargs), table_name)}
    return table_key, table_graph


def _np_put_fn(tp, c, d, s, n):
    tp("putcol", c, d, startrow=s, nrow=n)
    return np.asarray([True])


def xds_to_table(xds, table_name, columns=None):
    """
    Generates a dask array which writes the
    specified columns from an :class:`xarray.Dataset` into
    the CASA table specified by ``table_name`` when
    the :meth:`dask.array.Array.compute` method is called.

    Parameters
    ----------
    xds : :class:`xarray.Dataset`
        dataset containing the specified columns.
    table_name : str
        CASA table path
    columns : tuple or list, optional
        list of column names to write to the table.
        If ``None`` all columns will be written.

    Returns
    -------
    :class:`dask.array.Array`
        dask array representing the write to the
        datset.
    """

    table_key, dsk = table_open_graph(table_name, readonly=False)
    rows = xds.table_row.values

    if columns is None:
        columns = xds.data_vars.keys()
    elif isinstance(columns, string_types):
        columns = [columns]

    token = dask.base.tokenize(table_name, columns, rows)
    name = '-'.join((short_table_name(table_name), "putcol", token))

    chunk = 0
    chunks = []
    row_idx = 0

    for c in columns:
        data_array = getattr(xds, c)
        dask_array = data_array.data

        dsk.update(dask_array.__dask_graph__())

        array_chunks = data_array.chunks

        if not data_array.dims[row_idx] == 'row':
            raise ValueError("xds.%s.dims[0] != 'row'" % c)

        chunk_start = 0

        for chunk_size in array_chunks[row_idx]:
            chunk_end = chunk_start + chunk_size
            row_put_fns = []

            # Split into runs of consecutive rows within this chunk
            d = np.ediff1d(rows[chunk_start:chunk_end], to_begin=2, to_end=2)
            runs = np.nonzero(d != 1)[0] + chunk_start

            for run_start, run_end in zip(runs[:-1], runs[1:]):
                # Slice the dask array and then obtain the graph representing
                # the slice operation. There should be only one key-value pair
                # as we're slicing within a chunk
                data = dask_array[run_start:run_end]
                dsk_slice = data.__dask_graph__().dicts[data.name]
                assert len(dsk_slice) == 1
                dsk.update(dsk_slice)
                row_put_fns.append((_np_put_fn, table_key,
                                    c, dsk_slice.keys()[0],
                                    rows[run_start], run_end - run_start))

            dsk[(name, chunk)] = (np.logical_and.reduce, row_put_fns)
            chunks.append(1)
            chunk += 1
            chunk_start = chunk_end

    chunks = (tuple(chunks),)
    return da.Array(dsk, name, chunks, dtype=np.bool)


def _np_get_fn(tp, c, s, n):
    return tp("getcol", c, startrow=s, nrow=n)


def _list_get_fn(tp, c, s, n):
    return np.asarray(_np_get_fn(tp, c, s, n))


def generate_table_getcols(table_name, table_key, dsk_base,
                           column, shape, dtype, rows, rowchunks):
    """
    Generates a :class:`dask.array.Array` representing ``column``
    in ``table_name`` and backed by a series of
    :meth:`casacore.tables.table.getcol` commands.


    Parameters
    ----------
    table_name : str
        CASA table filename path
    table_key : tuple
        dask graph key referencing an opened table object.
    dsk_base : dict
        dask graph containing table object opening functionality.
    column : str
        Name of the column to generate
    shape : tuple
        Shape of the array
    dtype : np.dtype or object
        Data type of the array
    rows : np.ndarray
        CASA table row id's defining an ordering
        of the table data.
    rowchunks : integer
        The chunk size for the row dimension of the resulting
        :class:`dask.array.Array`s.

    Returns
    -------
    :class:`dask.array.Arrays`
        Dask array representing the column
    """

    token = dask.base.tokenize(table_name, column, rows)
    short_name = short_table_name(table_name)
    name = '-'.join((short_name, "getcol", column.lower(), token))
    dsk = {}

    chunk_extra = shape[1:]
    key_extra = (0,) * len(shape[1:])

    # infer the type of getter we should be using
    get_fn = _np_get_fn if isinstance(dtype, np.dtype) else _list_get_fn

    # Iterate through the rows in groups of rowchunks
    # For each iteration we generate one chunk
    # in the resultant dask array
    start_row = 0

    for chunk, chunk_size in enumerate(rowchunks[0]):
        end_row = start_row + chunk_size

        # Split into runs of consecutive rows within this chunk
        d = np.ediff1d(rows[start_row:end_row], to_begin=2, to_end=2)
        runs = np.nonzero(d != 1)[0] + start_row

        # How many rows in this chunk?
        chunk_size = 0
        # Store a list of lambdas executing getcols on consecutive runs
        row_get_fns = []

        for run_start, run_end in zip(runs[:-1], runs[1:]):
            run_len = run_end - run_start
            row_get_fns.append((get_fn, table_key, column,
                                rows[run_start], run_len))
            chunk_size += run_len

        # Create the key-value dask entry for this chunk
        dsk[(name, chunk) + key_extra] = (np.concatenate, row_get_fns)

        start_row = end_row

    chunks = rowchunks + tuple((c,) for c in chunk_extra)
    return da.Array(merge(dsk_base, dsk), name, chunks, dtype=dtype)


def lookup_table_schema(table_name, lookup_str):
    """
    Attempts to heuristically generate a table schema dictionary,
    given the ``lookup_str`` argument. If this fails,
    an empty dictionary is returned.

    Parameters
    ----------
    table_name : str
        CASA table path
    lookup_str : str or ``None``
        If a string, the resulting schema will be
        internally looked up in the known table schemas.
        If ``None``, the end of ``table_name`` will be
        inspected to perform the lookup.

    Returns
    -------
    dict
        A dictionary of the form :code:`{column:dims}`.
        e.g. :code:`{'UVW' : ('row', 'uvw'), 'DATA': ('row', 'chan', 'corr')}
    """
    schemas = registered_schemas()

    if lookup_str is None:
        def _search_schemas(table_name, schemas):
            """ Guess the schema from the table name """
            for k in schemas.keys():
                if table_name.endswith('::' + k):
                    return schemas[k]
            return {}

        return _search_schemas(table_name, schemas)
    # Get a registered table schema
    elif isinstance(lookup_str, string_types):
        return schemas.get(lookup_str, {})

    raise TypeError("Invalid table_schema type '%s'" % type(table_schema))


def xds_from_table_impl(table_name, table, table_schema,
                        table_graph, table_key,
                        columns, rows, chunks):
    """
    Parameters
    ----------
    table_name : str
        CASA table filename path
    table : :class:`casacore.tables.table`
        CASA table object, used to inspect metadata
        for creating Datasets
    table_schema : str or dict
        Table schema.
    table_graph : dict
        Dask graph containing ``table_key``
    table_key : tuple
        Tuple referencing the table open command
    columns : tuple or list
        Columns present on the returned dataset.
    rows : np.ndarray
        CASA table row id's defining an ordering
        of the table data.
    chunks : dict
        The chunk size for the dimensions of the resulting
        :class:`dask.array.Array`s.

    Returns
    -------
    :class:`xarray.Dataset`
        xarray dataset
    """

    nrows = rows.size
    col_metadata = {}

    missing = []

    if not isinstance(table_schema, collections.Mapping):
        table_schema = lookup_table_schema(table_name, table_schema)

    # Work out metadata for each column
    for c in columns:
        try:
            # Read the starting row
            row = table.getcol(c, startrow=rows[0], nrow=1)
        except Exception:
            log.warn("Ignoring '%s' column", c)
            missing.append(c)
        else:
            # Usually we get numpy arrays
            if isinstance(row, np.ndarray):
                shape = (nrows,) + row.shape[1:]
                dtype = row.dtype
            # In these cases, we're getting a list of (probably) strings
            elif isinstance(row, (list, tuple)):
                shape = (nrows,)
                dtype = object
            else:
                raise TypeError("Unhandled row type '%s'" % type(row))

            # Generate an xarray dimension schema
            # from supplied or inferred schemas if possible
            try:
                extra = table_schema[c].dims
            except KeyError:
                extra = tuple('%s-%d' % (c, i) for i in range(1, len(shape)))

            col_metadata[c] = (shape, ("row",) + extra, dtype)

    # Remove missing columns
    columns = columns.difference(missing)

    # Determine a row chunking scheme
    rowchunks = da.core.normalize_chunks(chunks['row'], (rows.size,))

    # Insert arrays into dataset in sorted order
    data_arrays = collections.OrderedDict()

    for c in sorted(columns):
        shape, dims, dtype = col_metadata[c]
        col_dask_array = generate_table_getcols(table_name, table_key,
                                                table_graph, c, shape, dtype,
                                                rows, rowchunks)

        data_arrays[c] = xr.DataArray(col_dask_array, dims=dims)

    # Create the dataset, assigning a table_row coordinate
    # associated with the row dimension
    return xr.Dataset(data_arrays, coords={'table_row': ('row', rows)})


def xds_from_table(table_name, columns=None,
                   index_cols=None, part_cols=None,
                   table_schema=None, chunks=None):
    """
    Generator producing multiple :class:`xarray.Dataset` objects
    from CASA table ``table_name`` with the rows lexicographically
    sorted according to the columns in ``index_cols``.
    If ``part_cols`` is supplied, the table data is partitioned into
    multiple :class:`xarray.Dataset` objects, each associated with a
    permutation of the unique values for the columns in ``part_cols``.

    Notes
    -----
    Both ``part_cols`` and ``index_cols`` should consist of
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
    In this case, it is probably better to partition the subtable
    by ``row``.

    There is a *special* partition column :code:`"__row__"`
    that can be used to partition the table by row.

    .. code-block:: python

        for spwds in xds_from_table("WSRT.MS::SPECTRAL_WINDOW",
                                            part_cols="__row__"):
            ...

    If :code:`"__row__"` is used for partioning, then no other
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
    part_cols : list or tuple, optional
        List of columns on which to partition the CASA table.
        Defaults to :code:`()`
    table_schema : str or dict, optional
        A schema dictionary defining the dimension naming scheme for
        each column in the table. For example:

        .. code-block:: python

            {"UVW" : ('uvw',), DATA" : ('chan', 'corr')}

        will result in the UVW and DATA arrays having dimensions
        :code:`('row', 'uvw')` and :code:`('row', 'chan', 'corr')`
        respectively.

        Alternatively a string can be supplied, which will be matched
        against existing default schemas. Examples here include
        ``MS``, ``ANTENNA`` and ``SPECTRAL_WINDOW``
        corresponding to ``Measurement Sets`` the ``ANTENNA`` subtable
        and the ``SPECTRAL_WINDOW`` subtable, respectively.

        If ``None`` is supplied, the end of ``table_name`` will be
        inspected to see if it matches any default schemas.

    chunks : list of dicts or dict, optional
        A :code:`{dim: chunk}` dictionary, specifying the chunking
        strategy of each dimension in the schema.
        Defaults to :code:`{'row': 100000 }`.

        * If a dict, the chunking strategy is applied to each partition.
        * If a list of dicts, each element is applied
          to the associated partition. The last element is
          extended over the remaining partitions if there
          are insufficient elements.

    Yields
    ------
    :class:`xarray.Dataset`
        datasets for each partition, each ordered by indexing columns
    """
    if chunks is None:
        chunks = [{'row': _DEFAULT_ROWCHUNKS}]
    elif isinstance(chunks, tuple):
        chunks = list(chunks)
    elif isinstance(chunks, dict):
        chunks = [chunks]

    if index_cols is None:
        index_cols = ()
    elif isinstance(index_cols, tuple):
        index_cols = list(index_cols)
    elif not isinstance(part_cols, list):
        index_cols = [index_cols]

    if part_cols is None:
        part_cols = ()
    elif isinstance(part_cols, tuple):
        part_cols = list(part_cols)
    elif not isinstance(part_cols, list):
        part_cols = [part_cols]

    table_key, dsk = table_open_graph(table_name)

    with pt.table(table_name) as T:
        columns = set(T.colnames() if columns is None else columns)

        # Handle the case where we partition on each table row
        if len(part_cols) == 1 and part_cols[0] == "__row__":
            # Get the rows giving the ordering
            order = orderby_clause(index_cols)
            query = "SELECT ROWID() AS __tablerow__ FROM $T %s" % order

            with pt.taql(query) as gq:
                rows = gq.getcol("__tablerow__")

            # Generate a dataset for each row
            for r in range(rows.size):
                ds = xds_from_table_impl(table_name, T, table_schema,
                                         dsk, table_key, columns,
                                         rows[r:r + 1], chunks[0])

                yield (ds.squeeze(drop=True)
                         .assign_attrs(table_row=rows[r]))

        # Otherwise partition by given columns
        elif len(part_cols) > 0:
            # Aggregate indexing column values so that we can
            # individually sort each group's rows
            index_group_cols = ["GAGGR(%s) AS GROUP_%s" % (c, c)
                                for c in index_cols]
            # Get the rows for each group
            index_group_cols.append("GROWID() as __tablerow__")

            select = select_clause(part_cols + index_group_cols)
            groupby = groupby_clause(part_cols)
            orderby = orderby_clause(index_cols)

            query = "%s FROM $T %s %s" % (select, groupby, orderby)

            with pt.taql(query) as gq:
                # For each group
                for i in range(0, gq.nrows()):
                    # Obtain this group's row ids and indexing columns
                    # Need reversed since last column is lexsort's
                    # primary sort key
                    key, rows = gq.getvarcol("__tablerow__", i, 1).popitem()
                    group_indices = tuple(gq.getvarcol("GROUP_%s" % c, i, 1)
                                          .pop(key)[0]
                                          for c in reversed(index_cols))

                    # Resort row id by indexing columns,
                    # eliminating the extra dimension introduced by getvarcol
                    group_rows = rows[0][np.lexsort(group_indices)]

                    # Get the singleton group partition values
                    group_values = tuple(gq.getvarcol(c, i, 1).pop(key)[0]
                                         for c in part_cols)

                    # Use the last chunk if there aren't enough
                    try:
                        group_chunks = chunks[i]
                    except IndexError:
                        group_chunks = chunks[-1]

                    ds = xds_from_table_impl(table_name, T, table_schema,
                                             dsk, table_key,
                                             columns.difference(part_cols),
                                             group_rows, group_chunks)

                    yield ds.assign_attrs(zip(part_cols, group_values))

        # No partioning case
        else:
            query = ("SELECT ROWID() as __tablerow__ "
                     "FROM $T %s" % orderby_clause(index_cols))

            with pt.taql(query) as gq:
                yield xds_from_table_impl(table_name, T, table_schema,
                                          dsk, table_key,
                                          columns.difference(part_cols),
                                          gq.getcol("__tablerow__"),
                                          chunks[0])


def xds_from_ms(ms, columns=None, index_cols=None, part_cols=None,
                chunks=None):
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
    part_cols  : tuple or list, optional
        Sequence of partioning columns.
        Defaults to :code:`%(parts)s`
    chunks : list of dicts or dict, optional
        Dictionaries of dimension chunks.

    Yields
    ------
    :class:`xarray.Dataset`
        xarray datasets for each partition
    """

    if index_cols is None:
        index_cols = _DEFAULT_INDEX_COLUMNS
    elif isinstance(index_cols, list):
        index_cols = tuple(index_cols)
    elif not isinstance(index_cols, tuple):
        index_cols = (index_cols,)

    if part_cols is None:
        part_cols = _DEFAULT_PARTITION_COLUMNS
    elif isinstance(part_cols, list):
        part_cols = tuple(part_cols)
    elif not isinstance(part_cols, tuple):
        part_cols = (part_cols,)

    for ds in xds_from_table(ms, columns=columns,
                             index_cols=index_cols, part_cols=part_cols,
                             table_schema="MS", chunks=chunks):
        yield ds


# Set docstring variables in try/except
# ``__doc__`` may not be present as
# ``python -OO`` strips docstrings
try:
    xds_from_ms.__doc__ %= {
        'index': _DEFAULT_INDEX_COLUMNS,
        'parts': _DEFAULT_PARTITION_COLUMNS}
except AttributeError:
    pass
