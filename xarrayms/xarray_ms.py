from __future__ import absolute_import
from __future__ import division
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

_DEFAULT_GROUP_COLUMNS = ["FIELD_ID", "DATA_DESC_ID"]
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


def _get_row_runs(rows, chunks):
    row_runs = []
    start_row = 0
    nruns = 0

    for chunk, chunk_size in enumerate(chunks[0]):
        end_row = start_row + chunk_size
        chunk_rows = rows[start_row:end_row]

        # Split into runs of consecutive rows within this chunk
        diff = np.ediff1d(chunk_rows, to_begin=-10, to_end=-10)
        idx = np.nonzero(diff != 1)[0]
        # Starting row and length of the run
        start_and_len = np.empty((idx.size - 1, 2), dtype=np.int64)
        start_and_len[:, 0] = chunk_rows[idx[:-1]]
        start_and_len[:, 1] = np.diff(idx)
        row_runs.append(start_and_len)

        start_row = end_row
        nruns += idx.size - 1

    if end_row != rows.size:
        raise ValueError("Chunk sum didn't match the number of rows")

    if 100.0 * len(chunks[0]) / nruns < 33.0:
        log.warn("Grouping and ordering strategy has produced "
                 "a fragmented MS row ordering. "
                 "Disk access may be slow.")

    return row_runs


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


def _chunk_putcols_np(table_proxy, column, runs, data):
    rr = 0

    for rs, rl in runs:
        table_proxy("putcol", column, data[rr:rr + rl],
                    startrow=rs, nrow=rl)
        rr += rl

    return np.full(runs.shape[0], True)


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

    out_chunk = 0

    # Get the DataArrays for each column
    col_arrays = [getattr(xds, c) for c in columns]
    # Tokenize putcol on the dask arrays
    token = dask.base.tokenize(table_name, col_arrays)
    name = '-'.join((short_table_name(table_name), "putcol",
                     str(columns), token))

    # Generate the graph for each column
    for c, data_array in zip(columns, col_arrays):
        dask_array = data_array.data
        dims = data_array.dims
        chunks = dask_array.chunks
        shape = dask_array.shape

        if dims[0] != 'row':
            raise ValueError("xds.%s.dims[0] != 'row'" % c)

        multiple = [(dim, chunk) for dim, chunk
                    in zip(dims[1:], chunks[1:])
                    if len(chunk) != 1]

        if len(multiple) > 0:
            raise ValueError("Column '%s' has multiple chunks in the "
                             "following dimensions '%s'. Only chunking "
                             "in 'row' is currently supported. "
                             "Use 'rechunk' so that the mentioned "
                             "dimensions contain a single chunk."
                             % (c, multiple))

        # Need extra chunk indices in the array keys
        # that we're retrieving
        key_extra = (0,) * len(chunks[1:])

        # Get row runs for the row chunks
        row_runs = _get_row_runs(rows, chunks)

        for chunk, row_run in enumerate(row_runs):
            # graph key for the array chunk that we'll write
            array_chunk = (dask_array.name, chunk) + key_extra
            # Add the write operation to the graph
            dsk[(name, out_chunk)] = (_chunk_putcols_np, table_key,
                                      c, row_run, array_chunk)
            out_chunk += 1

        # Add the arrays graph to final graph
        dsk.update(dask_array.__dask_graph__())

    return da.Array(dsk, name, ((1,) * out_chunk,), dtype=np.bool)


def _chunk_getcols_np(table_proxy, column, shape, dtype, runs):
    # results shape = (sum(row_lengths),) + shape
    result = np.empty((np.sum(runs[:, 1]),) + shape, dtype=dtype)
    rr = 0

    # Get data directly into the result array
    for rs, rl in runs:
        table_proxy("getcolnp", column, result[rr:rr + rl], rs, rl)
        rr += rl

    return result


def _chunk_getcols_object(table_proxy, column, shape, dtype, runs):
    # results shape = (sum(row_lengths),) + shape
    result = np.empty((np.sum(runs[:, 1]),) + shape, dtype=dtype)
    rr = 0

    # Wrap objects (probably strings) in numpy arrays
    for rs, rl in runs:
        data = table_proxy("getcol", column, rs, rl)
        result[rr:rr + rl] = np.asarray(data, dtype=dtype)
        rr += rl

    return result


def generate_table_getcols(table_name, table_key, dsk_base,
                           column, shape, dtype,
                           row_chunks, row_runs):
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
    row_chunks : tuple
        Chunk strategy for the row dimension. Should look like
        :code:`((r1,r2,...,rn),)`
    row_runs : list of :class:`numpy.ndarray`
        List of row runs for each chunk

    Returns
    -------
    :class:`dask.array.Arrays`
        Dask array representing the column
    """

    token = dask.base.tokenize(table_name, column, row_runs)
    short_name = short_table_name(table_name)
    name = '-'.join((short_name, "getcol", column.lower(), token))

    chunk_extra = shape[1:]
    key_extra = (0,) * len(shape[1:])

    # infer the type of getter we should be using
    if isinstance(dtype, np.dtype):
        _get_fn = _chunk_getcols_np
    else:
        _get_fn = _chunk_getcols_object

    # Iterate through the rows in groups of row_runs
    # For each iteration we generate one chunk
    # in the resultant dask array
    dsk = {(name, chunk) + key_extra: (_get_fn, table_key, column,
                                       chunk_extra, dtype, runs)
           for chunk, runs in enumerate(row_runs)}

    chunks = row_chunks + tuple((c,) for c in chunk_extra)
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


def column_metadata(table, columns, table_schema, rows):
    """
    Returns :code:`{column: (shape, dim_schema, dtype)}` metadata
    for each column in ``columns``.

    Parameters
    ----------
    table : :class:`pyrap.tables.table`
        CASA table object
    columns : list of str
        List of CASA table columns
    table_schema : dict
        Table schema for ``table``

    Returns
    -------
    dict
        :code:`{column: (shape, dim_schema, dtype)}`
    """

    nrows = rows.size
    column_metadata = collections.OrderedDict()

    # Work out metadata for each column
    for c in sorted(columns):
        try:
            # Read the starting row
            row = table.getcol(c, startrow=rows[0], nrow=1)
        except Exception:
            log.warn("Ignoring '%s' column")
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

            column_metadata[c] = (shape, ("row",) + extra, dtype)

    return column_metadata


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

    if not isinstance(table_schema, collections.Mapping):
        table_schema = lookup_table_schema(table_name, table_schema)

    # Get column metadata
    col_metadata = column_metadata(table, columns, table_schema, rows)

    # Determine a row chunking scheme
    row_chunks = da.core.normalize_chunks(chunks['row'], (rows.size,))

    # Get row runs for each chunk
    row_runs = _get_row_runs(rows, row_chunks)

    # Insert arrays into dataset in sorted order
    data_arrays = collections.OrderedDict()

    for column, (shape, dims, dtype) in col_metadata.items():
        col_dask_array = generate_table_getcols(table_name, table_key,
                                                table_graph, column,
                                                shape, dtype,
                                                row_chunks, row_runs)

        data_arrays[column] = xr.DataArray(col_dask_array, dims=dims)

    # Create the dataset, assigning a table_row coordinate
    # associated with the row dimension
    return xr.Dataset(data_arrays, coords={'table_row': ('row', rows)})


def xds_from_table(table_name, columns=None,
                   index_cols=None, group_cols=None,
                   table_schema=None, chunks=None):
    """
    Generator producing multiple :class:`xarray.Dataset` objects
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

        * If a dict, the chunking strategy is applied to each group.
        * If a list of dicts, each element is applied
          to the associated group. The last element is
          extended over the remaining groups if there
          are insufficient elements.

    Yields
    ------
    :class:`xarray.Dataset`
        datasets for each group, each ordered by indexing columns
    """
    if chunks is None:
        chunks = [{'row': _DEFAULT_ROWCHUNKS}]
    elif isinstance(chunks, tuple):
        chunks = list(chunks)
    elif isinstance(chunks, dict):
        chunks = [chunks]

    if index_cols is None:
        index_cols = []
    elif isinstance(index_cols, tuple):
        index_cols = list(index_cols)
    elif not isinstance(group_cols, list):
        index_cols = [index_cols]

    if group_cols is None:
        group_cols = []
    elif isinstance(group_cols, tuple):
        group_cols = list(group_cols)
    elif not isinstance(group_cols, list):
        group_cols = [group_cols]

    table_key, dsk = table_open_graph(table_name)

    with pt.table(table_name) as T:
        columns = set(T.colnames() if columns is None else columns)

        # Handle the case where we group on each table row
        if len(group_cols) == 1 and group_cols[0] == "__row__":
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

        # Otherwise group by given columns
        elif len(group_cols) > 0:
            # Aggregate indexing column values so that we can
            # individually sort each group's rows
            index_group_cols = ["GAGGR(%s) AS GROUP_%s" % (c, c)
                                for c in index_cols]
            # Get the rows for each group
            index_group_cols.append("GROWID() as __tablerow__")

            select = select_clause(group_cols + index_group_cols)
            groupby = groupby_clause(group_cols)
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
                    if len(group_indices) > 0:
                        group_rows = rows[0][np.lexsort(group_indices)]
                    else:
                        group_rows = rows[0]

                    # Get the singleton group values
                    group_values = tuple(gq.getvarcol(c, i, 1).pop(key)[0]
                                         for c in group_cols)

                    # Use the last chunk if there aren't enough
                    try:
                        group_chunks = chunks[i]
                    except IndexError:
                        group_chunks = chunks[-1]

                    ds = xds_from_table_impl(table_name, T, table_schema,
                                             dsk, table_key,
                                             columns.difference(group_cols),
                                             group_rows, group_chunks)

                    yield ds.assign_attrs(zip(group_cols, group_values))

        # No grouping case
        else:
            query = ("SELECT ROWID() as __tablerow__ "
                     "FROM $T %s" % orderby_clause(index_cols))

            with pt.taql(query) as gq:
                yield xds_from_table_impl(table_name, T, table_schema,
                                          dsk, table_key,
                                          columns.difference(group_cols),
                                          gq.getcol("__tablerow__"),
                                          chunks[0])


def xds_from_ms(ms, columns=None, index_cols=None, group_cols=None,
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
    group_cols  : tuple or list, optional
        Sequence of grouping columns.
        Defaults to :code:`%(parts)s`
    chunks : list of dicts or dict, optional
        Dictionaries of dimension chunks.

    Yields
    ------
    :class:`xarray.Dataset`
        xarray datasets for each group
    """

    if index_cols is None:
        index_cols = _DEFAULT_INDEX_COLUMNS
    elif isinstance(index_cols, list):
        index_cols = tuple(index_cols)
    elif not isinstance(index_cols, tuple):
        index_cols = (index_cols,)

    if group_cols is None:
        group_cols = _DEFAULT_GROUP_COLUMNS
    elif isinstance(group_cols, list):
        group_cols = tuple(group_cols)
    elif not isinstance(group_cols, tuple):
        group_cols = (group_cols,)

    for ds in xds_from_table(ms, columns=columns,
                             index_cols=index_cols, group_cols=group_cols,
                             table_schema="MS", chunks=chunks):
        yield ds


# Set docstring variables in try/except
# ``__doc__`` may not be present as
# ``python -OO`` strips docstrings
try:
    xds_from_ms.__doc__ %= {
        'index': _DEFAULT_INDEX_COLUMNS,
        'parts': _DEFAULT_GROUP_COLUMNS}
except AttributeError:
    pass
