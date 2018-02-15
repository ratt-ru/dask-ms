from __future__ import print_function
from future_builtins import zip

import collections
from functools import partial
import logging
import os
import os.path
import time

import dask
import dask.array as da
import numba
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

_DEFAULT_PARTITION_COLUMNS = ("FIELD_ID", "DATA_DESC_ID")
_DEFAULT_INDEX_COLUMNS = ("FIELD_ID", "DATA_DESC_ID", "TIME",)

_DEFAULT_ROWCHUNKS = 100000

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
        Shortenend path

    """
    return os.path.split(table_name.rstrip(os.sep))[1]

def table_open_graph(table_name, **kwargs):
    """
    Generate a dask graph containing table open commands

    Parameters
    ----------
    table_name : str
        CASA table name
    **kwargs (optional) :
        Keywords arguments passed to the :meth:`pyrap.tables.table`
        constructor, for e.g. :code:`readonly=False`

    Returns
    -------
    tuple
        Graph key associated with the opened table
    dict
        Dask graph containing the graph open command

    """
    token = dask.base.tokenize(table_name, kwargs)
    table_open_key = ('open', short_table_name(table_name), token)
    dsk = { table_open_key: (partial(TableProxy, **kwargs), table_name) }
    return table_open_key, dsk

@numba.jit
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
    columns (optional): tuple or list
        list of column names to write to the table.
        If ``None`` all columns will be written.

    Returns
    -------
    :class:`dask.array.Array`
        dask array representing the write to the
        datset.
    """

    table_open_key, dsk = table_open_graph(table_name, readonly=False)
    rows = xds.table_row.values

    if columns is None:
        columns = xds.data_vars.keys()
    elif isinstance(columns, string_types):
        columns = [columns]

    token = dask.base.tokenize(table_name, columns)
    name = '-'.join((short_table_name(table_name), "putcol", token))

    chunk = 0
    chunks = []
    row_idx = 0

    for c in columns:
        data_array = getattr(xds, c)
        dask_array = data_array.data

        dsk.update(dask_array.__dask_graph__())

        array_chunks = data_array.chunks
        array_shape = data_array.shape

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
                row_put_fns.append((_np_put_fn, table_open_key,
                                    c, dsk_slice.keys()[0],
                                    rows[run_start], run_end - run_start))

            dsk[(name, chunk)] = (np.logical_and.reduce, row_put_fns)
            chunks.append(1)
            chunk += 1
            chunk_start = chunk_end

    chunks = (tuple(chunks),)
    return da.Array(dsk, name, chunks, dtype=np.bool)

# jit some getcol wrapper calls
@numba.jit
def _np_get_fn(tp, c, s, n):
    return tp("getcol", c, startrow=s, nrow=n)

@numba.jit
def _list_get_fn(tp, c, s, n):
    return np.asarray(_np_get_fn(tp, c, s, n))

def generate_table_getcols(table_name, table_open_key, dsk_base,
                            column, shape, dtype, rows, rowchunks):
    """
    Generates a :class:`dask.array.Array` representing ``column``
    in ``table_name`` and backed by a series of
    `pyrap.tables.table.getcol` commands.


    Parameters
    ----------
    table_name : str
        CASA table filename path
    table_open_key : tuple
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

    token = dask.base.tokenize(table_name, column)
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
            row_get_fns.append((get_fn, table_open_key, column,
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
                    dsk, table_open_key,
                    columns, rows, chunks):
    """
    Parameters
    ----------
    table_name : str
        CASA table filename path
    table : :class:`pyrap.tables.table`
        CASA table object, used to inspect metadata
        for creating Datasets
    table_schema : str or dict
        Table schema.
    dsk : dict
        Dask graph containing ``table_open_key``
    table_open_key : tuple
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
            logging.warn("Ignoring '%s' column", c, exc_info=True)
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
        col_dask_array = generate_table_getcols(table_name, table_open_key,
                                                dsk, c, shape, dtype, rows,
                                                rowchunks)

        data_arrays[c] = xr.DataArray(col_dask_array, dims=dims)

    # Create the dataset, assigning coordinates in the process
    base_rows = np.arange(rows.size, dtype=np.int32)
    return xr.Dataset(data_arrays).assign_coords(table_row=rows, row=base_rows)


def xds_from_table(table_name, columns=None,
                    index_cols=None, part_cols=None,
                    table_schema=None, chunks=None):
    """
    Generator producing :class:`xarray.Dataset`(s) from the CASA table
    ``table_name`` with the rows lexicographically sorted according
    to the columns in ``index_cols``.
    If ``part_cols`` is supplied, the table data is partitioned into
    multiple :class:`xarray.Dataset`(s), each associated with a
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

    ..code-block:: python

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

    ..code-block:: python

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
    columns (optional) : list or tuple
        Columns present on the returned dataset.
        Defaults to all if ``None``
    index_cols (optional) : list or tuple
        List of CASA table indexing columns. Defaults to :code:`()`.
    part_cols (optional) : list or tuple
        List of columns on which to partition the CASA table.
        Defaults to :code:`()`
    table_schema (optional) : str or dict
        A schema dictionary defining the dimension naming scheme for
        each column in the table. For example:

        .. code-block:: python

            {"UVW" : ('uvw',), DATA" : ('chan', 'corr')}

        will result in the UVW and DATA arrays having dimensions
        :code:`('row', 'uvw')` and :code:`('row', 'chan', 'corr')`
        respectively.

        Alternatively a string can be supplied, which will be matched
        against existing default schemas. Examples here include
        :code:`"MS"`, :code`"ANTENNA"` and :code:`"SPECTRAL_WINDOW"`
        correspoonding to ``Measurement Sets`` the ``ANTENNA`` subtable
        and the ``SPECTRAL_WINDOW`` subtable, respectively.

        If ``None`` is supplied, the end of ``table_name`` will be
        inspected to see if it matches any default schemas.

    chunks (optional) : dict
        A :code:`{dim: chunk}` dictionary, specifying the chunking
        strategy for each dimension in the schema.
        Defaults to :code:`{'row': 100000 }`.

    Yields
    ------
    `xarray.Dataset`
        datasets for each partition, each ordered by indexing columns
    """
    if chunks is None:
        chunks = {'row': _DEFAULT_ROWCHUNKS }

    if index_cols is None:
        index_cols = ()
    elif isinstance(index_cols, list):
        index_cols = tuple(index_cols)
    elif not isinstance(index_cols, tuple):
        index_cols = (index_cols,)

    if part_cols is None:
        part_cols = ()
    elif isinstance(part_cols, list):
        part_cols = tuple(part_cols)
    elif not isinstance(part_cols, tuple):
        part_cols = (part_cols,)

    table_open_key, dsk = table_open_graph(table_name)

    def _create_dataset(table, columns, index_cols,
                        group_cols=(), group_values=()):
        """
        Generates a dataset, given:

        1. the partitioning defined by ``group_cols`` and ``group_values``
        2. the ordering defined by ``index_cols``

        and then deferring to :func:`xds_from_table_impl` to create the
        dataset with the row ordering.
        """
        if len(index_cols) > 0:
            orderby_clause = ', '.join(index_cols)
            orderby_clause = "ORDER BY %s" % orderby_clause
        else:
            orderby_clause = ''

        assert len(group_cols) == len(group_values)

        if len(group_cols) == 1 and group_cols[0] == "__row__":
            where_clause = 'WHERE ROWID()=%s' % group_values[0]
        elif len(group_cols) > 0:
            where_clause = ' AND '.join('%s=%s' % (c,v) for c, v
                                    in zip(group_cols, group_values))
            where_clause = 'WHERE %s' % where_clause
        else:
            where_clause = ''

        # Discover the row indices producing the
        # requested ordering for each group
        query = ("SELECT ROWID() AS __table_row__ FROM $table {wc} {oc}"
                    .format(oc=orderby_clause, wc=where_clause))

        with pt.taql(query) as row_query:
            rows = row_query.getcol("__table_row__")

        return xds_from_table_impl(table_name, table, table_schema,
                            dsk, table_open_key,
                            set(columns).difference(group_cols),
                            rows, chunks)

    with pt.table(table_name) as T:
        if columns is None:
            columns = T.colnames()

        # Group table_name by partitioning columns,
        # We'll generate a dataset for each unique group

        # Handle the case where we partition on each table row
        if len(part_cols) == 1 and part_cols[0] == "__row__":
            query = "SELECT ROWID() AS __row__ FROM $T"

            with pt.taql(query) as group_query:
                group_cols = group_query.colnames()
                groups = [group_query.getcol(c) for c in group_cols]

            # For each grouping
            for group_values in zip(*groups):
               ds = _create_dataset(T, columns, index_cols,
                                    group_cols, group_values)
               yield (ds.squeeze(drop=True)
                        .assign_attrs(table_row=ds.table_row.values[0]))

        # Otherwise partition by given columns
        elif len(part_cols) > 0:
            part_str = ', '.join(part_cols)
            query = "SELECT %s FROM $T GROUP BY %s" % (part_str, part_str)

            with pt.taql(query) as group_query:
                group_cols = group_query.colnames()
                groups = [group_query.getcol(c) for c in group_cols]

            # For each grouping
            for group_values in zip(*groups):
                ds = _create_dataset(T, columns, index_cols,
                                        group_cols, group_values)
                yield ds.assign_attrs(zip(group_cols, group_values))

        # No partioning case
        else:
            yield _create_dataset(T, columns, index_cols)

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
    columns (optional) : tuple or list
        Columns present on the resulting dataset.
        Defaults to all if ``None``.
    index_cols (optional) : tuple or list
        Sequence of indexing columns.
        Defaults to :code:`%(index)s`
    part_cols (optional) : tuple or list
        Sequence of partioning columns.
        Defaults to :code:`%(parts)s`
    chunks (optional) : dict
        Dictionary of dimension chunks.

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
        'parts': _DEFAULT_PARTITION_COLUMNS }
except AttributeError:
    pass
