from __future__ import print_function
from future_builtins import zip

import argparse
import collections
from collections import OrderedDict
from functools import partial
import itertools
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
    from cytoolz import pluck, merge
except ImportError:
    from toolz import pluck, merge

import xarray as xr

from .table_proxy import TableProxy
from .known_table_schemas import registered_schemas

_DEFAULT_INDEX_COLUMNS = ("FIELD_ID", "DATA_DESC_ID", "TIME",
                                        "ANTENNA1", "ANTENNA2")

_DEFAULT_PARTITION_COLUMNS = ("FIELD_ID", "DATA_DESC_ID")

_DEFAULT_ROWCHUNKS = 100000

def consecutive(index, stepsize=1):
    """ Partition index into list of arrays of consecutive indices """
    return np.split(index, np.where(np.diff(index) != stepsize)[0]+1)

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

    head, tail = os.path.split(table_name.rstrip(os.sep))
    token = dask.base.tokenize(tail, column)
    name = '-'.join((tail, "getcol", column.lower(), token))
    dsk = {}

    chunk_extra = shape[1:]
    key_extra = (0,) * len(shape[1:])

    # Iterate through the rows in groups of rowchunks
    # For each iteration we generate one chunk
    # in the resultant dask array
    chunks = []

    get_fn = _np_get_fn if isinstance(dtype, np.dtype) else _list_get_fn

    for chunk, start_row in enumerate(range(0, rows.size, rowchunks)):
        end_row = min(start_row + rowchunks, rows.size)

        # Split into runs of consecutive rows within this chunk
        d = np.ediff1d(rows[start_row:end_row], to_begin=2, to_end=2)
        runs = np.nonzero(d != 1)[0]

        # How many rows in this chunk?
        chunk_size = 0
        # Store a list of lambdas executing getcols on consecutive runs
        row_get_fns = []

        for start, end in zip(runs[:-1], runs[1:]):
            run_len = end - start
            row_get_fns.append((get_fn, table_open_key, column, start, run_len))
            chunk_size += run_len

        # Create the key-value dask entry for this chunk
        dsk[(name, chunk) + key_extra] = (np.concatenate, row_get_fns)
        chunks.append(chunk_size)

    chunks = tuple((tuple(chunks),)) + tuple((c,) for c in chunk_extra)
    return da.Array(merge(dsk_base, dsk), name, chunks, dtype=dtype)

def _np_put_fn(tp, c, d, s, n):
    tp("putcol", c, d, startrow=s, nrow=n)
    return np.asarray([True])

def xds_to_table(xds, table_name, columns=None):
    head, tail = os.path.split(table_name.rstrip(os.sep))
    kwargs = {'readonly': False}
    table_open_key = ('open', tail, dask.base.tokenize(table_name, kwargs))

    table_open = partial(TableProxy, **kwargs)

    dsk = { table_open_key : (table_open, table_name) }
    rows = xds.table_row.values

    if columns is None:
        columns = xds.data_vars.keys()
    elif isinstance(columns, string_types):
        columns = [columns]

    token = dask.base.tokenize(tail, columns)
    name = '-'.join((tail, "putcol", token))

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
                                    run_start, run_end - run_start))

            dsk[(name, chunk)] = (np.logical_and.reduce, row_put_fns)
            chunks.append(1)
            chunk += 1
            chunk_start = chunk_end

    chunks = (tuple(chunks),)
    return da.Array(dsk, name, chunks, dtype=np.bool)

def _xds_from_table(table_name, table, table_schema,
                    dsk, table_open_key,
                    ignore_cols, rows, rowchunks):
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
    ignore_cols : tuple or list
        Columns to ignore when creating the dataset.
    rows : np.ndarray
        CASA table row id's defining an ordering
        of the table data.
    rowchunks : integer
        The chunk size for the row dimension of the resulting
        :class:`dask.array.Array`s.

    Returns
    -------
    :class:`xarray.Dataset`
        xarray dataset
    """

    # The columns we're going to load, ignoring some
    columns = set(table.colnames()).difference(ignore_cols)
    nrows = rows.size
    col_metadata = {}

    missing = []
    schemas = registered_schemas()

    def _search_schemas(table_name, schemas):
        """ Guess the schema from the table name """
        for k in schemas.keys():
            if table_name.endswith('::' + k):
                return schemas[k]

        return {}

    if table_schema is None:
        table_schema = _search_schemas(table_name, schemas)

    # Get a registered table schema
    elif isinstance(table_schema, string_types):
        table_schema = registered_schemas().get(table_schema, {})

    # Attempt to get dimension schema for column
    if not isinstance(table_schema, collections.Mapping):
        raise TypeError("Invalid table_schema type '%s'" % type(table_schema))

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

            # Generate an xarray dimension schema
            # from supplied or inferred schemas if possible
            try:
                extra = table_schema[c].dims
            except KeyError:
                extra = tuple('%s-%d' % (c, i) for i in range(1, len(shape)))

            col_metadata[c] = (shape, ("row",) + extra, dtype)

    # Remove missing columns
    columns = columns.difference(missing)


    # Insert arrays into dataset in sorted order
    data_arrays = OrderedDict()

    for c in sorted(columns):
        shape, dims, dtype = col_metadata[c]
        col_dask_array = generate_table_getcols(table_name, table_open_key,
                                                dsk, c, shape, dtype, rows,
                                                rowchunks)

        data_arrays[c] = xr.DataArray(col_dask_array, dims=dims)

    # Create the dataset, assigning coordinates in the process
    base_rows = np.arange(rows.size, dtype=np.int32)
    return xr.Dataset(data_arrays).assign_coords(table_row=rows, row=base_rows)


def xds_from_table(table_name, index_cols=None, part_cols=None,
                                    table_schema=None,
                                    rowchunks=_DEFAULT_ROWCHUNKS):
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

    rowchunks (optional) : integer
        Number of rows to chunk along

    Yields
    ------
    `xarray.Dataset`
        datasets for each partition, each ordered by indexing columns
    """
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

    head, tail = os.path.split(table_name.rstrip(os.sep))
    table_open_key = ('open', tail, dask.base.tokenize(table_name))
    dsk = { table_open_key : (TableProxy, table_name) }

    def _create_dataset(table, index_cols, group_cols=(), group_values=()):
        """
        Generates a dataset, by generates a row ordering given

        1. the partitioning defined by ``group_cols`` and ``group_values``
        2. the ordering defined by ``index_cols``

        and then deferring to :func:`_xds_from_table` to create the
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

        return _xds_from_table(table_name, table, table_schema,
                            dsk, table_open_key,
                            group_cols, rows, rowchunks)

    with pt.table(table_name) as T:
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
               ds = _create_dataset(T, index_cols, group_cols, group_values)
               yield (ds.squeeze(drop=True)
                        .assign_attrs(table_row=ds.table_row.values[0]))

        # Otherwise partition by give columns
        elif len(part_cols) > 0:
            part_str = ', '.join(part_cols)
            query = "SELECT %s FROM $T GROUP BY %s" % (part_str, part_str)

            with pt.taql(query) as group_query:
                group_cols = group_query.colnames()
                groups = [group_query.getcol(c) for c in group_cols]

            # For each grouping
            for group_values in zip(*groups):
                ds = _create_dataset(T, index_cols, group_cols, group_values)
                yield ds.assign_attrs(zip(group_cols, group_values))

        # No partioning case
        else:
            yield _create_dataset(T, index_cols)

def xds_from_ms(ms, index_cols=None, part_cols=None,
                    rowchunks=_DEFAULT_ROWCHUNKS):
    """
    Constructs an xarray dataset from a Measurement Set
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

    for ds in xds_from_table(ms, index_cols=index_cols, part_cols=part_cols,
                                    table_schema="MS", rowchunks=rowchunks):
        yield ds
