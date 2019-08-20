# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import dask
import dask.array as da
from dask.highlevelgraph import HighLevelGraph
import numpy as np
import pyrap.tables as pt

from daskms.columns import dim_extents_array
from daskms.descriptors.builder import AbstractDescriptorBuilder
from daskms.descriptors.builder_factory import filename_builder_factory
from daskms.descriptors.builder_factory import string_builder_factory
from daskms.ordering import row_run_factory
from daskms.table import table_exists
from daskms.table_proxy import TableProxy, WRITELOCK
from daskms.utils import short_table_name


log = logging.getLogger(__name__)


def ndarray_putcol(row_runs, table_proxy, column, data):
    """ Put data into the table """
    putcol = table_proxy._table.putcol
    rr = 0

    table_proxy._acquire(WRITELOCK)

    try:
        for rs, rl in row_runs:
            putcol(column, data[rr:rr + rl], startrow=rs, nrow=rl)
            rr += rl

    finally:
        table_proxy._release(WRITELOCK)


def ndarray_putcolslice(row_runs, blc, trc, table_proxy, column, data):
    """ Put data into the table """
    putcolslice = table_proxy._table.putcolslice
    rr = 0

    table_proxy._acquire(WRITELOCK)

    try:
        for rs, rl in row_runs:
            putcolslice(column, data[rr:rr + rl], blc, trc,
                        startrow=rs, nrow=rl)
            rr += rl

    finally:
        table_proxy._release(WRITELOCK)


def putter_wrapper(row_orders, *args):
    """
    Wrapper which should run I/O operations within
    the table_proxy's associated executor
    """
    # Infer number of shape arguments
    nextent_args = len(args) - 3
    # Extract other arguments
    table_proxy, column, data = args[nextent_args:]

    # Handle dask compute_meta gracefully
    if len(row_orders) == 0:
        return np.empty((0,)*len(data.shape), dtype=np.bool)

    row_runs, resort = row_orders

    if resort is not None:
        data = data[resort]

    # Infer output shape before possible list conversion
    out_shape = (1,) * len(data.shape)

    # NOTE(sjperkins)
    # python-casacore wants to put lists of objects, but
    # because dask.array handles ndarrays we're passed
    # ndarrays of python objects (strings).
    # Without this conversion python-casacore can segfault
    # See https://github.com/ska-sa/dask-ms/issues/42
    if data.dtype == np.object:
        data = data.tolist()

    # There are other dimensions beside row
    if nextent_args > 0:
        blc, trc = zip(*args[:nextent_args])
        table_proxy._ex.submit(ndarray_putcolslice, row_runs, blc, trc,
                               table_proxy, column, data).result()
    else:
        table_proxy._ex.submit(ndarray_putcol, row_runs, table_proxy,
                               column, data).result()

    return np.full(out_shape, True)


def descriptor_builder(table, descriptor):
    if descriptor is None:
        return filename_builder_factory(table)
    elif isinstance(descriptor, AbstractDescriptorBuilder):
        return descriptor
    else:
        return string_builder_factory(descriptor)


def _create_table(table, datasets, columns, descriptor):
    builder = descriptor_builder(table, descriptor)
    table_desc, dminfo = builder.execute(datasets)

    from daskms.descriptors.ms import MSDescriptorBuilder
    from daskms.descriptors.ms_subtable import MSSubTableDescriptorBuilder

    if isinstance(builder, MSDescriptorBuilder):
        # Create the MS
        with pt.default_ms(table, tabdesc=table_desc, dminfo=dminfo):
            pass
    elif isinstance(builder, MSSubTableDescriptorBuilder):
        # Create the MS subtable
        subtable = builder.subtable
        create_dir = short_table_name(table).rstrip(subtable)
        with pt.default_ms_subtable(builder.subtable, create_dir,
                                    tabdesc=table_desc, dminfo=dminfo):
            pass
    else:
        # Create the table
        with pt.table(table, table_desc, dminfo=dminfo, ack=False):
            pass

    return TableProxy(pt.table, table, ack=False,
                      readonly=False, lockoptions='user')


def _updated_table(table, datasets, columns, descriptor):
    table_proxy = TableProxy(pt.table, table, ack=False,
                             readonly=False, lockoptions='user')

    table_columns = set(table_proxy.colnames().result())
    missing = set(columns) - table_columns

    # Add missing columns to the table
    if len(missing) > 0:
        # NOTE(sjperkins)
        # Updating a table with new columns with data managers
        # is a little tricky. Trying to update existing data managers
        # seems to incur casacore's internal wrath.
        #
        # Here, we
        # 1. Build a full table description from all variables
        # 2. Take only the column descriptions for the missing variables.
        # 3. Create Data Managers associated with missing variables,
        #    discarding any that currently exist on the table
        builder = descriptor_builder(table, descriptor)
        variables = builder.dataset_variables(datasets)
        default_desc = builder.default_descriptor()
        table_desc = builder.descriptor(variables, default_desc)
        table_desc = {m: table_desc[m] for m in missing}

        # Original Data Manager Groups
        odminfo = {g['NAME'] for g in table_proxy.getdminfo()
                                                 .result()
                                                 .values()}

        # Construct a dminfo object with Data Manager Groups not present
        # on the original dminfo object
        dminfo = {"*%d" % (i + 1): v for i, v
                  in enumerate(builder.dminfo(table_desc).values())
                  if v['NAME'] not in odminfo}

        # Add the columns
        table_proxy.addcols(table_desc, dminfo=dminfo).result()

    return table_proxy


def update_datasets(table, datasets, columns, descriptor):
    table_proxy = _updated_table(table, datasets, columns, descriptor)
    table_name = short_table_name(table)
    writes = []
    row_orders = []

    # Sort datasets on (not has "ROWID", index) such that
    # datasets with ROWID's are handled first, while
    # those without (which imply appends to the MS)
    # are handled last
    sorted_datasets = sorted(enumerate(datasets),
                             key=lambda t: ("ROWID" not in t[1].data_vars,
                                            t[0]))

    # Establish row orders for each dataset
    for di, ds in sorted_datasets:
        try:
            rowid = ds.ROWID.data
        except AttributeError:
            # No ROWID's, assume they're missing from the table
            # and remaining datasets. Generate addrows
            # NOTE(sjperkins)
            # This could be somewhat brittle, but exists to
            # update of MS subtables once they've been
            # created (empty) along with the main MS by a call to default_ms.
            # Users could also it to append rows to an existing table.
            # An xds_append_to_table is probably the correct solution...
            last_datasets = datasets[di:]
            last_row_orders = add_row_order_factory(table_proxy, last_datasets)
            row_orders.extend(last_row_orders)
            # We have established row orders for all datasets
            # at this point, quit the loop
            break
        else:
            # Generate row orderings from existing row IDs
            row_order = rowid.map_blocks(row_run_factory,
                                         sort_dir="write",
                                         dtype=np.object)
            row_orders.append(row_order)

    assert len(row_orders) == len(datasets)

    for (di, ds), row_order in zip(sorted_datasets, row_orders):
        data_vars = ds.data_vars

        # Generate a dask array for each column
        for column in columns:
            try:
                column_entry = data_vars[column]
            except KeyError:
                log.warning("Ignoring '%s' not present "
                            "on dataset %d" % (column, di))
                continue

            full_dims = column_entry.dims
            array = column_entry.data
            args = [row_order, ("row",)]

            # We only need to pass in dimension extent arrays if
            # there is more than one chunk in any of the non-row columns.
            # In that case, we can putcol, otherwise putcolslice is required
            if not all(len(c) == 1 for c in array.chunks[1:]):
                # Add extent arrays
                for d, c in zip(full_dims[1:], array.chunks[1:]):
                    args.append(dim_extents_array(d, c))
                    args.append((d,))

            # Add other variables
            args.extend([table_proxy, None,
                         column, None,
                         array, full_dims])

            # Name of the dask array representing this column
            token = dask.base.tokenize(di, args)
            name = "-".join((table_name, 'write', column, token))

            write_col = da.blockwise(putter_wrapper, full_dims,
                                     *args,
                                     # All dims shrink to 1,
                                     # a single bool is returned
                                     adjust_chunks={d: 1 for d in full_dims},
                                     name=name,
                                     dtype=np.bool)

            writes.append(write_col.ravel())

    return da.concatenate(writes)


def _add_row_wrapper(table, rows, checkrow=-1):
    startrow = table.nrows()

    if checkrow != -1 and startrow != checkrow:
        raise ValueError("Inconsistent starting row %d %d"
                         % (startrow, checkrow))

    table.addrows(rows)

    return (np.array([[startrow, rows]], dtype=np.int32), None)


def add_row_orders(data, table_proxy, prev=None):
    """
    Adds rows to a table and returns associated row orderings.

    We want to be able to handle undefined row chunk sizes (np.nan) when
    writing dask arrays to a new table. This means that we may not know
    the starting row, or the number of rows in each chunk upfront.
    This poses further challenges when adding new rows to a table.

    This function addresses this by ingesting a chunk of data. From this
    the number of rows in the chunk can be determined and added to the table.
    The starting row and number of rows in the chunk are then returned
    as a result, which is passed as input to the ``add_row_orders`` call,
    operating on an adjacent chunk of row data.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        numpy array from which the number of rows will be derived.
        The first dimension :code:`data.shape[0]`
        should contain the number of rows.
    table_proxy : :class:`daskms.table_proxy.TableProxy`
        Table Proxy object
    prev : tuple or None
        Previous row run array. This argument serves two purposes:

        1. It is used to determine the *starting row* of the current
           row ordering.
        2. When this function is embedded in a dask graph, it establishes a
           dependency on the dask task which creates the previous rows.

        Defaults to ``None``. If ``None``, assumes we're adding rows to
        an empty table. :code:`([[0, rows]], None)` is returned.

        If a :code:`(row_run, resort)` tuple, the first entry in
        the starting row of the previous ``row_run`` is added
        to it's length to produce the starting row of the current row run.

    Returns
    -------
    :class:`numpy.ndarray`
        Row runs of shape :code:`(nruns, 2)` where the first component
        contains the starting row and the last, the number of rows.
    None
        Indicate that row resorting should not occur by default
    """
    rows = data.shape[0]

    # This is the first link in the chain
    if prev is None:
        return (table_proxy.submit(_add_row_wrapper,
                                   WRITELOCK, rows, -1)
                .result())
    else:
        # There's a previous link in the chain
        prev_runs, _ = prev
        startrow = prev_runs.sum()

        return (table_proxy.submit(_add_row_wrapper,
                                   WRITELOCK, rows, startrow)
                .result())


def add_row_order_factory(table_proxy, datasets):
    """
    Generate arrays which add the appropriate rows for each array row chunk
    of a dataset, as well as returning the appropriate row ordering
    for that chunk

    Each array chunk (and by implication dataset) is linked by a call to
    :func:`daskms.writes.add_row_orders` to a previous chunk,
    either in the same array or the previous array.

    This establishes an order on how:

    1. Rows are added to the table.
    2. Column writes are performed.

    Returns
    -------
    list of :class:`dask.array.Array`
        row orderings for each dataset
    """
    prev_key = None
    prev_deps = []
    row_add_ops = []

    for di, ds in enumerate(datasets):
        data_vars = ds.data_vars
        found = False

        for k, (dims, array, _) in data_vars.items():
            # Need something with a row dimension
            if not dims[0] == 'row':
                continue

            found = True
            token = dask.base.tokenize(array)
            name = '-'.join(('add-rows', str(di), token))
            layers = {}

            for b in range(array.numblocks[0]):
                key = (name, b)
                array_key = (array.name, b) + (0,)*(array.ndim - 1)
                layers[key] = (add_row_orders, array_key,
                               table_proxy, prev_key)
                prev_key = key

            graph = HighLevelGraph.from_collections(name, layers,
                                                    prev_deps + [array])
            chunks = (array.chunks[0],)
            row_adds = da.Array(graph, name, chunks, dtype=np.object)
            row_add_ops.append(row_adds)
            prev_deps = [row_adds]

            break

        if not found:
            raise ValueError("Couldn't find an array with "
                             "which to establish a row ordering "
                             "in dataset %d" % di)

    return row_add_ops


def create_datasets(table_name, datasets, columns, descriptor):
    """
    Create new dataset
    """
    table_proxy = _create_table(table_name, datasets, columns, descriptor)
    row_orders = add_row_order_factory(table_proxy, datasets)
    short_name = short_table_name(table_name)
    writes = []

    for di, (ds, row_order) in enumerate(zip(datasets, row_orders)):
        data_vars = ds.data_vars

        for column in columns:
            try:
                (dims, array, _) = data_vars[column]
            except KeyError:
                log.warn("Column %s doesn't exist on dataset %d "
                         "and will be ignored" % (column, di))
                continue

            args = [row_order, ("row",)]

            # We only need to pass in dimension extent arrays if
            # there is more than one chunk in any of the non-row columns.
            # In that case, we can putcol, otherwise putcolslice is required
            if not all(len(c) == 1 for c in array.chunks[1:]):
                # Add extent arrays
                for d, c in zip(dims[1:], array.chunks[1:]):
                    args.append(dim_extents_array(d, c))
                    args.append((d,))

            # Add other variables
            args.extend([table_proxy, None,
                         column, None,
                         array, dims])

            # Name of the dask array representing this column
            token = dask.base.tokenize(di, args)
            name = "-".join((short_name, 'write', column, token))

            write_col = da.blockwise(putter_wrapper, dims,
                                     *args,
                                     # All dims shrink to 1,
                                     # a single bool is returned
                                     adjust_chunks={d: 1 for d in dims},
                                     name=name,
                                     dtype=np.bool)

            # Flatten the writes so that they can be simply
            # concatenated together into a final aggregated array
            writes.append(write_col.ravel())

    return da.concatenate(writes)


def write_datasets(table, datasets, columns, descriptor=None):
    # Promote datasets to list
    if isinstance(datasets, tuple):
        datasets = list(datasets)
    elif not isinstance(datasets, list):
        datasets = [datasets]

    # If no columns are defined, write all dataset variables by default
    if not columns:
        columns = set.union(*(set(ds.data_vars.keys()) for ds in datasets))
        columns = list(sorted(columns))

    if not table_exists(table):
        return create_datasets(table, datasets, columns,
                               descriptor=descriptor)
    else:
        return update_datasets(table, datasets, columns,
                               descriptor=descriptor)
