# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import dask
import dask.array as da
import numpy as np
import pyrap.tables as pt

from xarrayms.dataset import Variable
from xarrayms.columns import (infer_casa_type, dim_extents_array)
from xarrayms.ordering import row_run_factory
from xarrayms.table import table_exists
from xarrayms.table_proxy import TableProxy, WRITELOCK
from xarrayms.utils import short_table_name

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
    # See https://github.com/ska-sa/xarray-ms/issues/42
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


def _create_table(table, datasets, columns):
    ds = datasets[0]
    data_vars = ds.variables
    row = sum(ds.dims['row'] for ds in datasets)

    coldescs = []

    for k, var in data_vars.items():
        desc = dask_column_descriptor(k, var)
        coldescs.append(desc)

    table_desc = pt.maketabdesc(coldescs)

    table_proxy = TableProxy(pt.table, table, table_desc, ack=False,
                             readonly=False, lockoptions='user')

    table_proxy.addrows(row).result()

    return table_proxy


def _updated_table(table, datasets, columns):
    table_proxy = TableProxy(pt.table, table, ack=False,
                             readonly=False, lockoptions='user')

    table_columns = set(table_proxy.colnames().result())
    missing = set(columns) - table_columns
    first_data_vars = datasets[0].variables

    # Create column metadata for each missing column and
    # add it to the table if necessary
    if len(missing) > 0:
        coldescs = []

        for m in missing:
            desc = dask_column_descriptor(m, first_data_vars[m])
            coldescs.append(desc)

        table_desc = pt.maketabdesc(coldescs)
        table_proxy.addcols(table_desc).result()

    return table_proxy


def dask_column_descriptor(column, variable):
    """
    Generate a CASA column descriptor from a Dataset Variable.

    Parameters
    ----------
    column : str
        Column name
    variable : :class:`xarrayms.dataset.Variable`
        Dataset variable

    Returns
    -------
    dict
        CASA column descriptor
    """

    if isinstance(variable, Variable):
        variable = [variable]
    elif not isinstance(variable, (tuple, list)):
        variable = [variable]

    descs = []

    for v in variable:
        dims, var, _ = v
        ndim = len(dims)

        # Only consider dimensions other than row
        if ndim > 0 and dims[0] == 'row':
            dims = dims[1:]
            shape = var.shape[1:]
            ndim -= 1
        else:
            shape = var.shape

        dtype = var.dtype.type
        casa_type = infer_casa_type(dtype)

        desc = {'_c_order': True,
                'comment': '%s column' % column,
                'dataManagerGroup': '',
                'dataManagerType': '',
                'keywords': {},
                'maxlen': 0,
                'option': 0,
                'valueType': casa_type}

        # An ndim of 0 seems to imply a scalar which is not the
        # same thing as not having dimensions other than row
        if ndim > 0:
            desc['option'] = 4
            desc['shape'] = list(shape)
            desc['ndim'] = ndim

        descs.append({'name': column, 'desc': desc})

    return descs[0]


def write_datasets(table, datasets, columns):
    # Promote datasets to list
    if isinstance(datasets, tuple):
        datasets = list(datasets)
    else:
        datasets = [datasets]

    if not table_exists(table):
        table_proxy = _create_table(table, datasets, columns)
    else:
        table_proxy = _updated_table(table, datasets, columns)

    table_name = short_table_name(table)
    writes = []

    for di, ds in enumerate(datasets):
        try:
            rowid = ds.ROWID
        except AttributeError:
            rowid = da.arange(ds.dims['row'], chunks=ds.chunks['row'])

        row_order = rowid.map_blocks(row_run_factory,
                                     sort_dir="write",
                                     dtype=np.object)
        data_vars = ds.variables

        for column in columns:
            try:
                column_entry = data_vars[column]
            except KeyError:
                log.warning("Ignoring '%s' not present "
                            "on dataset %d" % di)
                continue

            full_dims = column_entry.dims
            array = column_entry.var
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
