# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import mul

import numpy as np
import pyrap.tables as pt

from six.moves import reduce

# Map Measurement Set string types to numpy types
MS_TO_NP_TYPE_MAP = {
    'INT': np.int32,
    'FLOAT': np.float32,
    'DOUBLE': np.float64,
    'BOOLEAN': np.bool,
    'COMPLEX': np.complex64,
    'DCOMPLEX': np.complex128
}


def _dm_spec(coldesc, tile_mem_limit=4*1024*1024):
    """
    Create data manager spec for a given column description,
    by adding a DEFAULTTILESHAPE that fits within the memory limit
    """

    # Get the reversed column shape. DEFAULTTILESHAPE is deep in
    # casacore and its necessary to specify their ordering here
    # ntilerows is the dim that will change least quickly
    rev_shape = list(reversed(coldesc["shape"]))

    if None in rev_shape:
        return {}

    ntilerows = 1
    np_dtype = MS_TO_NP_TYPE_MAP[coldesc["valueType"].upper()]
    nbytes = np.dtype(np_dtype).itemsize

    # Try bump up the number of rows in our tiles while they're
    # below the memory limit for the tile

    while reduce(mul, rev_shape + [2*ntilerows], nbytes) < tile_mem_limit:
        ntilerows *= 2

    return {"DEFAULTTILESHAPE": np.int32(rev_shape + [ntilerows])}


def _update_col(table_desc, dm_group_spec,
                column, manager_group,
                options, shape):
    kw = {'ndim': len(shape), 'dataManagerGroup': manager_group}

    can_tile = None not in shape

    if can_tile:
        # If we have a fixed shape set the fixed shape flag
        # and use a tile storage manager
        kw.update(option=options | 4, shape=shape,
                  dataManagerType='TiledColumnStMan')
    else:
        # Unset the fixed shape flag
        kw.update(option=options & ~4,
                  dataManagerType="StandardStMan")

    table_desc[column].update(kw),

    if can_tile:
        dm_group_spec[manager_group] = _dm_spec(table_desc[column])


def _new_col(column, manager_group, default,
             shape, valuetype, options, **kw):

    kw.update(datamanagergroup=manager_group)

    can_tile = None not in shape

    # Use a tiled storage manager and a fixed shape column
    if can_tile:
        kw.update(shape=shape, datamanagertype='TiledColumnStMan')
        options |= 4
    # Use standard storage manager and a variably shaped column
    else:
        kw.update(datamanagertype='StandardStMan')
        options &= ~4

    desc = pt.tablecreatearraycoldesc(column, default, ndim=len(shape),
                                      valuetype=valuetype,
                                      options=options, **kw)

    if can_tile:
        return desc, {manager_group: _dm_spec(desc['desc'])}
    else:
        return desc, {}


def _ms_desc_and_dm_info(nchan, ncorr, add_imaging_cols=False):
    """
    Creates Table Description and Data Manager Information objects that
    describe a MeasurementSet.

    Creates additional DATA, IMAGING_WEIGHT and possibly
    MODEL_DATA and CORRECTED_DATA columns.

    Columns are given fixed shapes defined by the arguments to this function.

    Parameters
    ----------
    nchan : int
        Nimber of channels
    ncorr : int
        Number of correlations
    add_imaging_cols : bool, optional
        Add imaging columns. Defaults to False.

    Returns
    -------
    table_spec : dict
        Table specification dictionary
    dm_info : dict
        Data Manager Information dictionary
    """

    # Columns that will be modified.
    # We want to keep things like their keywords,
    # but modify their shapes, dimensions, options and data managers
    modify_columns = {"WEIGHT", "SIGMA", "FLAG", "FLAG_CATEGORY",
                      "UVW", "ANTENNA1", "ANTENNA2"}

    # Get the required table descriptor for an MS
    table_desc = pt.required_ms_desc("MAIN")

    # Take columns we wish to modify
    extra_table_desc = {c: d for c, d in table_desc.items()
                        if c in modify_columns}

    # Used to set the SPEC for each Data Manager Group
    dmgroup_spec = {}

    # Update existing columns with shape and data manager information
    _update_col(extra_table_desc, dmgroup_spec,
                "UVW", "Uvw", 1, [3])
    _update_col(extra_table_desc, dmgroup_spec,
                "WEIGHT", "Weight", 0, [ncorr])
    _update_col(extra_table_desc, dmgroup_spec,
                "SIGMA", "Sigma", 0, [ncorr])
    _update_col(extra_table_desc, dmgroup_spec,
                "FLAG", "Flag", 0, [nchan, ncorr])
    _update_col(extra_table_desc, dmgroup_spec,
                "FLAG_CATEGORY", "FlagCategory", 0, [1, nchan, ncorr])

    # Create new columns for integration into the MS
    additional_columns = []

    col_desc, dm_spec = _new_col("DATA", "Data", 0+0j,
                                 [nchan, ncorr], 'complex', options=0,
                                 comment="The Visibility DATA Column",
                                 keywords={"UNIT": "Jy"})
    additional_columns.append(col_desc)
    dmgroup_spec.update(dm_spec)

    col_desc, dm_spec = _new_col("WEIGHT_SPECTRUM", "WeightSpectrum", 1.0,
                                 [nchan, ncorr], 'float', options=0,
                                 comment="Per-channel weights",
                                 keywords={"UNIT": "Jy"})

    # Add Imaging Columns, if requested
    if add_imaging_cols:
        col_desc, dm_spec = _new_col("IMAGING_WEIGHT", "ImagingWeight", 0.0,
                                     [nchan], 'float', options=0,
                                     comment="Weight set by imaging task "
                                              "(e.g. uniform weighting)")
        additional_columns.append(col_desc)
        dmgroup_spec.update(dm_spec)

        col_desc, dm_spec = _new_col("MODEL_DATA", "ModelData", 0+0j,
                                     [nchan, ncorr], 'complex', options=0,
                                     comment="Model Visibilities",
                                     keywords={"UNIT": "Jy"})
        additional_columns.append(col_desc)
        dmgroup_spec.update(dm_spec)

        col_desc, dm_spec = _new_col("CORRECTED_DATA", "CorrectedData", 0+0j,
                                     [nchan, ncorr], 'complex', options=0,
                                     comment="Corrected Visibilities",
                                     keywords={"UNIT": "Jy"})
        additional_columns.append(col_desc)
        dmgroup_spec.update(dm_spec)

    # Update extra table description with additional columns
    extra_table_desc.update(pt.maketabdesc(additional_columns))

    # Update the original table descriptor with modifications/additions
    # Need this to construct a complete Data Manager specification
    # that includes the original columns
    table_desc.update(extra_table_desc)
    # pprint(table_desc)

    # Construct DataManager Specification
    dminfo = pt.makedminfo(table_desc, dmgroup_spec)

    return extra_table_desc, dminfo


def empty_ms(ms_name, nchan, ncorr, add_imaging_cols):
    """
    Creates an empty Measurement Set with Fixed Column Shapes.
    Unlikely to work with multiple SPECTRAL_WINDOW's with
    different shapes, or multiple POLARIZATIONS
    with different correlations.

    Interface likely to change to somehow support this in future.

    Parameters
    ----------
    ms_name : str
        Measurement Set filename
    nchan : int
        Number of channels
    ncorr : int
        Number of correlations
    add_imaging_cols : bool
        True if ``MODEL_DATA``, ``CORRECTED_DATA`` and ``IMAGING_WEIGHTS``
        columns should be added.
    """

    table_desc, dm_info = _ms_desc_and_dm_info(nchan, ncorr, add_imaging_cols)

    with pt.default_ms(ms_name, table_desc, dm_info):
        pass
