# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict, namedtuple
import logging

from xarrayms.dataset import Variable
from xarrayms.columns import infer_casa_type

log = logging.getLogger(__name__)


def variable_column_descriptor(column, variable):
    """
    Generate a CASA column descriptor from a Dataset Variable
    or list of Dataset Variables.

    Parameters
    ----------
    column : str
        Column name
    variable : :class:`xarrayms.dataset.Variable` or \
       list of :class:`xarrayms.dataset.Variable`

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

    dtypes = set()
    ndims = set()
    shapes = set()

    for v in variable:
        dtypes.add(v.dtype)

        if v.ndim == 0:
            # Scalar array, ndim == 0 in numpy and CASA
            ndims.append(0)
        elif v.dims[0] == 'row':
            # Row only, so ndim must be removed from the descriptor
            # Add a marker to distinguish in case of multiple
            # shapes
            if len(v.dims) == 1:
                ndims.add('row')
            # Other dims, add dimension data, excluding the row
            else:
                ndims.add(v.ndim - 1)
                shapes.add(v.shape[1:])
        else:
            # No row prefix, add dimension and shape
            ndims.add(v.dim)
            shapes.add(v.shape)

    # Fail on multiple dtypes
    if len(dtypes) > 1:
        raise TypeError("Inconsistent data types %s found in dataset "
                        "variables for column %s. CASA Table columns "
                        "must have a single type" % (list(dtypes), column))

    casa_type = infer_casa_type(dtypes.pop())

    desc = {'_c_order': True,
            'comment': '%s column' % column,
            'dataManagerGroup': 'StandardStMan',
            'dataManagerType': 'StandardStMan',
            'keywords': {},
            'maxlen': 0,
            'option': 0,
            'valueType': casa_type}

    if len(ndims) == 0:
        raise ValueError("No dimensionality information found")
    elif len(ndims) == 1:
        # Add any non-row dimension information to the descriptor
        ndim = ndims.pop()

        if ndim != 'row':
            desc['ndim'] = ndim

            # If there's only one shape, we can create a FixedShape column
            if len(shapes) == 1:
                shape = shapes.pop()
                assert len(shape) == ndim
                desc['shape'] = shape
    else:
        # Anything goes...
        log.warn("Multiple dimensions %s found in dataset variables "
                 "for column %s. You appear to be attempting something "
                 "exotic!", list(ndims), column)

        desc['ndim'] = -1

    return {'name': column, 'desc': desc}


DataManagerGroup = namedtuple("DataManagerGroup", ["columns", "spec", "type"])


def dminfo_factory(column_descs, dm_group_spec=None):
    """
    Creates a dminfo dictionary from a series of column descriptions
    and a data manager group specification.

    Rationale: A column descriptor for the DATA + MODEL_DATA columns
    which share a data manager might look as follows:

    .. code-block:: python

        desc {
            "DATA": {
                '_c_order': True,
                'comment': 'Data column',
                'dataManagerGroup': 'TiledData',
                'dataManagerType': 'TiledColumnStMan',
                'keywords': {},
                'maxlen': 0,
                'option': 0,
                'ndim': 2,
                'shape': array([64, 4], dtype=int32),
                'valueType': 'complex'},
            "MODEL_DATA": {
                ...
                'dataManagerGroup': 'Data',
                'dataManagerType': 'TiledColumnStMan',
                ...
            }
        }

    while the associated entry in dminfo might look as follows:

    .. code-block:: python

     '*3': {
        'COLUMNS': ['DATA', 'MODEL_DATA'],
        'NAME': 'TiledData',
        'SEQNR': 2,
        'SPEC': {
            'DEFAULTTILESHAPE': array([ 4, 64, 16], dtype=int32),
            'HYPERCUBES': {'*1': {
                'BucketSize': 32768,
                'CellShape': array([ 4, 64], dtype=int32),
                'CubeShape': array([   4,   64, 6552], dtype=int32),
                'ID': {},
                'TileShape': array([ 4, 64, 16], dtype=int32)}},
            'MAXIMUMCACHESIZE': 0,
            'MaxCacheSize': 0,
            'SEQNR': 2},
        'TYPE': 'TiledColumnStMan'},

    As there can be a many to one relationship between column descriptions
    (``column_desc``) and dminfo entries, this function allows the user
    to specify the :code:`{NAME: SPEC}` in ``dm_group_spec``

    Parameters
    ----------
    column_descs : dict or list
        Either a :code:`{name: desc}` dictionary or a
        :code:`[{'name': column, 'desc': desc}] list
    dm_group_spec:
        A :code:`{data_manager_name: data_manager_spec}` dictionary.
        For example:

        .. code-block:: python

            {'TiledData': {'DEFAULTTILESHAPE': np.array([4, 64, 16]}}

    Returns
    -------
    dict
        A dminfo dictionary
    """
    dm_group_spec = dm_group_spec or {}

    # Convert list to dict
    if isinstance(column_descs, list):
        try:
            column_descs = {e['name']: e['desc'] for e in column_descs}
        except KeyError:
            raise ValueError("column_descs was provided as a list, but "
                             "doesn't seem to contain a sequence of "
                             "{'name': column, 'desc': desc} entries.")
    elif not isinstance(column_descs, dict):
        raise TypeError("column_descs must be a dict or a list of dicts")

    dm_groups = OrderedDict()

    for column, desc in column_descs.items():
        # Handle empty strings/None
        # StandardStMan is default for both group and type
        dm_group = desc.get("dataManagerGroup", None) or "StandardStMan"
        dm_type = desc.get("dataManagerType", None) or "StandardStMan"

        try:
            group_obj = dm_groups[dm_group]
        except KeyError:
            group_obj = DataManagerGroup([column],
                                         dm_group_spec.get(dm_group, {}),
                                         dm_type)

            dm_groups[dm_group] = group_obj
        else:
            if group_obj.type != dm_type:
                raise ValueError("Mismatched dataManagerType '%s' "
                                 "for dataManagerGroup '%s' "
                                 "Previously the type was '%s' "
                                 "in columns '%s'" %
                                 (dm_type, dm_group,
                                  group_obj.type, group_obj.columns))

            group_obj.columns.append(column)

    return {
        '*%d' % (i+1): {
            'NAME': group,
            'SEQNR': i,
            'COLUMNS': dm_group.columns,
            'TYPE': dm_group.type,
            'SPEC': dm_group.spec,
        } for i, (group, dm_group) in enumerate(dm_groups.items())
    }
