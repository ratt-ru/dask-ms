# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from xarrayms.columns import infer_casa_type
from xarrayms.dataset import Variable

_descriptor_plugins = {}


def register_descriptor_plugin(name):
    def decorator(cls):
        _descriptor_plugins[name] = cls
        return cls

    return decorator


@six.add_metaclass(abc.ABCMeta)
class Plugin(object):
    @staticmethod
    def variable_descriptor(column, variable):
        return variable_column_descriptor(column, variable)

    @abc.abstractmethod
    def default_descriptor(self):
        pass

    @abc.abstractmethod
    def descriptor(self, variables, default_desc):
        pass

    @abc.abstractmethod
    def dminfo(self, table_desc):
        pass


class DefaultPlugin(Plugin):
    def default_descriptor(self):
        return {}

    def descriptor(self, variables, default_desc):
        desc = default_desc

        for k, v in variables.items():
            try:
                desc[k] = default_desc[k]
            except KeyError:
                desc[k] = self.variable_descriptor(k, v)

        return desc

    def dminfo(self, table_desc):
        return {}


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

    return desc

