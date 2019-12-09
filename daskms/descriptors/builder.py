# -*- coding: utf-8 -*-

import abc
from ast import parse
from collections import defaultdict
import logging

from daskms.columns import infer_casa_type
from daskms.dataset import Variable


log = logging.getLogger(__name__)

descriptor_builders = {}


def is_valid_variable_name(name):
    try:
        parse('{} = None'.format(name))
        return True
    except (SyntaxError, ValueError, TypeError):
        return False


def register_descriptor_builder(name):
    def decorator(cls):
        if name in descriptor_builders:
            raise ValueError("'%s' already registered as a "
                             "descriptor builder" % name)

        if not is_valid_variable_name(name):
            raise ValueError("'%s' is not a valid "
                             "Python variable name." % name)

        descriptor_builders[name] = cls
        return cls

    return decorator


class AbstractDescriptorBuilder(object, metaclass=abc.ABCMeta):
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

    @staticmethod
    def dataset_variables(datasets):
        variables = defaultdict(list)

        if not isinstance(datasets, list):
            datasets = list(datasets)

        for ds in datasets:
            for column, variable in ds.data_vars.items():
                variables[column].append(variable)

        return variables

    def execute(self, datasets):
        default_desc = self.default_descriptor()
        variables = self.dataset_variables(datasets)
        table_desc = self.descriptor(variables, default_desc)
        dminfo = self.dminfo(table_desc)

        return table_desc, dminfo


class DefaultDescriptorBuilder(AbstractDescriptorBuilder):
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
    Generate a CASA column descriptor from a Variable
    or list of Variables.

    Parameters
    ----------
    column : str
        Column name
    variable : :class:`daskms.Variable` or \
       list of :class:`daskms.Variable`

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
