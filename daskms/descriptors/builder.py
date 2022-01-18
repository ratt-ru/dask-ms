# -*- coding: utf-8 -*-

import abc
from ast import parse
from collections import defaultdict
from daskms.dataset_schema import ColumnSchema, DatasetSchema
import logging

from daskms.columns import infer_casa_type


log = logging.getLogger(__name__)

descriptor_builders = {}


def is_valid_variable_name(name):
    try:
        parse(f'{name} = None')
        return True
    except (SyntaxError, ValueError, TypeError):
        return False


def register_descriptor_builder(name):
    def decorator(cls):
        if name in descriptor_builders:
            raise ValueError(f"'{name}' already registered as "
                             f"a descriptor builder")

        if not is_valid_variable_name(name):
            raise ValueError(f"'{name}' is not a valid Python variable name.")

        descriptor_builders[name] = cls
        return cls

    return decorator


class AbstractDescriptorBuilder(metaclass=abc.ABCMeta):
    @staticmethod
    def variable_descriptor(column, column_schema):
        return variable_column_descriptor(column, column_schema)

    @abc.abstractmethod
    def default_descriptor(self):
        pass

    @abc.abstractmethod
    def descriptor(self, column_schemas, default_desc):
        pass

    @abc.abstractmethod
    def dminfo(self, table_desc):
        pass

    @staticmethod
    def dataset_variables(schemas):
        variables = defaultdict(list)

        if not isinstance(schemas, list):
            schemas = list(schemas)

        if not all(isinstance(s, DatasetSchema) for s in schemas):
            raise TypeError(f"'schemas' must be a DataSchema or "
                            f"a tuple/list of  DatasetSchemas "
                            f"Got {list(map(type, schemas))}.")

        for schema in schemas:
            for column, variable in schema.data_vars.items():
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

    def descriptor(self, column_schemas, default_desc):
        desc = default_desc

        for k, v in column_schemas.items():
            try:
                desc[k] = default_desc[k]
            except KeyError:
                desc[k] = self.variable_descriptor(k, v)

        return desc

    def dminfo(self, table_desc):
        return {}


def variable_column_descriptor(column, column_schema):
    """
    Generate a CASA column descriptor from a ColumnSchema
    or list of ColumnSchemas.

    Parameters
    ----------
    column : str
        Column name
    variable : :class:`daskms.data_schema.ColumnSchema` or \
       list of :class:`daskms.data_schema.ColumnSchema`

        Dataset variable

    Returns
    -------
    dict
        CASA column descriptor
    """

    if isinstance(column_schema, ColumnSchema):
        column_schema = [column_schema]
    elif not isinstance(column_schema, (tuple, list)):
        column_schema = [column_schema]

    dtypes = set()
    ndims = set()
    shapes = set()

    for v in column_schema:
        dtypes.add(v.dtype)

        if v.ndim == 0:
            # Scalar array, ndim == 0 in numpy and CASA
            ndims.append(0)
        else:
            if not v.dims[0] == "row":
                log.warning(f"Column {column} doesn't start with "
                            f"with a 'row' dimension: {v.dims} "
                            f"and will be ignored")
                continue

            # Row only, so ndim must be removed from the descriptor
            # Add a marker to distinguish in case of multiple
            # shapes
            if v.ndim == 1:
                ndims.add('row')
            # Other dims, add dimension data, excluding the row
            else:
                ndims.add(v.ndim - 1)
                shapes.add(v.shape[1:])

    # Fail on multiple dtypes
    if len(dtypes) > 1:
        raise TypeError("Inconsistent data types %s found in dataset "
                        "variables for column %s. CASA Table columns "
                        "must have a single type" % (list(dtypes), column))

    casa_type = infer_casa_type(dtypes.pop())

    desc = {'_c_order': True,
            'comment': f'{column} column',
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
        log.warning(f"Multiple dimensions {ndims} found in dataset variables "
                    f"for column {column}. You appear to be attempting "
                    f"something exotic!")

        desc['ndim'] = -1

    return desc
