# -*- coding: utf-8 -*-

from functools import reduce
import logging
from operator import mul

import numpy as np
import pyrap.tables as pt

from daskms.columns import infer_dtype
from daskms.descriptors.builder import (register_descriptor_builder,
                                        AbstractDescriptorBuilder)
from daskms.dataset import data_var_dims, DimensionInferenceError


log = logging.getLogger(__name__)


@register_descriptor_builder("ms")
class MSDescriptorBuilder(AbstractDescriptorBuilder):
    INDEX_COLS = ("ARRAY_ID", "DATA_DESC_ID", "FIELD_ID",
                  "OBSERVATION_ID", "PROCESSOR_ID",
                  "SCAN_NUMBER", "STATE_ID")
    DATA_COLS = ("DATA", "MODEL_DATA", "CORRECTED_DATA")

    def __init__(self, fixed=True):
        super(AbstractDescriptorBuilder, self).__init__()
        self.DEFAULT_MS_DESC = pt.required_ms_desc()
        self.REQUIRED_FIELDS = set(self.DEFAULT_MS_DESC.keys())
        self.fixed = fixed
        self.ms_dims = None

    def default_descriptor(self):
        desc = self.DEFAULT_MS_DESC.copy()

        # Imaging DATA columns
        desc.update({column: {
            '_c_order': True,
            'comment': f'The Visibility {column} Column',
            'dataManagerGroup': 'StandardStMan',
            'dataManagerType': 'StandardStMan',
            'keywords': {'UNIT': 'Jy'},
            'maxlen': 0,
            'ndim': 2,  # (chan, corr)
            'option': 0,
            'valueType': 'COMPLEX'}
            for column in self.DATA_COLS})

        desc['IMAGING_WEIGHT'] = {
            '_c_order': True,
            'comment': 'Weight set by imaging task (e.g. uniform weighting)',
            'dataManagerGroup': 'StandardStMan',
            'dataManagerType': 'StandardStMan',
            'keywords': {},
            'maxlen': 0,
            'ndim': 1,  # (chan,)
            'option': 0,
            'valueType': 'FLOAT'}

        desc['WEIGHT_SPECTRUM'] = {
            '_c_order': True,
            'comment': 'Per-channel weights',
            'dataManagerGroup': 'StandardStMan',
            'dataManagerType': 'StandardStMan',
            'keywords': {},
            'maxlen': 0,
            'ndim': 2,  # (chan, corr)
            'option': 0,
            'valueType': 'FLOAT'}

        desc['SIGMA_SPECTRUM'] = {
            '_c_order': True,
            'comment': 'Per-channel sigmas',
            'dataManagerGroup': 'StandardStMan',
            'dataManagerType': 'StandardStMan',
            'keywords': {},
            'maxlen': 0,
            'ndim': 2,  # (chan, corr)
            'option': 0,
            'valueType': 'FLOAT'}

        return desc

    def descriptor(self, column_schemas, default_desc):
        try:
            desc = {k: default_desc[k] for k in self.REQUIRED_FIELDS}
        except KeyError as e:
            raise RuntimeError(f"'{str(e)}' not in REQUIRED_FIELDS")

        # Put indexing columns into an Incremental Storage Manager by default
        for column in self.INDEX_COLS:
            desc[column]['dataManagerGroup'] = 'IndexingGroup'
            desc[column]['dataManagerType'] = 'IncrementalStMan'
            desc[column]['option'] |= 1

        for k, lv in column_schemas.items():
            try:
                desc[k] = default_desc[k]
            except KeyError:
                desc[k] = self.variable_descriptor(k, lv)

        if self.fixed:
            ms_dims = self.infer_ms_dims(column_schemas)
            desc = self.fix_columns(column_schemas, desc, ms_dims)

        return desc

    @staticmethod
    def infer_ms_dims(variables):
        class DummyVar:
            __slots__ = ("dims", "shape")

            def __init__(self, dims, shape):
                self.dims = dims
                self.shape = shape

        def trim_row_dims(column, variables):
            ret_val = []

            for var in variables:
                if var.dims[0] != "row":
                    raise ValueError(f"'row' is not the first "
                                     f"dimension in the dimensions "
                                     f"{var.dims} of column {column}")

                ret_val.append(DummyVar(var.dims[1:], var.shape[1:]))

            return ret_val

        # Create a dictionary of all variables in all datasets
        expanded_vars = {(k, i): v for k, lv in variables.items()
                         for i, v in enumerate(trim_row_dims(k, lv))}

        # Now try find consistent dimension sizes across all variables
        try:
            dim_sizes = data_var_dims(expanded_vars)
        except DimensionInferenceError:
            log.warning("Unable to determine fixed column shapes as "
                        "input variable dimension sizes are inconsistent",
                        exc_info=True)

            return {}
        else:
            return dim_sizes

    @staticmethod
    def _maybe_fix_column(column, desc, shape):
        """ Try set column to fixed if it exists in the descriptor """
        try:
            col_desc = desc[column]
        except KeyError:
            return

        # Can't Tile STRING arrays
        if col_desc['valueType'].upper() == "STRING":
            return

        col_desc['shape'] = shape
        col_desc['ndim'] = len(shape)
        col_desc['option'] |= 4
        col_desc['dataManagerGroup'] = f"{column}_GROUP"
        col_desc['dataManagerType'] = "TiledColumnStMan"

    def fix_columns(self, variables, desc, dim_sizes):
        """ Set large columns to fixed columns """

        # NOTE(JSKenyon): Attempt to make custom (non-standard) columns fixed
        # shape when possible. Only chan and corr dimensions are handled at
        # the moment.
        for col_name, schemas in variables.items():
            # Don't attempt to fix columns with object type.
            if schemas[0].dtype == np.dtype('O'):
                continue

            schema_dims = schemas[0].dims
            # First dim must be row (for now).
            if schema_dims[0] != "row":
                continue
            # Don't bother fixing row-only columns.
            if len(schema_dims) == 1:
                continue

            try:
                shape = tuple([dim_sizes[d] for d in schema_dims[1:]])
            except KeyError:  # We don't understand the dims.
                log.warning(f"Could not fix shape for column {col_name} "
                            f"with dimensions {schema_dims}.")
                continue

            if np.nan in shape:
                log.warning(f"Could not fix shape for column {col_name} "
                            f"with np.nan in shape.")
                continue

            self._maybe_fix_column(col_name, desc, shape)

        return desc

    def _fit_tile_shape(self, desc):
        """ Infer a tile shape """
        try:
            shape = desc['shape']
        except KeyError:
            raise ValueError(f"No shape in descriptor {desc}")
        else:
            rev_shape = tuple(reversed(shape))

        try:
            casa_type = desc['valueType']
        except KeyError:
            raise ValueError(f"No valueType in descriptor {desc}")
        else:
            dtype = infer_dtype(casa_type, desc)
            nbytes = np.dtype(dtype).itemsize

        rows = 1

        while reduce(mul, rev_shape + (2*rows,), 1)*nbytes < 4*1024*1024:
            rows *= 2

        return {"DEFAULTTILESHAPE": np.int32(rev_shape + (2*rows,))}

    def dminfo(self, table_desc):
        """
        Create Data Manager Info for the MS, adding Tiled Shapes to
        all TiledColumnStMan groups.
        """

        dm_groups = {}

        for column, desc in table_desc.items():
            # Ignore keywords
            if column.startswith("_"):
                continue

            dmtype = desc.get('dataManagerType', 'StandardStMan')
            group = desc.get('dataManagerGroup', 'StandardStMan')

            try:
                dm_group = dm_groups[group]
            except KeyError:
                # Create the group
                dm_groups[group] = dm_group = {'COLUMNS': [column],
                                               'NAME': group,
                                               'TYPE': dmtype}

                # Create a tiling SPEC if the group's TYPE is right
                if dmtype.startswith('Tiled'):
                    dm_group['SPEC'] = self._fit_tile_shape(desc)
            else:
                # Sanity check the TYPE
                if dmtype != dm_group['TYPE']:
                    raise TypeError("DataManagerType is not the same "
                                    "across all columns of "
                                    "DataManagerGroup %s" % group)

                dm_group['COLUMNS'].append(column)

        # Now create the dminfo object
        return {'*%d' % (i + 1): dm_group for i, dm_group
                in enumerate(dm_groups.values())}
