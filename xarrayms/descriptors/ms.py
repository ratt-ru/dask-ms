# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import logging
from operator import mul

import numpy as np
import pyrap.tables as pt
from six.moves import reduce

from xarrayms.columns import infer_dtype
from xarrayms.descriptors.plugin import register_descriptor_plugin, Plugin
from xarrayms.dataset import data_var_dims, data_var_chunks


log = logging.getLogger(__name__)


@register_descriptor_plugin("ms")
class MeasurementSetPlugin(Plugin):
    INDEX_COLS = ("ARRAY_ID", "DATA_DESC_ID", "FIELD_ID",
                  "OBSERVATION_ID", "PROCESSOR_ID",
                  "SCAN_NUMBER", "STATE_ID")
    DATA_COLS = ("DATA", "MODEL_DATA", "CORRECTED_DATA")

    def __init__(self, fixed=True):
        super(Plugin, self).__init__()
        self.DEFAULT_MS_DESC = pt.required_ms_desc()
        self.REQUIRED_FIELDS = set(self.DEFAULT_MS_DESC.keys())
        self.fixed = fixed

    def default_descriptor(self):
        desc = self.DEFAULT_MS_DESC.copy()

        # Imaging DATA columns
        desc.update({column: {
            '_c_order': True,
            'comment': 'The Visibility %s Column' % column,
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

    def descriptor(self, variables, default_desc):
        try:
            desc = {k: default_desc[k] for k in self.REQUIRED_FIELDS}
        except KeyError as e:
            raise RuntimeError("'%s' not in REQUIRED_FIELDS" % str(e))

        # Put indexing columns into an Incremental Storage Manager by default
        for column in self.INDEX_COLS:
            desc[column]['dataManagerGroup'] = 'IndexingGroup'
            desc[column]['dataManagerType'] = 'IncrementalStMan'
            desc[column]['option'] |= 1

        for k, lv in variables.items():
            try:
                desc[k] = default_desc[k]
            except KeyError:
                desc[k] = self.variable_descriptor(k, lv)

        if self.fixed:
            desc = self.fix_columns(variables, desc)

        return desc

    @staticmethod
    def _maybe_fix_column(column, desc, shape):
        try:
            col_desc = desc[column]
        except KeyError:
            pass
        else:
            col_desc['shape'] = shape
            col_desc['ndim'] = len(shape)
            col_desc['option'] |= 4
            col_desc['dataManagerGroup'] = "%s_GROUP" % column
            col_desc['dataManagerType'] = "TiledColumnStMan"

    def fix_columns(self, variables, desc):
        # We need to find sizes of the channel and correlation dimensions
        expanded_vars = {v.var.name: v for k, lv in variables.items()
                         for v in lv}

        try:
            dim_sizes = data_var_dims(expanded_vars)
        except ValueError as e:
            log.warning("Unable fix column shapes as input variable"
                        "dimension sizes are inconsistent.",
                        exc_info=True)
            return desc

        try:
            chan = dim_sizes['chan']
        except KeyError:
            log.warning("Unable to infer 'chan' dimension "
                        "from input variables")
            return desc
        else:
            self._maybe_fix_column('IMAGING_WEIGHT', desc, (chan,))

        try:
            corr = dim_sizes['corr']
        except KeyError:
            log.warning("Unable to infer 'corr' dimension "
                        "from input variables")
            return desc

        self._maybe_fix_column('FLAG', desc, (chan, corr))
        self._maybe_fix_column('DATA', desc, (chan, corr))
        self._maybe_fix_column('MODEL_DATA', desc, (chan, corr))
        self._maybe_fix_column('CORRECTED_DATA', desc, (chan, corr))
        self._maybe_fix_column('WEIGHT_SPECTRUM', desc, (chan, corr))
        self._maybe_fix_column('SIGMA_SPECTRUM', desc, (chan, corr))

        try:
            flagcat = dim_sizes['flagcat']
        except KeyError:
            log.warning("Unable to infer 'flagcat' dimension "
                        "from input variables")
        else:
            self._maybe_fix_column("FLAG_CATEGORY", desc,
                                   (flagcat, chan, corr))

        return desc

    def _fit_tile_shape(self, desc):
        try:
            shape = desc['shape']
        except KeyError:
            raise ValueError("No shape in descriptor %s" % desc)
        else:
            rev_shape = list(reversed(shape))

        try:
            casa_type = desc['valueType']
        except KeyError:
            raise ValueError("No valueType in descriptor %s" % desc)
        else:
            dtype = infer_dtype(casa_type, desc)
            nbytes = np.dtype(dtype).itemsize

        rows = 1

        while reduce(mul, rev_shape + [2*rows], 1)*nbytes < 4*1024*1024:
            rows *= 2

        return {"DEFAULTTILESHAPE": np.int32(rev_shape + [2*rows])}

    def dminfo(self, table_desc):
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
                dm_groups[group] = dm_group = {
                    'COLUMNS': [column],
                    'NAME': group,
                    'TYPE': dmtype,
                }

                if dmtype.startswith('Tiled'):
                    dm_group['SPEC'] = self._fit_tile_shape(desc)
            else:
                if not dmtype == dm_group['TYPE']:
                    raise TypeError("DataManagerType is not the same "
                                    "across all columns of "
                                    "DataManagerGroup %s" % group)
                dm_group['COLUMNS'].append(column)

        dminfo = {}

        for i, (group, dm_group) in enumerate(dm_groups.items()):
            dminfo['*%d' % (i + 1)] = dm_group

        return dminfo
