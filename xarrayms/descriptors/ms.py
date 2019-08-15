# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyrap.tables as pt

from xarrayms.descriptors.plugin import register_descriptor_plugin, Plugin


@register_descriptor_plugin("ms")
class MeasurementSetPlugin(Plugin):
    INDEX_COLS = ("ARRAY_ID", "DATA_DESC_ID", "FIELD_ID",
                  "OBSERVATION_ID", "PROCESSOR_ID",
                  "SCAN_NUMBER", "STATE_ID")
    DATA_COLS = ("DATA", "MODEL_DATA", "CORRECTED_DATA")

    def __init__(self):
        super(Plugin, self).__init__()
        self.DEFAULT_MS_DESC = pt.required_ms_desc()
        self.REQUIRED_FIELDS = set(self.DEFAULT_MS_DESC.keys())

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

        # Put indexing columns into an
        # Incremental Storage Manager by default
        for column in self.INDEX_COLS:
            desc[column]['dataManagerGroup'] = 'IndexingGroup'
            desc[column]['dataManagerType'] = 'IncrementalStMan'
            desc[column]['option'] |= 1

        for k, v in variables.items():
            try:
                desc[k] = default_desc[k]
            except KeyError:
                desc[k] = self.variable_descriptor(k, v)

        return desc

    def dminfo(self, table_desc):
        return {}
