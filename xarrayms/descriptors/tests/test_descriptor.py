# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from xarrayms.descriptors.dminfo import dminfo_factory


def test_dminfo_factory():
    col_descs = {
        "DATA": {
            'dataManagerGroup': 'Data',
            'dataManagerType': 'TiledColumnStMan',
        },
        "MODEL_DATA": {
            'dataManagerGroup': 'Data',
            'dataManagerType': 'TiledColumnStMan',
        },
        "OTHER": {
            'dataManagerGroup': 'Other',
            'dataManagerType': 'StandardStMan',
        },
        # Default case
        "FOO": {}
    }

    group_spec = {
        'Data': {'DEFAULTTILE_SHAPE': [16, 64, 4]},
        'Other': {'DEFAULTTILE_SHAPE': [16]},
    }

    expected = {
        '*1': {'COLUMNS': ['DATA', 'MODEL_DATA'],
               'NAME': 'Data',
               'SEQNR': 0,
               'SPEC': {'DEFAULTTILE_SHAPE': [16, 64, 4]},
               'TYPE': 'TiledColumnStMan'},
        '*2': {'COLUMNS': ['OTHER'],
               'NAME': 'Other',
               'SEQNR': 1,
               'SPEC': {'DEFAULTTILE_SHAPE': [16]},
               'TYPE': 'StandardStMan'},
        '*3': {'COLUMNS': ['FOO'],
               'NAME': 'StandardStMan',
               'SEQNR': 2,
               'SPEC': {},
               'TYPE': 'StandardStMan'}}

    assert dminfo_factory(col_descs, group_spec) == expected

    # Test list form
    col_descs = [{'name': k, 'desc': v} for k, v in col_descs.items()]
    assert dminfo_factory(col_descs, group_spec) == expected
