# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

from six import string_types
from daskms.utils import short_table_name

# https://casa.nrao.edu/Memos/229.html#SECTION00061000000000000000
MS_SCHEMA = {
    'UVW': {
        'dask': {'dims': ('uvw',)},
    },
    'UVW2': {
        'dask': {'dims': ('uvw',)},
    },
    'DATA': {
        'dask': {'dims': ('chan', 'corr')},
    },
    'FLOAT_DATA': {
        'dask': {'dims': ('chan', 'corr')},
    },
    'SIGMA': {
        'dask': {'dims': ('corr',)},
    },
    'SIGMA_SPECTRUM': {
        'dask': {'dims': ('chan', 'corr')},
    },
    'WEIGHT': {
        'dask': {'dims': ('corr',)},
    },
    'WEIGHT_SPECTRUM': {
        'dask': {'dims': ('chan', 'corr')},
    },
    'FLAG': {
        'dask': {'dims': ('chan', 'corr')},
    },
    'FLAG_CATEGORY': {
        'dask': {'dims': ('flagcat', 'chan', 'corr')},
    },

    # Extra imaging columns
    'MODEL_DATA': {
        'dask': {'dims': ('chan', 'corr')},
    },
    'CORRECTED_DATA': {
        'dask': {'dims': ('chan', 'corr')},
    },
    'IMAGING_WEIGHT': {
        'dask': {'dims': ('chan',)},
    }
}

ANTENNA_SCHEMA = {
    "POSITION": {
        'dask': {'dims': ('xyz',)},
    },
    "OFFSET": {
        'dask': {'dims': ('xyz',)},
    },
}

FIELD_SCHEMA = {
    "DELAY_DIR": {
        'dask': {'dims': ('field-dir', 'field-poly')},
    },
    "PHASE_DIR": {
        'dask': {'dims': ('field-dir', 'field-poly')},
    },
    "REFERENCE_DIR": {
        'dask': {'dims': ('field-dir', 'field-poly')},
    },
}

SPECTRAL_WINDOW = {
    "CHAN_FREQ": {
        'dask': {'dims': ('chan',)},
    },
    "CHAN_WIDTH": {
        'dask': {'dims': ('chan',)},
    },
    "EFFECTIVE_BW": {
        'dask': {'dims': ('chan',)},
    },
    "RESOLUTION": {
        'dask': {'dims': ('chan',)},
    },
}

POLARIZATION = {
    "CORR_TYPE": {
        'dask': {'dims': ('corr',)},
    },
    "CORR_PRODUCT": {
        'dask': {'dims': ('corr', 'corrprod_idx')},
    },
}

_SUBTABLE_SCHEMAS = {
    "ANTENNA": ANTENNA_SCHEMA,
    "FIELD": FIELD_SCHEMA,
    "SPECTRAL_WINDOW": SPECTRAL_WINDOW,
    "POLARIZATION": POLARIZATION,
}

_ALL_SCHEMAS = {
    "MS": MS_SCHEMA
}

_ALL_SCHEMAS.update(_SUBTABLE_SCHEMAS)
_ALL_SCHEMAS["TABLE"] = {}


def infer_table_type(table_name):
    """ Guess the schema from the table name """
    if table_name[-3:].upper().endswith(".MS"):
        return "MS"

    for k in _SUBTABLE_SCHEMAS.keys():
        if table_name.endswith('::' + k):
            return k

    return "TABLE"


def lookup_table_schema(table_name, lookup_str):
    """
    Attempts to heuristically generate a table schema dictionary,
    given the ``lookup_str`` argument. If this fails,
    an empty dictionary is returned.

    Parameters
    ----------
    table_name : str
        CASA table path
    lookup_str : str or ``None``
        If a string, the resulting schema will be
        internally looked up in the known table schemas.
        If ``None``, the end of ``table_name`` will be
        inspected to perform the lookup.

    Returns
    -------
    dict
        A dictionary of the form
        :code:`{column: {'dask': {...}, 'casa': {...}}}`.
    """
    if lookup_str is None:
        table_type = infer_table_type(short_table_name(table_name))

        try:
            return _ALL_SCHEMAS[table_type]
        except KeyError:
            raise ValueError("No schema registered for "
                             "table type '%s'" % table_type)

    if not isinstance(lookup_str, (tuple, list)):
        lookup_str = [lookup_str]

    table_schema = {}

    for ls in lookup_str:
        if isinstance(ls, Mapping):
            table_schema.update(ls)
        # Get a registered table schema
        elif isinstance(ls, string_types):
            table_schema.update(_ALL_SCHEMAS.get(ls, {}))
        else:
            raise TypeError("Invalid lookup_str type '%s'" % type(ls))

    return table_schema
