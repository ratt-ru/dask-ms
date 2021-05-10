# -*- coding: utf-8 -*-

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

from daskms.utils import table_path_split

# Measurement Set sub-tables
SUBTABLES = ("ANTENNA", "DATA_DESCRIPTION", "DOPPLER",
             "FEED", "FIELD", "FLAG_CMD", "FREQ_OFFSET",
             "HISTORY", "OBSERVATION", "POINTING", "POLARIZATION",
             "PROCESSOR", "SOURCE", "SPECTRAL_WINDOW", "STATE",
             "SYSCAL", "WEATHER")


# https://casa.nrao.edu/Memos/229.html#SECTION00061000000000000000
MS_SCHEMA = {
    'UVW': {'dims': ('uvw',)},
    'UVW2': {'dims': ('uvw',)},
    'DATA': {'dims': ('chan', 'corr')},
    'FLOAT_DATA': {'dims': ('chan', 'corr')},
    'SIGMA': {'dims': ('corr',)},
    'SIGMA_SPECTRUM': {'dims': ('chan', 'corr')},
    'WEIGHT': {'dims': ('corr',)},
    'WEIGHT_SPECTRUM': {'dims': ('chan', 'corr')},
    'FLAG': {'dims': ('chan', 'corr')},
    'FLAG_CATEGORY': {'dims': ('flagcat', 'chan', 'corr')},
    # Extra imaging columns
    'MODEL_DATA': {'dims': ('chan', 'corr')},
    'CORRECTED_DATA': {'dims': ('chan', 'corr')},
    'IMAGING_WEIGHT': {'dims': ('chan',)},
    # Extra WSClean imaging columns
    'IMAGING_WEIGHT_SPECTRUM': {'dims': ('chan', 'corr')},
}

ANTENNA_SCHEMA = {
    "POSITION": {'dims': ('xyz',)},
    "OFFSET": {'dims': ('xyz',)},
}

FEED_SCHEMA = {
    "BEAM_OFFSET": {'dims': ('receptors', 'radec')},
    "POLARIZATION_TYPE": {'dims': ('receptors',)},
    "POL_RESPONSE": {'dims': ('receptors', 'receptors-2')},
    "POSITION": {'dims': ("xyz",)},
    "RECEPTOR_ANGLE": {'dims': ("receptors",)},
}


FIELD_SCHEMA = {
    "DELAY_DIR": {'dims': ('field-poly', 'field-dir')},
    "PHASE_DIR": {'dims': ('field-poly', 'field-dir')},
    "REFERENCE_DIR": {'dims': ('field-poly', 'field-dir')},
}

OBSERVATION_SCHEMA = {
    "LOG": {'dims': ('log',)},
    "SCHEDULE": {'dims': ('schedule',)},
    "TIME_RANGE": {'dims': ('obs-exts',)},
}


POINTING_SCHEMA = {
    "DIRECTION": {'dims': ('point-poly', 'radec')},
    "ENCODER": {'dims': ('radec',)},
    "POINTING_OFFSET": {'dims': ('point-poly', 'radec')},
    "SOURCE_OFFSET": {'dims': ('point-poly', 'radec')},
    "TARGET": {'dims': ('point-poly', 'radec')},
}


POLARIZATION_SCHEMA = {
    "CORR_TYPE": {'dims': ('corr',)},
    "CORR_PRODUCT": {'dims': ('corr', 'corrprod_idx')},
}

SPECTRAL_WINDOW_SCHEMA = {
    "CHAN_FREQ": {'dims': ('chan',)},
    "CHAN_WIDTH": {'dims': ('chan',)},
    "EFFECTIVE_BW": {'dims': ('chan',)},
    "RESOLUTION": {'dims': ('chan',)},
}

SOURCE_SCHEMA = {
    "DIRECTION": {'dims': ('radec',)},
    "POSITION": {'dims': ('position',)},
    "PROPER_MOTION": {'dims': ('radec_per_sec',)},
    "REST_FREQUENCY": {'dims': ('lines',)},
    "SYSVEL": {'dims': ('lines',)},
    "TRANSITION": {'dims': ('lines',)},
}

_SUBTABLE_SCHEMAS = {
    "ANTENNA": ANTENNA_SCHEMA,
    "FEED": FEED_SCHEMA,
    "FIELD": FIELD_SCHEMA,
    "OBSERVATION": OBSERVATION_SCHEMA,
    "POINTING": POINTING_SCHEMA,
    "POLARIZATION": POLARIZATION_SCHEMA,
    "SOURCE": SOURCE_SCHEMA,
    "SPECTRAL_WINDOW": SPECTRAL_WINDOW_SCHEMA,
}

_ALL_SCHEMAS = {
    "MS": MS_SCHEMA
}

_ALL_SCHEMAS.update(_SUBTABLE_SCHEMAS)
_ALL_SCHEMAS["TABLE"] = {}


def infer_table_type(table_name):
    """ Guess the schema from the table name """
    _, table, subtable = table_path_split(table_name)

    if not subtable and table[-3:].upper().endswith(".MS"):
        return "MS"

    if subtable in _SUBTABLE_SCHEMAS.keys():
        return subtable

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
        :code:`{column: {'dims': (...)}}`.
    """
    if lookup_str is None:
        table_type = infer_table_type(table_name)

        try:
            return _ALL_SCHEMAS[table_type]
        except KeyError:
            raise ValueError(f"No schema registered "
                             f"for table type '{table_type}'")

    if not isinstance(lookup_str, (tuple, list)):
        lookup_str = [lookup_str]

    table_schema = {}

    for ls in lookup_str:
        if isinstance(ls, Mapping):
            table_schema.update(ls)
        # Get a registered table schema
        elif isinstance(ls, str):
            table_schema.update(_ALL_SCHEMAS.get(ls, {}))
        else:
            raise TypeError(f"Invalid lookup_str type '{type(ls)}'")

    return table_schema
