"""
Measurement Set columns usually have fixed dimensions.
Give them names here so that the xarray Dataset can
align shared dimensions, 'chan', for instance.

dtype's are also a possibility here, but valueType
seems to be present in the Column Description.
"""

import attr

ColumnSchema = attr.make_class("ColumnSchema", ["dims"])

# https://casa.nrao.edu/Memos/229.html#SECTION00061000000000000000
MS_SCHEMA = {
    "TIME":             ColumnSchema(()),
    "TIME_EXTRA_PREC":  ColumnSchema(()),
    "ANTENNA1":         ColumnSchema(()),
    "ANTENNA2":         ColumnSchema(()),
    "ANTENNA3":         ColumnSchema(()),
    "FEED1":            ColumnSchema(()),
    "FEED2":            ColumnSchema(()),
    "FEED3":            ColumnSchema(()),
    "DATA_DESC_ID":     ColumnSchema(()),
    "PROCESSOR_ID":     ColumnSchema(()),
    "PHASE_ID":         ColumnSchema(()),
    "FIELD_ID":         ColumnSchema(()),
    "INTERVAL":         ColumnSchema(()),
    "EXPOSURE":         ColumnSchema(()),
    "TIME_CENTROID":    ColumnSchema(()),
    "PULSAR_BIN":       ColumnSchema(()),
    "PULSAR_GATE":      ColumnSchema(()),
    "SCAN_NUMBER":      ColumnSchema(()),
    "ARRAY_ID":         ColumnSchema(()),
    "OBSERVATION_ID":   ColumnSchema(()),
    "STATE_ID":         ColumnSchema(()),
    "BASELINE_REF":     ColumnSchema(()),
    "UVW":              ColumnSchema(('(u,v,w)',)),
    "UVW2":             ColumnSchema(('(u,v,w)',)),
    "DATA":             ColumnSchema(('chans', 'corrs')),
    "FLOAT_DATA":       ColumnSchema(('chans', 'corrs')),
    "VIDEO_POINT":      ColumnSchema(('corrs',)),
    "LAG_DATA":         ColumnSchema(('corrs',)),
    "SIGMA":            ColumnSchema(('corrs',)),
    "SIGMA_SPECTRUM":   ColumnSchema(('chans', 'corrs')),
    "WEIGHT":           ColumnSchema(('corrs',)),
    "WEIGHT_SPECTRUM":  ColumnSchema(('chans', 'corrs')),
    "FLAG":             ColumnSchema(('chans', 'corrs')),
    "FLAG_CATEGORY":    ColumnSchema(('flagcats', 'chans', 'corrs')),
    "FLAG_ROWS":        ColumnSchema(()),

    # Extra imaging columns
    "MODEL_DATA":       ColumnSchema(('chans', 'corrs')),
    "CORRECTED_DATA":   ColumnSchema(('chans', 'corrs')),
    "IMAGING_WEIGHT":   ColumnSchema(('chans',)),
}

ANTENNA_SCHEMA = {
    "POSITION":         ColumnSchema(('(x,y,z)',)),
    "OFFSET":           ColumnSchema(('(x,y,z)',)),
}

FIELD_SCHEMA = {
    "DELAY_DIR":        ColumnSchema(('dir', 'poly+1')),
    "PHASE_DIR":        ColumnSchema(('dir', 'poly+1')),
    "REFERENCE_DIR":    ColumnSchema(('dir', 'poly+1')),
}

SPECTRAL_WINDOW = {
    "CHAN_FREQ":        ColumnSchema(('chans',)),
    "CHAN_WIDTH":       ColumnSchema(('chans',)),
    "EFFECTIVE_BW":     ColumnSchema(('chans',)),
    "RESOLUTION":       ColumnSchema(('chans',)),
}

def registered_schemas():
    return {
        "MS" : MS_SCHEMA,
        "ANTENNA": ANTENNA_SCHEMA,
        "FIELD": FIELD_SCHEMA,
        "SPECTRAL_WINDOW": SPECTRAL_WINDOW,
    }