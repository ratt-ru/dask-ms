import attr

TableSchema = attr.make_class("TableSchema", ["dims"])

# https://casa.nrao.edu/Memos/229.html#SECTION00061000000000000000
MS_SCHEMA = {
    "TIME":             TableSchema(()),
    "TIME_EXTRA_PREC":  TableSchema(()),
    "ANTENNA1":         TableSchema(()),
    "ANTENNA2":         TableSchema(()),
    "ANTENNA3":         TableSchema(()),
    "FEED1":            TableSchema(()),
    "FEED2":            TableSchema(()),
    "FEED3":            TableSchema(()),
    "DATA_DESC_ID":     TableSchema(()),
    "PROCESSOR_ID":     TableSchema(()),
    "PHASE_ID":         TableSchema(()),
    "FIELD_ID":         TableSchema(()),
    "INTERVAL":         TableSchema(()),
    "EXPOSURE":         TableSchema(()),
    "TIME_CENTROID":    TableSchema(()),
    "PULSAR_BIN":       TableSchema(()),
    "PULSAR_GATE":      TableSchema(()),
    "SCAN_NUMBER":      TableSchema(()),
    "ARRAY_ID":         TableSchema(()),
    "OBSERVATION_ID":   TableSchema(()),
    "STATE_ID":         TableSchema(()),
    "BASELINE_REF":     TableSchema(()),
    "UVW":              TableSchema(('(u,v,w)',)),
    "UVW2":             TableSchema(('(u,v,w)',)),
    "DATA":             TableSchema(('chans', 'corrs')),
    "FLOAT_DATA":       TableSchema(('chans', 'corrs')),
    "VIDEO_POINT":      TableSchema(('corrs',)),
    "LAG_DATA":         TableSchema(('corrs',)),
    "SIGMA":            TableSchema(('corrs',)),
    "SIGMA_SPECTRUM":   TableSchema(('chans', 'corrs')),
    "WEIGHT":           TableSchema(('corrs',)),
    "WEIGHT_SPECTRUM":  TableSchema(('chans', 'corrs')),
    "FLAG":             TableSchema(('chans', 'corrs')),
    "FLAG_CATEGORY":    TableSchema(('flagcats', 'chans', 'corrs')),
    "FLAG_ROWS":        TableSchema(()),

    # Extra imaging columns
    "MODEL_DATA":       TableSchema(('chans', 'corrs')),
    "CORRECTED_DATA":   TableSchema(('chans', 'corrs')),
    "IMAGING_WEIGHT":   TableSchema(('chans',)),
}

ANTENNA_SCHEMA = {
    "POSITION":         TableSchema(('(x,y,z)',)),
    "OFFSET":           TableSchema(('(x,y,z)',)),
}

FIELD_SCHEMA = {
    "DELAY_DIR":        TableSchema(('dir', 'poly+1')),
    "PHASE_DIR":        TableSchema(('dir', 'poly+1')),
    "REFERENCE_DIR":    TableSchema(('dir', 'poly+1')),
}

SPECTRAL_WINDOW = {
    "CHAN_FREQ":        TableSchema(('nchan',)),
    "CHAN_WIDTH":       TableSchema(('nchan',)),
    "EFFECTIVE_BW":     TableSchema(('nchan',)),
    "RESOLUTION":       TableSchema(('nchan',)),
}

def registered_schemas():
    return {
        "MS" : MS_SCHEMA,
        "ANTENNA": ANTENNA_SCHEMA,
        "FIELD": FIELD_SCHEMA,
        "SPECTRAL_WINDOW": SPECTRAL_WINDOW,
    }