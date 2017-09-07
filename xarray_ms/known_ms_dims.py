# https://casa.nrao.edu/Memos/229.html#SECTION00061000000000000000
MS_SCHEMA = {
    "TIME":             ('nrow'),
    "TIME_EXTRA_PREC":  ('nrow'),
    "ANTENNA1":         ('nrow',),
    "ANTENNA2":         ('nrow',),
    "ANTENNA3":         ('nrow',),
    "FEED1":            ('nrow',),
    "FEED2":            ('nrow',),
    "FEED3":            ('nrow',),
    "DATA_DESC_ID":     ('nrow',),
    "PROCESSOR_ID":     ('nrow',),
    "PHASE_ID":         ('nrow',),
    "FIELD_ID":         ('nrow',),
    "INTERVAL":         ('nrow',),
    "EXPOSURE":         ('nrow',),
    "TIME_CENTROID":    ('nrow',),
    "PULSAR_BIN":       ('nrow',),
    "PULSAR_GATE":      ('nrow',),
    "SCAN_NUMBER":      ('nrow',),
    "ARRAY_ID":         ('nrow',),
    "OBSERVATION_ID":   ('nrow',),
    "STATE_ID":         ('nrow',),
    "BASELINE_REF":     ('nrow',),
    "UVW":              ('nrow', 'u,v,w'),
    "UVW2":             ('nrow', 'u,v,w'),
    "DATA":             ('nrow', 'nchan', 'ncorr'),
    "MODEL_DATA":       ('nrow', 'nchan', 'ncorr'),
    "CORRECTED_DATA":   ('nrow', 'nchan', 'ncorr'),
    "FLOAT_DATA":       ('nrow', 'nchan', 'ncorr'),
    "VIDEO_POINT":      ('nrow', 'ncorr'),
    "LAG_DATA":         ('nrow', 'ncorr'),
    "SIGMA":            ('nrow', 'ncorr'),
    "SIGMA_SPECTRUM":   ('nrow', 'nchan', 'ncorr'),
    "FLAG":             ('nrow', 'nchan', 'ncorr'),
    "FLAG_CATEGORY":    ('nrow', 'nflagcat', 'nchan', 'ncorr'),
    "FLAG_ROW":         ('nrow',),
}


def registered_schemas():
    return {
        "MEASUREMENTSET" : MS_SCHEMA
    }