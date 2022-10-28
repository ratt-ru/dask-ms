from daskms import xds_from_ms, xds_from_table
from daskms.casa_schema import CasaSchema

import numpy as np


def test_casa_unified_schema_main(example_ms):
    datasets = xds_from_ms(example_ms)
    assert len(datasets) == 2

    schema = CasaSchema.from_datasets(datasets)

    for ds in datasets:
        for column, var in ds.data_vars.items():
            s = schema.data_vars[column]
            assert s.dims == var.dims[1:]
            # assert s.shape == {v.shape[1:] for c, v in ds.data_vars.items() if c == column}
            assert np.dtype(s.dtype) == var.dtype
            assert isinstance(var.data, s.type)
            assert s.attrs == var.attrs

    assert schema.to_casa_schema() == {
        "ANTENNA1": {
            "_c_order": True,
            "comment": "ANTENNA1 column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "INTEGER",
        },
        "ANTENNA2": {
            "_c_order": True,
            "comment": "ANTENNA2 column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "INTEGER",
        },
        "ARRAY_ID": {
            "_c_order": True,
            "comment": "ARRAY_ID column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "INTEGER",
        },
        "DATA": {
            "_c_order": True,
            "comment": "DATA column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "ndim": 2,
            "valueType": "COMPLEX",
        },
        "EXPOSURE": {
            "_c_order": True,
            "comment": "EXPOSURE column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {"QuantumUnits": ["s"]},
            "maxlen": 0,
            "option": 0,
            "valueType": "DOUBLE",
        },
        "FEED1": {
            "_c_order": True,
            "comment": "FEED1 column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "INTEGER",
        },
        "FEED2": {
            "_c_order": True,
            "comment": "FEED2 column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "INTEGER",
        },
        "FLAG_ROW": {
            "_c_order": True,
            "comment": "FLAG_ROW column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "BOOLEAN",
        },
        "INTERVAL": {
            "_c_order": True,
            "comment": "INTERVAL column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {"QuantumUnits": ["s"]},
            "maxlen": 0,
            "option": 0,
            "valueType": "DOUBLE",
        },
        "OBSERVATION_ID": {
            "_c_order": True,
            "comment": "OBSERVATION_ID column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "INTEGER",
        },
        "PROCESSOR_ID": {
            "_c_order": True,
            "comment": "PROCESSOR_ID column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "INTEGER",
        },
        "SCAN_NUMBER": {
            "_c_order": True,
            "comment": "SCAN_NUMBER column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "INTEGER",
        },
        "STATE_ID": {
            "_c_order": True,
            "comment": "STATE_ID column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "INTEGER",
        },
        "TIME": {
            "_c_order": True,
            "comment": "TIME column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {
                "MEASINFO": {"Ref": "UTC", "type": "epoch"},
                "QuantumUnits": ["s"],
            },
            "maxlen": 0,
            "option": 0,
            "valueType": "DOUBLE",
        },
        "TIME_CENTROID": {
            "_c_order": True,
            "comment": "TIME_CENTROID column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {
                "MEASINFO": {"Ref": "UTC", "type": "epoch"},
                "QuantumUnits": ["s"],
            },
            "maxlen": 0,
            "option": 0,
            "valueType": "DOUBLE",
        },
        "UVW": {
            "_c_order": True,
            "comment": "UVW column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {
                "MEASINFO": {"Ref": "ITRF", "type": "uvw"},
                "QuantumUnits": ["m", "m", "m"],
            },
            "maxlen": 0,
            "ndim": 1,
            "option": 0,
            "shape": (3,),
            "valueType": "DOUBLE",
        },
    }


def test_casa_unified_schema_spw(example_ms):
    datasets = xds_from_table(f"{example_ms}::SPECTRAL_WINDOW")
    assert len(datasets) == 1

    schema = CasaSchema.from_datasets(datasets)

    for ds in datasets:
        for column, var in ds.data_vars.items():
            s = schema.data_vars[column]
            assert s.dims == var.dims[1:]
            assert s.shape == {var.shape[1:]}
            assert np.dtype(s.dtype) == var.dtype
            assert isinstance(var.data, s.type)
            assert s.attrs == var.attrs

    np.testing.assert_equal(
        schema.to_casa_schema(),
        {
            "CHAN_FREQ": {
                "_c_order": True,
                "comment": "CHAN_FREQ column",
                "dataManagerGroup": "StandardStMan",
                "dataManagerType": "StandardStMan",
                "keywords": {
                    "MEASINFO": {
                        "TabRefCodes": np.array(
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 64], dtype=np.uint32
                        ),
                        "TabRefTypes": [
                            "REST",
                            "LSRK",
                            "LSRD",
                            "BARY",
                            "GEO",
                            "TOPO",
                            "GALACTO",
                            "LGROUP",
                            "CMB",
                            "Undefined",
                        ],
                        "VarRefCol": "MEAS_FREQ_REF",
                        "type": "frequency",
                    },
                    "QuantumUnits": ["Hz"],
                },
                "maxlen": 0,
                "ndim": 1,
                "option": 0,
                "shape": (16,),
                "valueType": "DOUBLE",
            },
            "CHAN_WIDTH": {
                "_c_order": True,
                "comment": "CHAN_WIDTH column",
                "dataManagerGroup": "StandardStMan",
                "dataManagerType": "StandardStMan",
                "keywords": {"QuantumUnits": ["Hz"]},
                "maxlen": 0,
                "ndim": 1,
                "option": 0,
                "shape": (16,),
                "valueType": "DOUBLE",
            },
            "FLAG_ROW": {
                "_c_order": True,
                "comment": "FLAG_ROW column",
                "dataManagerGroup": "StandardStMan",
                "dataManagerType": "StandardStMan",
                "keywords": {},
                "maxlen": 0,
                "option": 0,
                "valueType": "BOOLEAN",
            },
            "FREQ_GROUP": {
                "_c_order": True,
                "comment": "FREQ_GROUP column",
                "dataManagerGroup": "StandardStMan",
                "dataManagerType": "StandardStMan",
                "keywords": {},
                "maxlen": 0,
                "option": 0,
                "valueType": "INTEGER",
            },
            "FREQ_GROUP_NAME": {
                "_c_order": True,
                "comment": "FREQ_GROUP_NAME column",
                "dataManagerGroup": "StandardStMan",
                "dataManagerType": "StandardStMan",
                "keywords": {},
                "maxlen": 0,
                "option": 0,
                "valueType": "STRING",
            },
            "IF_CONV_CHAIN": {
                "_c_order": True,
                "comment": "IF_CONV_CHAIN column",
                "dataManagerGroup": "StandardStMan",
                "dataManagerType": "StandardStMan",
                "keywords": {},
                "maxlen": 0,
                "option": 0,
                "valueType": "INTEGER",
            },
            "MEAS_FREQ_REF": {
                "_c_order": True,
                "comment": "MEAS_FREQ_REF column",
                "dataManagerGroup": "StandardStMan",
                "dataManagerType": "StandardStMan",
                "keywords": {},
                "maxlen": 0,
                "option": 0,
                "valueType": "INTEGER",
            },
            "NAME": {
                "_c_order": True,
                "comment": "NAME column",
                "dataManagerGroup": "StandardStMan",
                "dataManagerType": "StandardStMan",
                "keywords": {},
                "maxlen": 0,
                "option": 0,
                "valueType": "STRING",
            },
            "NET_SIDEBAND": {
                "_c_order": True,
                "comment": "NET_SIDEBAND column",
                "dataManagerGroup": "StandardStMan",
                "dataManagerType": "StandardStMan",
                "keywords": {},
                "maxlen": 0,
                "option": 0,
                "valueType": "INTEGER",
            },
            "NUM_CHAN": {
                "_c_order": True,
                "comment": "NUM_CHAN column",
                "dataManagerGroup": "StandardStMan",
                "dataManagerType": "StandardStMan",
                "keywords": {},
                "maxlen": 0,
                "option": 0,
                "valueType": "INTEGER",
            },
            "REF_FREQUENCY": {
                "_c_order": True,
                "comment": "REF_FREQUENCY column",
                "dataManagerGroup": "StandardStMan",
                "dataManagerType": "StandardStMan",
                "keywords": {
                    "MEASINFO": {
                        "TabRefCodes": np.array(
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 64], dtype=np.uint32
                        ),
                        "TabRefTypes": [
                            "REST",
                            "LSRK",
                            "LSRD",
                            "BARY",
                            "GEO",
                            "TOPO",
                            "GALACTO",
                            "LGROUP",
                            "CMB",
                            "Undefined",
                        ],
                        "VarRefCol": "MEAS_FREQ_REF",
                        "type": "frequency",
                    },
                    "QuantumUnits": ["Hz"],
                },
                "maxlen": 0,
                "option": 0,
                "valueType": "DOUBLE",
            },
            "TOTAL_BANDWIDTH": {
                "_c_order": True,
                "comment": "TOTAL_BANDWIDTH column",
                "dataManagerGroup": "StandardStMan",
                "dataManagerType": "StandardStMan",
                "keywords": {"QuantumUnits": ["Hz"]},
                "maxlen": 0,
                "option": 0,
                "valueType": "DOUBLE",
            },
        },
    )
