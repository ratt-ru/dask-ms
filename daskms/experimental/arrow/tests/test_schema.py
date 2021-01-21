import pickle

from daskms import xds_from_ms
from daskms.experimental.arrow.schema import (dict_dataset_schema,
                                              dataset_schema)


def test_schema_creation(ms):
    schema = dict_dataset_schema(xds_from_ms(ms))
    assert pickle.loads(pickle.dumps(schema)) == schema
    dataset_schema(schema)
