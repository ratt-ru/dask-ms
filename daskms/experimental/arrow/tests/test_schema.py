from daskms import xds_from_ms
from daskms.experimental.arrow.schema import dataset_schema


def test_schema_creation(ms):
    schema = dataset_schema(xds_from_ms(ms))
