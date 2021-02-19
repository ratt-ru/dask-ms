import json
import numpy as np

from daskms.reads import read_datasets
from daskms.dataset_schema import DatasetSchema


def test_dataset_schema(ms):
    datasets = read_datasets(ms, [], [], [])
    assert len(datasets) == 1
    ds = datasets[0]

    row, chan, corr = (ds.dims[d] for d in ("row", "chan", "corr"))
    cdata = np.random.random((row, chan, corr)).astype(np.complex64)

    ds = ds.assign(**{"CORRECTED_DATA": (("row", "chan", "corr"), cdata)})

    ds = ds.assign_coords(**{
        "row": ("row", np.arange(row)),
        "chan": ("chan", np.arange(chan)),
        "corr": ("corr", np.arange(corr)),
    })

    # We can shift between objects and dict representation
    ds_schema = DatasetSchema.from_dataset(ds)
    assert DatasetSchema.from_dict(ds_schema.to_dict()) == ds_schema

    # And the dict repr can go through JSON, although
    # we don't compare because JSON converts tuples to lists
    serialized = json.dumps(ds_schema.to_dict()).encode()
    DatasetSchema.from_dict(json.loads(serialized.decode()))
