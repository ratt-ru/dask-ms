import base64
import os
from pathlib import Path
from threading import Lock
import weakref

from daskms.dataset import Dataset
from daskms.experimental.arrow.schema import dict_dataset_schema, dataset_schema
from daskms.utils import arg_hasher
from fasteners.process_lock import InterProcessLock

try:
    import pyarrow as pa
except ImportError as e:
    pyarrow_import_error = e
else:
    pyarrow_import_error = None

_DATASET_TYPES = (Dataset,)
_DATASET_TYPE = Dataset

try:
    import xarray as xr
except ImportError as e:
    xarray_import_error = e
else:
    xarray_import_error = None
    _DATASET_TYPES += (xr.Dataset,)
    _DATASET_TYPE = xr.Dataset


_dataset_cache = weakref.WeakValueDictionary()
_dataset_lock = Lock()


class WriteDatasetMetaClass(type):
    """
    https://en.wikipedia.org/wiki/Multiton_pattern

    """
    def __call__(cls, *args, **kwargs):
        key = arg_hasher((cls,) + args + (kwargs,))

        try:
            return _dataset_cache[key]
        except KeyError:
            with _dataset_lock:
                try:
                    return _dataset_cache[key]
                except KeyError:
                    instance = type.__call__(cls, *args, **kwargs)
                    _dataset_cache[key] = instance
                    return instance


class WriteDataset(metaclass=WriteDatasetMetaClass):
    def __init__(self, path, schema):
        self.path = Path(path)
        self.schema = schema
        self.lock = Lock()

    def __reduce__(self):
        return (WriteDataset, (self.path, self.schema))

    @property
    def dataset(self):
        if hasattr(self, "_dataset"):
            return self._dataset

        with self.lock:
            key = b"%d" % arg_hasher((self.path, self.schema))
            name = base64.urlsafe_b64encode(key).rstrip(b'=').decode('ascii')
            lock = InterProcessLock(self.path.parents[0] / f".{name}.lock")
            acquired = lock.acquire(blocking=False)

            try:
                if acquired:
                    os.makedirs(self.path, exist_ok=False)
            finally:
                if acquired:
                    lock.release()

            self._dataset = True


def xds_to_parquet(xds, path):
    if not isinstance(path, Path):
        path = Path(path)

    if isinstance(xds, _DATASET_TYPES):
        xds = [xds]
    elif isinstance(xds, (tuple, list)):
        if not all(isinstance(ds, _DATASET_TYPES) for ds in xds):
            raise TypeError("xds must be a Dataset or list of Datasets")
    else:
        raise TypeError("xds must be a Dataset or list of Datasets")

    schema = dict_dataset_schema(xds)

    dataset = WriteDataset(path, schema)
    dataset.dataset