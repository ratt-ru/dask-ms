import atexit
from collections import defaultdict
from contextlib import contextmanager
import logging

import six

try:
    from dask.utils import SerializableLock as Lock
except ImportError:
    from threading import Lock

log = logging.getLogger('xarray-ms')

class OpenCache(object):
    def __init__(self, maxsize=100):
        self.refcount = defaultdict(lambda: 0)
        self.maxsize = 0
        self.cache = {}
        self.lock = Lock()

    def f__setstate__(self, d):
        cache_keys = d.pop('cache')

        self.__dict__.update(d)
        self.cache = {}

        # Reconstruct the cache from key entries
        for k in cache_keys:
            fn, args, kwargset = k

            file = fn(*args, **{ k: v for k, v in kwargset })
            self.cache[k] = file
            self.refcount[k] =1

    def f__getstate__(self):
        return {
            'refcount': self.refcount,
            'maxsize': self.maxsize,
            'cache': self.cache.keys(),
            'lock': self.lock,
        }

    @contextmanager
    def open(self, myopen, *args, **kwargs):
        key = (myopen,) + (args,) + (frozenset(kwargs.items()),)
        with self.lock:
            try:
                file = self.cache[key]
            except KeyError:
                file = myopen(*args, **kwargs)
                self.cache[key] = file

            self.refcount[key] += 1

            if len(self.cache) > self.maxsize:
                pass
                # Clear old files intelligently

        try:
            yield file
        finally:
            with self.lock:
                self.refcount[key] -= 1

    def clear(self):
        with self.lock:
            for key, file in six.iteritems(self.cache):
                try:
                    file.close()
                except AttributeError:
                    log.warn("Unable to call 'close()' on key '%s'" % key)

            self.cache.clear()
            self.refcount.clear()

FILE_CACHE = OpenCache()

def __clear_file_cache():
    global FILE_CACHE
    FILE_CACHE.clear()

atexit.register(__clear_file_cache)