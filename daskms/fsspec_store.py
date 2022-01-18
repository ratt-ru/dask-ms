from pathlib import Path

import fsspec

from daskms.patterns import Multiton


class DaskMSStore(metaclass=Multiton):
    def __init__(self, url, **storage_options):
        self.url = url
        self.map = fsspec.get_mapper(url, **storage_options)
        self.fs = self.map.fs
        self.protocol = fsspec.core.split_protocol(url)[0]
        self.storage_options = storage_options

        path = fsspec.core.strip_protocol(url)
        bits = path.split("::", 1)

        if len(bits) == 1:
            path = bits[0]
            table = "MAIN"
        elif len(bits) == 2:
            path, table = bits

            if table == "MAIN":
                raise ValueError("MAIN is a reserved table name")
        else:
            raise RuntimeError(f"len(bits): {len(bits)} not in (1, 2)")

        self.path = path
        self.table = table

    @staticmethod
    def from_url_storage_options(url, storage_options):
        return DaskMSStore(url, **storage_options)

    def __reduce__(self):
        return (DaskMSStore.from_url_storage_options,
                (self.url, self.storage_options))

    def __getitem__(self, key):
        return self.map[key]

    def ls(self, path=None):
        path = path or self.path
        return list(map(Path, self.fs.ls(path, detail=False)))

    def rglob(self, pattern, **kwargs):
        paths = self.fs.glob(f"{self.path}/**/{pattern}", **kwargs)
        return (p.lstrip(self.path) for p in paths)

    def open_file(self, path, *args, **kwargs):
        return self.fs.open(f"{self.path}/{path}", *args, **kwargs)

    def makedirs(self, path, *args, **kwargs):
        return self.fs.makedirs(f"{self.path}/{path}", *args, **kwargs)

    def subtable_store(self, subtable):
        if subtable == "MAIN":
            return self

        return DaskMSStore(self.url + "/" + subtable,
                           **self.storage_options)
