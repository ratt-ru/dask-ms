from pathlib import Path

import fsspec


class DaskMSStore:
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

    def __getitem__(self, key):
        return self.map[key]

    def ls(self, path=None):
        path = path or self.path
        return list(map(Path, self.fs.ls(path, detail=False)))

    def open_file(self, path):
        pass

    def subtable_store(self, subtable):
        if subtable == "MAIN":
            return self

        return DaskMSStore(self.url + "/" + subtable,
                           **self.storage_options)
