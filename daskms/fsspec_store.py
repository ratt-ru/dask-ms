import fsspec


class Store:
    def __init__(self, url, **storage_options):
        self.url = url
        self.map = fsspec.get_mapper(url, **storage_options)
        self.path = self.map.fs._strip_protocol(url)

    def __getitem__(self, key):
        return self.map[key]