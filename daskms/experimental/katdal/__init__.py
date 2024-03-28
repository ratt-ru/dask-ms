try:
    import katdal  # noqa
except ImportError as e:
    raise ImportError("pip install dask-ms[katdal] for katdal support")


from daskms.experimental.katdal.katdal_import import katdal_import
