try:
    import katdal
except ImportError as e:
    raise ImportError("katdal is not installed\n" "pip install dask-ms[katdal]") from e
