import logging
import logging.handlers

def get_logger():
    # Console formatter, mention name
    cfmt = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(cfmt)

    logger = logging.getLogger('xarray-ms')
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.propagate = False

    return logger

log = get_logger()

from xarray_ms import xds_from_table, xds_to_table

def clear_file_cache():
    from .file_cache import FILE_CACHE
    FILE_CACHE.clear()
