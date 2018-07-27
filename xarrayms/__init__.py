from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = """Simon Perkins"""
__email__ = 'sperkins@ska.ac.za'
__version__ = '0.1.3'


def get_logger():
    import logging
    import logging.handlers

    # Console formatter, mention name
    cfmt = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(cfmt)

    logger = logging.getLogger(__name__)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.propagate = False

    return logger


log = get_logger()

from xarrayms.xarray_ms import (xds_from_table,
                                xds_to_table, xds_from_ms)  # noqa
from xarrayms.table_proxy import TableProxy                 # noqa
