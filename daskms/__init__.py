# -*- coding: utf-8 -*-

import logging

__author__ = """Simon Perkins"""
__email__ = "sperkins@ska.ac.za"
__version__ = "0.2.8"

logging.getLogger(__name__).addHandler(logging.NullHandler())

from daskms.dask_ms import (xds_from_storage_ms, xds_from_table,      # noqa
                                xds_to_table,    # noqa
                                xds_from_ms,     # noqa
                                xds_from_storage_ms,     # noqa
                                xds_from_storage_table)  # noqa

from daskms.dataset import Dataset, Variable     # noqa
from daskms.table_proxy import TableProxy        # noqa
