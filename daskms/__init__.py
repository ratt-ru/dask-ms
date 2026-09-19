# -*- coding: utf-8 -*-

# Activate the casacure backend (DASK_MS_BACKEND=casacure) before any
# module pulls in casacore.tables via lazy_import.
from daskms.casacure_backend import activate_casacure_backend

import logging

activate_casacure_backend()

__author__ = """Simon Perkins"""
__email__ = "sperkins@ska.ac.za"
__version__ = "0.2.32"

logging.getLogger(__name__).addHandler(logging.NullHandler())

from daskms.dask_ms import (
    xds_from_table,  # noqa
    xds_from_ms,  # noqa
    xds_from_storage_ms,  # noqa
    xds_from_storage_table,  # noqa
    xds_to_table,  # noqa
    xds_to_storage_table,
)  # noqa

from daskms.dataset import Dataset, Variable  # noqa
from daskms.table_proxy import TableProxy  # noqa
