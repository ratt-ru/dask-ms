# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

__author__ = """Simon Perkins"""
__email__ = 'sperkins@ska.ac.za'
__version__ = '0.2.0-alpha2'

logging.getLogger(__name__).addHandler(logging.NullHandler())

from daskms.dask_ms import (xds_from_table,  # noqa
                                xds_to_table,    # noqa
                                xds_from_ms)     # noqa

from daskms.table_proxy import TableProxy      # noqa
