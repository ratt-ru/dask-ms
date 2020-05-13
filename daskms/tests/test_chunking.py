# -*- coding: utf-8 -*-

import pytest

from daskms.chunking import MSChunking
from daskms.table_proxy import TableProxy

def test_chunking():
    chunks = MSChunking("/home/sperkins/data/AF0236_spw01.ms/")

    chunks(1, 2)