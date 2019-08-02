# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as pjoin

collect_ignore = ["setup.py"]


def pytest_addoption(parser):
    parser.addoption('--stress', action='store_true', dest="stress",
                     default=False, help="Enable stress tests")


def pytest_configure(config):
    if not config.option.stress:
        setattr(config.option, 'markexpr', 'not stress')

    config.addinivalue_line("markers", "stress: long running stress tests")
