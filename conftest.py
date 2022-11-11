# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as pjoin

collect_ignore = ["setup.py"]


def pytest_addoption(parser):
    parser.addoption(
        "--stress",
        action="store_true",
        dest="stress",
        default=False,
        help="Enable stress tests",
    )
    parser.addoption(
        "--optional",
        action="store_true",
        dest="optional",
        default=False,
        help="Enable optional tests",
    )
    parser.addoption(
        "--applications",
        action="store_true",
        dest="applications",
        default=False,
        help="Enable application tests",
    )


def pytest_configure(config):
    # Add non-standard markers
    config.addinivalue_line("markers", "stress: long running stress tests")
    config.addinivalue_line("markers", "optional: optional tests")
    config.addinivalue_line("markers", "applications: application tests")

    # Enable/disable them based on parsed config
    disable_str = []

    if not config.option.stress:
        disable_str.append("not stress")

    if not config.option.optional:
        disable_str.append("not optional")

    if not config.option.applications:
        disable_str.append("not applications")

    disable_str = " and ".join(disable_str)

    if disable_str != "":
        print(disable_str)
        setattr(config.option, "markexpr", disable_str)
