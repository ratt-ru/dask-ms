# -*- coding: utf-8 -*-
__pytest_run_marker__ = {"in_pytest": False}


def in_pytest():
    """Return True if we're marked as executing inside pytest"""
    return __pytest_run_marker__["in_pytest"]


def mark_in_pytest(in_pytest=True):
    """Mark if we're in a pytest run"""
    if type(in_pytest) is not bool:
        raise TypeError(f"in_pytest {in_pytest} is not a boolean")

    __pytest_run_marker__["in_pytest"] = in_pytest
