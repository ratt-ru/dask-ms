# -*- coding: utf-8 -*-
__pytest_run_marker__ = {'in_pytest': False}


# Tag indicating that missing packages should generate an
# exception, regardless of the 'in_pytest' marker
# Used for testing exception raising behaviour
force_missing_pkg_exception = object()


def in_pytest():
    """ Return True if we're marked as executing inside pytest """
    return __pytest_run_marker__['in_pytest']


def mark_in_pytest(in_pytest=True):
    """ Mark if we're in a pytest run """
    if type(in_pytest) is not bool:
        raise TypeError('in_pytest %s is not a boolean' % in_pytest)

    __pytest_run_marker__['in_pytest'] = in_pytest
