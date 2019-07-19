# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def select_clause(select_cols):
    if select_cols is None or len(select_cols) == 0:
        return "SELECT * "

    return " ".join(("SELECT", ", ".join(select_cols)))


def orderby_clause(index_cols):
    if len(index_cols) == 0:
        return ""

    return " ".join(("ORDERBY", ", ".join(index_cols)))


def groupby_clause(group_cols):
    if len(group_cols) == 0:
        return ""

    return " ".join(("GROUPBY", ", ".join(group_cols)))


def where_clause(group_cols, group_vals):
    if len(group_cols) == 0:
        return ""

    assign_str = ["%s=%s" % (c, v) for c, v in zip(group_cols, group_vals)]
    return " ".join(("WHERE", " AND ".join(assign_str)))
