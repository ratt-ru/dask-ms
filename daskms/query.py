# -*- coding: utf-8 -*-


def select_clause(select_cols):
    if select_cols is None or len(select_cols) == 0:
        return "SELECT * "

    return "\n\t".join(("SELECT", ",\n\t".join(select_cols)))


def orderby_clause(index_cols):
    if len(index_cols) == 0:
        return ""

    return "\n\t".join(("ORDERBY", ",\n\t".join(index_cols)))


def groupby_clause(group_cols):
    if len(group_cols) == 0:
        return ""

    return "\n\t".join(("GROUPBY", ",\n\t".join(group_cols)))


def where_clause(group_cols, group_vals):
    if len(group_cols) == 0:
        return ""

    assign_str = [f"{c}={v}" for c, v in zip(group_cols, group_vals)]
    return "\n\t".join(("WHERE", " AND\n\t".join(assign_str)))
