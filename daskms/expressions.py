import ast
import sys

import dask.array as da

operators = {
    ast.Mult: da.multiply,
    ast.Div: da.divide,
    ast.FloorDiv: da.floor_divide,
    ast.Add: da.add,
    ast.Sub: da.subtract,
    ast.USub: da.negative
}


class Visitor(ast.NodeTransformer):
    def __init__(self, datasets):
        assert type(datasets) is list
        self.datasets = datasets

    if sys.version_info[1] >= 8:
        def visit_Constant(self, node):
            return node.n
    else:
        def visit_Num(self, node):
            return node.n

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            raise ValueError("Multiple assignment targets unsupported")

        var_name = node.targets[0].id
        values = self.visit(node.value)

        if type(values) is not list:
            raise TypeError("Expression did not result in a list of values")

        if len(self.datasets) != len(values):
            raise ValueError("len(datasets) != len(values)")

        dim = ("row", "chan", "corr")

        return [ds.assign(**{var_name: (dim, v)})
                for ds, v in zip(self.datasets, values)]

    def visit_UnaryOp(self, node):
        try:
            op = operators[type(node.op)]
        except KeyError:
            raise ValueError(f"Unsupported operator {type(node.op)}")

        value = self.visit(node.operand)

        if type(value) is list:
            return [op(v) for v in value]
        else:
            return op(value)

    def visit_BinOp(self, node):
        try:
            op = operators[type(node.op)]
        except KeyError:
            raise ValueError(f"Unsupported operator {type(node.op)}")

        left = self.visit(node.left)
        right = self.visit(node.right)
        rtype = type(right)
        ltype = type(left)

        if ltype is list and rtype is list:
            assert len(left) == len(right)
            return [op(l, r) for l, r in zip(left, right)]  # noqa: E741
        elif ltype is list:
            return [op(l, right) for l in left]  # noqa: E741
        elif rtype is list:
            return [op(left, r) for r in right]
        else:
            return op(left, right)

    def visit_Name(self, node):
        columns = []

        for i, ds in enumerate(self.datasets):
            xdarray = getattr(ds, node.id)

            try:
                dims = xdarray.dims
            except AttributeError:
                raise TypeError(f"{type(xdarray)} does not look "
                                f"like a valid Dataset Array")
            else:
                if dims != ("row", "chan", "corr"):
                    raise ValueError(f"{xdarray} does not look "
                                     f"like a valid DATA array. "
                                     f"Should have (row, chan, corr) dims"
                                     f"Instead has {dims}")

            columns.append(xdarray.data)

        return columns


def data_column_expr(statement, datasets):
    """
    Produces a list of new datasets with a
    variable set to the result of the
    supplied assignment statement:

    .. code-block:: python

        ds = data_column_expr("FLAG_DATA = DATA / (DIR1_DATA + DIR2_DATA)",
                              datasets)
        flag(ds[0].FLAG_DATA.data)

    Parameters
    ----------
    statement : str
        For example, :code:`EXPR = DATA / (DIR1_DATA + DIR2_DATA + DIR3_DATA)`.
        Can contain data column names as well as numeric literal values.
    datasets : list of Datasets or Dataset
        Datasets containing the DATA columns referenced in the statement
    """
    if isinstance(datasets, (list, tuple)):
        promoted_datasets = list(datasets)
    else:
        promoted_datasets = [datasets]

    mod = ast.parse(statement)
    assert isinstance(mod, ast.Module)

    if len(mod.body) != 1:
        raise ValueError("Single Assignment Statement Only")

    expr = mod.body[0]

    if not isinstance(expr, ast.Assign):
        raise ValueError("Single Assignment Statement Only")

    v = Visitor(promoted_datasets)
    new_datasets = v.visit(expr)

    if not isinstance(promoted_datasets, (tuple, list)):
        return new_datasets[0]

    return new_datasets
