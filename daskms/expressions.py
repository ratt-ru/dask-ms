import ast

import dask.array as da

operators = {
    ast.Mult: da.multiply,
    ast.Div: da.divide,
    ast.Add: da.add,
    ast.Sub: da.subtract,
    ast.USub: da.negative
}


class Visitor(ast.NodeTransformer):
    def __init__(self, datasets):
        self.datasets = datasets

    def visit_Constant(self, node):
        pass

    def visit_Num(self, node):
        return node.n

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


def data_column_expr(expression, datasets):
    """
    Produces a list of new datasets with a
    ``DASK_EXPRESSION`` variable set to the result of the
    supplied expression:

    .. code-block:: python

        datasets = data_column_expr("DATA / (DIR1_DATA + DIR2_DATA)", datasets)

    Parameters
    ----------
    expression : str
        :code:`DATA / (DIR1_DATA + DIR2_DATA + DIR3_DATA)`.
        Can contain data column names as well as numeric literal values.
    datasets : list of Datasets or Dataset
        Datasets containing the DATA columns referenced in the expression
    """
    if isinstance(datasets, (list, tuple)):
        promoted_datasets = list(datasets)
    else:
        promoted_datasets = [datasets]

    mod = ast.parse(expression)
    assert isinstance(mod, ast.Module)

    if len(mod.body) != 1:
        raise ValueError("Single Expression Only")

    expr = mod.body[0]

    if not isinstance(expr, ast.Expr):
        raise ValueError("Single Expression Only")

    v = Visitor(promoted_datasets)
    node = v.visit(expr)

    new_datasets = []

    for i, ds in enumerate(datasets):
        nds = ds.assign(DATA_EXPRESSION=(("row", "chan", "corr"),
                                         node.value[i]))
        new_datasets.append(nds)

    if not isinstance(promoted_datasets, (tuple, list)):
        return new_datasets[0]

    return new_datasets
