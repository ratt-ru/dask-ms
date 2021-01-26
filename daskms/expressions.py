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
    def __init__(self, dataset):
        self.dataset = dataset

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
        value = self.visit(node.value)
        dim = ("row", "chan", "corr")
        return self.dataset.assign(**{var_name: (dim, value)})

    def visit_UnaryOp(self, node):
        try:
            op = operators[type(node.op)]
        except KeyError:
            raise ValueError(f"Unsupported operator {type(node.op)}")

        return op(self.visit(node.operand))

    def visit_BinOp(self, node):
        try:
            op = operators[type(node.op)]
        except KeyError:
            raise ValueError(f"Unsupported operator {type(node.op)}")

        return op(self.visit(node.left), self.visit(node.right))

    def visit_Name(self, node):
        xdarray = getattr(self.dataset, node.id)

        try:
            dims = xdarray.dims
        except AttributeError:
            raise TypeError(f"{type(xdarray)} does not look "
                            f"like a valid Dataset Array")

        if dims != ("row", "chan", "corr"):
            raise ValueError(f"{xdarray} does not look "
                             f"like a valid DATA array. "
                             f"Should have (row, chan, corr) dims"
                             f"Instead has {dims}")

        return xdarray.data


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

    new_datasets = []

    for ds in promoted_datasets:
        new_datasets.append(Visitor(ds).visit(expr))

    if isinstance(datasets, (list, tuple)):
        return new_datasets
    else:
        return new_datasets[0]
