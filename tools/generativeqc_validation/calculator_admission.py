"""Locate the constructor-reachable public force admission for offline audits.

This is source discovery, not a substitute for executing admission/force gates.
Moving admission into a constructor helper must neither lose it nor qualify an
unreachable lookalike. Helper names and the number of delegation levels are not
part of the public contract.
"""

import ast


def force_admission_owner(source: str) -> ast.FunctionDef:
    """Find the unique force-admission owner reachable from Calculator.__init__."""
    classes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "Calculator"
    ]
    if len(classes) != 1:
        raise RuntimeError("public Calculator owner is missing or ambiguous")
    methods: dict[str, ast.FunctionDef] = {}
    for node in classes[0].body:
        if isinstance(node, ast.FunctionDef):
            if node.name in methods:
                raise RuntimeError("public Calculator method is ambiguous")
            methods[node.name] = node
    if "__init__" not in methods:
        raise RuntimeError("public Calculator constructor is missing or ambiguous")
    pending = ["__init__"]
    visited: set[str] = set()
    owners = []
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        method = methods[name]
        # Nested callbacks are not constructor execution merely because their
        # bodies mention self; only follow calls in this method's own scope.
        nodes = []
        work = list(method.body)
        while work:
            node = work.pop()
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ):
                continue
            nodes.append(node)
            work.extend(ast.iter_child_nodes(node))
        if any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "semilocal_force"
                for target in node.targets
            )
            for node in nodes
        ):
            owners.append(method)
        for node in nodes:
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and method.args.args
                and node.func.value.id == method.args.args[0].arg
                and node.func.attr in methods
            ):
                pending.append(node.func.attr)
    if len(owners) != 1:
        raise RuntimeError("public CUDA force capability owner is ambiguous")
    return owners[0]
