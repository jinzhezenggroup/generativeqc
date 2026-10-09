"""Conservative, stdlib-only audit of local NumPy zero materialization.

This is a source-level producer proof, not a sparsity or performance proof. It
assumes ordinary NumPy semantics (no runtime monkey-patching or overloaded index
side effects). Unsupported control flow, calls, aliases, and mutations fail
closed. Cardinalities bound written coordinates, not necessarily nonzero values.
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from generativeqc_compiler.common import materialization

RULE_ID = "python.structured-zero-materialization"
# Ordinary NumPy array-construction APIs; exclude asarray/reshape/transpose,
# which may return views. Conditional execution and positive extent are unknown.
_NUMPY_ARRAY_CREATORS = (
    "empty",
    "zeros",
    "ones",
    "full",
    "empty_like",
    "zeros_like",
    "ones_like",
    "full_like",
    "concatenate",
    "stack",
)


def _text(node: ast.AST) -> str:
    return ast.unparse(node)


def _names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _plain(node: ast.AST) -> bool:
    return not any(
        isinstance(
            item,
            (
                ast.Call,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
                ast.Lambda,
                ast.NamedExpr,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
                ast.Starred,
            ),
        )
        for item in ast.walk(node)
    )


def _binding_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Walk binding statements without inheriting nested definition scopes."""
    yield node
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        # Defaults/decorators execute in the enclosing scope at definition time.
        eager = [*node.args.defaults, *node.args.kw_defaults]
        if not isinstance(node, ast.Lambda):
            eager.extend(node.decorator_list)
            # Annotations are eager on supported older Python versions unless
            # postponed; conservatively reject their potential rebindings.
            eager.append(node.returns)
            eager.extend(
                arg.annotation
                for arg in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                    node.args.vararg,
                    node.args.kwarg,
                )
                if arg is not None
            )
        for expression in eager:
            if expression is not None:
                yield from _binding_nodes(expression)
        return
    if isinstance(node, ast.ClassDef):
        for expression in [*node.bases, *node.keywords, *node.decorator_list]:
            yield from _binding_nodes(expression)
        return
    for child in ast.iter_child_nodes(node):
        yield from _binding_nodes(child)


def _bindings(body: list[ast.stmt], inherited: dict[str, str]) -> dict[str, str]:
    """Resolve imports, rejecting rebinding, including lexical local shadows."""
    result = dict(inherited)
    imported: dict[str, str] = {}
    stores: set[str] = set()
    for stmt in body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name == "numpy":
                    imported[alias.asname or alias.name] = "numpy"
                else:
                    stores.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                if alias.name == "*":
                    return {}
                name = alias.asname or alias.name
                if stmt.module == "numpy" and stmt.level == 0:
                    imported[name] = f"numpy.{alias.name}"
                else:
                    stores.add(name)
        else:
            for item in _binding_nodes(stmt):
                if isinstance(
                    item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    stores.add(item.name)
                # Some lexical bindings store names as strings rather than
                # ast.Name(Store), including exception and pattern captures.
                if (
                    isinstance(item, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar))
                    and item.name
                ):
                    stores.add(item.name)
                if isinstance(item, ast.MatchMapping) and item.rest:
                    stores.add(item.rest)
                if isinstance(item, (ast.Import, ast.ImportFrom)):
                    for alias in item.names:
                        if alias.name == "*":
                            return {}
                        stores.add(
                            alias.asname
                            or (
                                alias.name.split(".")[0]
                                if isinstance(item, ast.Import)
                                else alias.name
                            )
                        )
                if isinstance(item, ast.Name) and isinstance(
                    item.ctx, (ast.Store, ast.Del)
                ):
                    stores.add(item.id)
                if isinstance(item, ast.Attribute) and isinstance(
                    item.ctx, (ast.Store, ast.Del)
                ):
                    stores.update(_names(item.value))
    result.update(imported)
    return {key: value for key, value in result.items() if key not in stores}


def _numpy_call(node: ast.AST, member: str, bindings: dict[str, str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    return (isinstance(fn, ast.Name) and bindings.get(fn.id) == f"numpy.{member}") or (
        isinstance(fn, ast.Attribute)
        and fn.attr == member
        and isinstance(fn.value, ast.Name)
        and bindings.get(fn.value.id) == "numpy"
    )


def _product(values: list[str]) -> str:
    return " * ".join(f"({value})" for value in values) or "1"


def _full_slice(node: ast.AST, dimension: ast.AST) -> bool:
    return (
        isinstance(node, ast.Slice)
        and (node.lower is None or _text(node.lower) == "0")
        and (node.upper is None or _text(node.upper) == _text(dimension))
        and (node.step is None or _text(node.step) == "1")
    )


def _support(
    index: ast.AST,
    shape: list[ast.AST],
    bindings: dict[str, str],
    loops: dict[str, ast.Call],
) -> dict[str, Any] | None:
    dims = [_text(dim) for dim in shape]
    if _numpy_call(index, "ix_", bindings):
        if _names(index) & loops.keys():
            return None
        if (
            index.keywords
            or len(index.args) != len(shape)
            or not all(map(_plain, index.args))
        ):
            return None
        axes = [_text(arg) for arg in index.args]
        return {
            "kind": "cartesian-block",
            "support": _text(index),
            "upper_bound": _product([f"len({a})" for a in axes]),
            "full": False,
        }
    for member, kind in (
        ("diag_indices", "diagonal"),
        ("triu_indices", "upper-triangle"),
        ("tril_indices", "lower-triangle"),
    ):
        if not _numpy_call(index, member, bindings):
            continue
        # Only square, main-diagonal forms; offsets and rectangular triangles
        # need additional domain reasoning and are deliberately unsupported.
        if (
            len(shape) != 2
            or dims[0] != dims[1]
            or len(index.args) != 1
            or index.keywords
        ):
            return None
        if not _plain(index.args[0]) or _text(index.args[0]) != dims[0]:
            return None
        n = dims[0]
        count = n if kind == "diagonal" else f"({n}) * (({n}) + 1) // 2"
        return {
            "kind": kind,
            "support": _text(index),
            "upper_bound": count,
            "full": False,
        }
    axes = list(index.elts) if isinstance(index, ast.Tuple) else [index]
    if len(axes) != len(shape) or not all(map(_plain, axes)):
        return None
    terms, seen, restricted = [], set(), False
    for axis, dimension in zip(axes, shape):
        expression = _text(axis)
        if isinstance(axis, ast.Slice):
            if _names(axis) & loops.keys():
                return None
            full = _full_slice(axis, dimension)
            restricted |= not full
            terms.append(
                _text(dimension)
                if full
                else f"len(range({_text(dimension)})[{expression}])"
            )
        elif isinstance(axis, ast.Constant) and type(axis.value) is int:
            restricted = True
        elif isinstance(axis, ast.Name) and axis.id in loops:
            if expression in seen:
                restricted = True
                continue
            seen.add(expression)
            domain = loops[axis.id]
            terms.append(f"len({_text(domain)})")
            restricted |= _text(domain) not in {
                f"range({_text(dimension)})",
                f"range(0, {_text(dimension)})",
            }
        else:
            # A repeated unknown name or attribute is not necessarily an
            # advanced integer index. Slices, None and scalar booleans can
            # select the entire tensor even in out[index, index]. Without an
            # index-type proof, only the range-bound names above are admitted.
            return None
    diagonal = len(set(map(_text, axes))) == 1 and not isinstance(axes[0], ast.Slice)
    return {
        "kind": "diagonal" if diagonal else "slice-block",
        "support": _text(index),
        "upper_bound": _product(terms),
        "full": not restricted,
    }


def _returned(node: ast.AST | None, name: str) -> bool:
    if isinstance(node, ast.Name):
        return node.id == name
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _returned(node.operand, name)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        for scalar, array in ((node.left, node.right), (node.right, node.left)):
            if (
                isinstance(scalar, ast.Constant)
                and type(scalar.value) in (int, float)
                and (type(scalar.value) is int or math.isfinite(scalar.value))
                and _returned(array, name)
            ):
                return True
    return False


def _full_union(
    indices: list[ast.AST], shape: list[ast.AST], bindings: dict[str, str]
) -> bool:
    """Recognize complementary triangles or two complementary slab writes."""
    if any(_numpy_call(index, "triu_indices", bindings) for index in indices) and any(
        _numpy_call(index, "tril_indices", bindings) for index in indices
    ):
        return True
    for first in indices:
        for second in indices:
            if not isinstance(first, ast.Tuple) or not isinstance(second, ast.Tuple):
                continue
            differing = []
            for axis, (left, right, dim) in enumerate(
                zip(first.elts, second.elts, shape)
            ):
                if not (_full_slice(left, dim) and _full_slice(right, dim)):
                    differing.append(axis)
            if len(differing) != 1:
                continue
            axis = differing[0]
            left, right = first.elts[axis], second.elts[axis]
            if (
                isinstance(left, ast.Slice)
                and isinstance(right, ast.Slice)
                and left.lower is None
                and right.upper is None
                and left.step is None
                and right.step is None
                and left.upper is not None
                and right.lower is not None
                and _text(left.upper) == _text(right.lower)
            ):
                return True
    return False


def _candidate(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    position: int,
    bindings: dict[str, str],
    path: str,
    scope: str,
) -> dict[str, Any] | None:
    stmt = function.body[position]
    if (
        not isinstance(stmt, ast.Assign)
        or len(stmt.targets) != 1
        or not isinstance(stmt.targets[0], ast.Name)
    ):
        return None
    name, allocation = stmt.targets[0].id, stmt.value
    if not _numpy_call(allocation, "zeros", bindings):
        return None
    if not allocation.args or len(allocation.args) > 2:
        return None
    if any(
        kw.arg not in {"dtype", "order"} or not _plain(kw.value)
        for kw in allocation.keywords
    ):
        return None
    shape_node = allocation.args[0]
    if not isinstance(shape_node, (ast.Tuple, ast.List)) or len(shape_node.elts) < 2:
        return None
    shape = list(shape_node.elts)
    if not all(map(_plain, allocation.args)):
        return None
    # A local import later in the function cannot supply this allocation.
    for later in function.body[position + 1 :]:
        if isinstance(later, (ast.Import, ast.ImportFrom)) and {
            alias.asname or alias.name for alias in later.names
        } & _names(allocation.func):
            return None
    if any(
        isinstance(item, (ast.Global, ast.Nonlocal)) and name in item.names
        for item in ast.walk(function)
    ):
        return None
    # A closure could retain an alias to the local cell without mentioning it in
    # the straight-line producer region.
    if any(
        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        and item is not function
        and name in _names(item)
        for item in ast.walk(function)
    ):
        return None
    writes: list[dict[str, Any]] = []
    indices: list[ast.AST] = []
    dependencies, rebound, loop_names = _names(shape_node), set(), set()

    def visit(body: list[ast.stmt], loops: dict[str, ast.Call]) -> bool:
        for item in body:
            if isinstance(item, ast.For):
                call = item.iter
                if (
                    not isinstance(item.target, ast.Name)
                    or item.target.id == name
                    or item.target.id in _names(shape_node)
                    or item.target.id in loops
                    or item.orelse
                    or not isinstance(call, ast.Call)
                    or not isinstance(call.func, ast.Name)
                    or bindings.get(call.func.id) != "builtins.range"
                    or call.keywords
                    or not 1 <= len(call.args) <= 3
                    or _names(call) & loops.keys()
                    or not all(map(_plain, call.args))
                ):
                    return False
                loop_names.add(item.target.id)
                dependencies.update(_names(call) - {"range"})
                if not visit(item.body, {**loops, item.target.id: call}):
                    return False
            elif isinstance(item, ast.Assign) and len(item.targets) == 1:
                target = item.targets[0]
                if not _plain(item.value) or name in _names(item.value):
                    return False
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == name
                ):
                    if name in _names(target.slice):
                        return False
                    support = _support(target.slice, shape, bindings, loops)
                    if support is None or support["full"]:
                        return False
                    dependencies.update(_names(target.slice) - loops.keys())
                    support.update(
                        line=item.lineno,
                        index=_text(target.slice),
                        loops={key: _text(value) for key, value in loops.items()},
                    )
                    writes.append(support)
                    indices.append(target.slice)
                elif isinstance(target, ast.Name) and target.id != name:
                    rebound.add(target.id)
                else:
                    return False
            elif not isinstance(item, ast.Pass):
                return False
        return True

    returned = None
    for item in function.body[position + 1 :]:
        if isinstance(item, ast.Return):
            if _returned(item.value, name):
                returned = item
            break
        if not visit([item], {}):
            return None
    if (
        returned is None
        or not writes
        or dependencies & (rebound | loop_names)
        or loop_names & rebound
        or _full_union(indices, shape, bindings)
    ):
        return None
    dense = _product([_text(dim) for dim in shape])
    upper = (
        f"min({dense}, "
        + " + ".join(f"({write['upper_bound']})" for write in writes)
        + ")"
    )
    triangle = any("triangle" in write["kind"] for write in writes)
    evidence = [
        f"line {stmt.lineno}: {_text(stmt)}",
        "Nonzero support is contained in the union of these producer writes:",
    ]
    evidence.extend(
        f"line {write['line']}: {name}[{write['index']}]"
        + (f" with {write['loops']}" if write["loops"] else "")
        for write in writes
    )
    evidence += [
        f"Written-coordinate upper bound: {upper}; allocated elements: {dense}",
        f"line {returned.lineno}: {_text(returned)}; downstream dense consumer unresolved",
        (
            "No escaping aliases or unsupported mutations/calls in the analyzed producer region; "
            "index-set size, overlap, and strict storage reduction are not inferred."
        ),
    ]
    if triangle:
        evidence.append(
            "Triangular support retains quadratic growth; it is not a lower-order sparse domain."
        )
    diagnostic = materialization.materialization_diagnostic(
        origin="python-source",
        subject={
            "path": str(path),
            "line": stmt.lineno,
            "function": scope,
            "buffer": name,
        },
        dense_elements=dense,
        support_kind="union-upper-bound",
        written_elements=upper,
        domains=writes,
        certificate_scope="local producer-return under ordinary NumPy semantics; written-coordinate upper bound, not exact cardinality or numerical nonzeros",
        layout="dense-array",
    )
    return {
        "rule_id": RULE_ID,
        "path": str(path),
        "line": stmt.lineno,
        "function": scope,
        "evidence": evidence,
        "confidence": "high",
        "disposition": "review-required",
        "action": "Review producer support cardinality and downstream layout requirements before changing storage.",
        "details": {
            "materialization_diagnostic": diagnostic,
            "allocation": _text(allocation),
            "allocated_elements": dense,
            "write_support": writes,
            "support_interpretation": "NumPy indexing coordinates at each write line",
            "support_upper_bound": upper,
            "return_line": returned.lineno,
            "consumer": "unresolved",
            "strict_reduction_proven": False,
            "growth": "quadratic-triangle" if triangle else "symbolic",
        },
    }


def _functions(
    body: list[ast.stmt], prefix: str, bindings: dict[str, str]
) -> Iterator[tuple[ast.AST, str, dict[str, str]]]:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = f"{prefix}.{node.name}" if prefix else node.name
            local = _bindings(node.body, bindings)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parameters = {
                    arg.arg
                    for arg in (
                        *node.args.posonlyargs,
                        *node.args.args,
                        *node.args.kwonlyargs,
                    )
                }
                parameters.update(
                    arg.arg
                    for arg in (node.args.vararg, node.args.kwarg)
                    if arg is not None
                )
                local = {
                    key: value for key, value in local.items() if key not in parameters
                }
                yield node, scope, local
            # Class namespaces do not provide lexical bindings to methods.
            yield from _functions(
                node.body, scope, bindings if isinstance(node, ast.ClassDef) else local
            )


def _loop_allocations(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    bindings: dict[str, str],
    path: str,
    scope: str,
) -> list[dict[str, Any]]:
    """Inventory NumPy array-creating calls in lexical loops, not runtime events.

    Only ordinary NumPy binding semantics are assumed. Function-local imports
    must dominate the call as top-level statements; conditional/late imports
    do not establish a proven binding. A loop is not necessarily prepared replay.
    """
    local_imports: dict[str, list[tuple[int, bool]]] = {}

    class BindingVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            pass

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            pass

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            pass

        def visit_Lambda(self, node: ast.Lambda) -> None:
            pass

        def visit_Import(self, node: ast.Import | ast.ImportFrom) -> None:
            for alias in node.names:
                bound = alias.asname or (
                    alias.name.split(".")[0]
                    if isinstance(node, ast.Import)
                    else alias.name
                )
                local_imports.setdefault(bound, []).append(
                    (node.lineno, node in function.body)
                )

        visit_ImportFrom = visit_Import

    binding_visitor = BindingVisitor()
    for statement in function.body:
        binding_visitor.visit(statement)

    findings: list[dict[str, Any]] = []

    class LoopVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.loops: list[ast.For | ast.AsyncFor | ast.While] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # Definition inside a loop does not execute the function body.
            pass

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            pass

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            pass

        def visit_Lambda(self, node: ast.Lambda) -> None:
            pass

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
            # Constructing a generator evaluates only its outermost iterable.
            # Its body, filters and remaining iterables are deferred until use.
            self.visit(node.generators[0].iter)

        def _visit_body(
            self,
            node: ast.For | ast.AsyncFor | ast.While,
            body: list[ast.stmt],
            orelse: list[ast.stmt],
        ) -> None:
            self.loops.append(node)
            for statement in body:
                self.visit(statement)
            self.loops.pop()
            # The loop's 'else' runs after the loop, not on each iteration.
            # An enclosing loop still makes it a repeated candidate.
            for statement in orelse:
                self.visit(statement)

        def visit_For(self, node: ast.For) -> None:
            # The iterable is evaluated once on entry, not once per own cycle.
            self.visit(node.iter)
            self._visit_body(node, node.body, node.orelse)

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            self.visit(node.iter)
            self._visit_body(node, node.body, node.orelse)

        def visit_While(self, node: ast.While) -> None:
            # The while condition is re-evaluated at each iteration.
            self.loops.append(node)
            self.visit(node.test)
            for statement in node.body:
                self.visit(statement)
            self.loops.pop()
            for statement in node.orelse:
                self.visit(statement)

        def visit_Call(self, node: ast.Call) -> None:
            if self.loops:
                member = next(
                    (
                        name
                        for name in _NUMPY_ARRAY_CREATORS
                        if _numpy_call(node, name, bindings)
                    ),
                    None,
                )
                if member in {"concatenate", "stack"} and (
                    any(isinstance(arg, ast.Starred) for arg in node.args)
                    or (
                        len(node.args) >= 3
                        and not (
                            isinstance(node.args[2], ast.Constant)
                            and node.args[2].value is None
                        )
                    )
                    or any(
                        keyword.arg is None
                        or (
                            keyword.arg == "out"
                            and not (
                                isinstance(keyword.value, ast.Constant)
                                and keyword.value.value is None
                            )
                        )
                        for keyword in node.keywords
                    )
                ):
                    # Known-out variants can reuse caller-owned backing storage;
                    # **kwargs may also supply out, so avoid asserting allocation.
                    member = None
                name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.value.id
                    if isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    else None
                )
                if (
                    member is not None
                    and name is not None
                    and all(
                        line < node.lineno and at_function_level
                        for line, at_function_level in local_imports.get(name, ())
                    )
                ):
                    headers = [
                        f"for {ast.unparse(loop.target)} in {ast.unparse(loop.iter)}"
                        if isinstance(loop, (ast.For, ast.AsyncFor))
                        else f"while {ast.unparse(loop.test)}"
                        for loop in self.loops
                    ]
                    loop_targets = " ".join(
                        ast.unparse(loop.target).lower()
                        for loop in self.loops
                        if isinstance(loop, (ast.For, ast.AsyncFor))
                    )
                    if any(
                        key in loop_targets
                        for key in ("tile", "panel", "chunk", "block", "batch")
                    ):
                        role = "per-tile-candidate"
                    elif any(
                        key in loop_targets for key in ("iter", "step", "epoch", "scf")
                    ):
                        role = "per-iteration-candidate"
                    else:
                        role = "unknown-loop"
                    findings.append(
                        {
                            "rule_id": "python.loop-host-allocation",
                            "path": str(path),
                            "line": node.lineno,
                            "function": scope,
                            "column": node.col_offset + 1,
                            "evidence": [
                                f"line {node.lineno}: {ast.unparse(node)}",
                                "NumPy array creation is lexically in a loop; execution, positive size, and backing bytes are not measured.",
                            ],
                            "confidence": "structural-site",
                            "disposition": "needs-role-and-runtime-review",
                            "action": "Check prepared replay reachability, array shape and reusable ownership before changing this site.",
                            "details": {
                                "numpy_operation": member,
                                "loop_context": headers,
                                "phase_hint": role,
                                "count_kind": "static-site-not-runtime-count",
                                "requested_bytes": None,
                            },
                        }
                    )
            self.generic_visit(node)

    visitor = LoopVisitor()
    for statement in function.body:
        visitor.visit(statement)
    return findings


def audit_python(text: str, path: str) -> list[dict[str, Any]]:
    """Return producer-only findings; syntax errors and unsupported cases skip."""
    try:
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, ValueError):
        return []
    findings = []
    for function, scope, bindings in _functions(
        tree.body, "", _bindings(tree.body, {"range": "builtins.range"})
    ):
        for position in range(len(function.body)):
            finding = _candidate(function, position, bindings, path, scope)
            if finding is not None:
                findings.append(finding)
        findings.extend(_loop_allocations(function, bindings, path, scope))
    return findings
