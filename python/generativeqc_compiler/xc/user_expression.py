"""Bounded declaration adapter into the canonical scalar Graph; no execution admission."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.integral.expr import Expr, Graph

SCHEMA = "generativeqc.xc-expression.v2"
FEATURES = ("rho_a", "rho_b")
MAX_SOURCE = 4096
MAX_AST = 256
MAX_DEPTH = 32
MAX_GRAPH = 512
MAX_DEGREE = 8


def _coefficient(value: Fraction) -> Fraction:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 256:
        raise ValueError("coefficient exceeds 256-bit budget")
    if not math.isfinite(float(value)):
        raise ValueError("nonfinite coefficient")
    return value


def _capture(source: str) -> tuple[Graph, Expr]:
    if type(source) is not str:
        raise TypeError(
            "expression requires a string declaration, never a callable or Graph"
        )
    if not source.strip() or len(source) > MAX_SOURCE:
        raise ValueError("expression source must contain 1..4096 characters")
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("invalid bounded scalar declaration") from exc
    pending: list[tuple[ast.AST, int]] = [(tree, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > MAX_AST or depth > MAX_DEPTH:
            raise ValueError("expression exceeds AST node/depth budget")
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    graph = Graph()
    checked_nodes = 0

    def check_constants() -> None:
        """Bound generated coefficients even when syntax was input-dependent."""
        nonlocal checked_nodes
        for node in graph.nodes[checked_nodes:]:
            if node.operation == "constant":
                if not isinstance(node.payload, Fraction):
                    raise ValueError("unsupported scalar coefficient payload")
                _coefficient(node.payload)
        checked_nodes = len(graph.nodes)

    def product(left: Expr, right: Expr) -> Expr:
        # Preserve ordinary binary construction and its canonical identity. Only
        # bypass its legacy FP64 unity shortcut for an exact non-unity value.
        for operand in (left, right):
            node = graph.node(operand)
            if (
                node.operation == "constant"
                and isinstance(node.payload, Fraction)
                and node.payload != 1
                and float(node.payload) == 1.0
            ):
                return graph.multiply_many((left, right))
        return left * right

    def visit(node: ast.AST) -> tuple[Expr, int, Fraction | None]:
        constant = None
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if type(node.value) is float and not math.isfinite(node.value):
                raise ValueError("nonfinite literal")
            constant = _coefficient(Fraction(str(node.value)))
            result, degree = graph.constant(constant), 0
        elif isinstance(node, ast.Name) and node.id in FEATURES:
            result, degree = graph.variable(node.id), 1
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            result, degree, constant = visit(node.operand)
            if isinstance(node.op, ast.USub):
                result = product(graph.constant(-1), result)
                constant = -constant if constant is not None else None
        elif isinstance(node, ast.BinOp):
            left, ld, lc = visit(node.left)
            right, rd, rc = visit(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                subtract = isinstance(node.op, ast.Sub)
                if subtract:
                    right = product(graph.constant(-1), right)
                result = left + right
                degree = max(ld, rd)
                if lc is not None and rc is not None:
                    constant = lc - rc if subtract else lc + rc
            elif isinstance(node.op, ast.Mult):
                result, degree = product(left, right), ld + rd
                if lc is not None and rc is not None:
                    constant = lc * rc
            elif isinstance(node.op, ast.Div) and rc is not None and rc != 0:
                result, degree = product(left, graph.constant(_coefficient(1 / rc))), ld
                if lc is not None:
                    constant = lc / rc
            elif (
                isinstance(node.op, ast.Pow)
                and rc is not None
                and rc.denominator == 1
                and 0 <= rc <= 4
            ):
                result, degree = left.pow(float(rc)), ld * int(rc)
                if lc is not None:
                    constant = lc ** int(rc)
            else:
                raise ValueError(
                    "unsupported operation: divide only by nonzero constants; powers 0..4"
                )
        else:
            raise ValueError(
                "unsupported syntax or ingredient; only pure rho_a/rho_b polynomials"
            )
        if degree > MAX_DEGREE or len(graph.nodes) > MAX_GRAPH:
            raise ValueError("expression exceeds degree/graph budget")
        if constant is not None:
            _coefficient(constant)
        check_constants()
        return result, degree, constant

    root, _, _ = visit(tree.body)
    graph, (root,) = graph.canonicalize_associative((root,))
    if len(graph.nodes) > MAX_GRAPH:
        raise ValueError("expression exceeds canonical graph budget")
    checked_nodes = 0
    check_constants()
    return graph, root


def _payload(graph: Graph, root: Expr) -> dict:
    graph.node(root)
    keys: dict[int, tuple] = {}

    def structural_key(identifier: int) -> tuple:
        if identifier not in keys:
            node = graph.nodes[identifier]
            children = tuple(structural_key(i) for i in node.arguments)
            if node.operation in ("add", "multiply"):
                children = tuple(sorted(children))
            keys[identifier] = node.operation, str(node.payload), children
        return keys[identifier]

    nodes: list[list] = []
    indices: dict[int, int] = {}

    def visit(identifier: int) -> int:
        if identifier not in indices:
            node = graph.nodes[identifier]
            arguments = node.arguments
            if node.operation in ("add", "multiply"):
                arguments = tuple(sorted(arguments, key=structural_key))
            references = [visit(i) for i in arguments]
            indices[identifier] = len(nodes)
            nodes.append([node.operation, references, str(node.payload)])
        return indices[identifier]

    root_index = visit(root.identifier)
    return {
        "schema": SCHEMA,
        "features": list(FEATURES),
        "spin": "collinear-spin-resolved",
        "energy": "per-volume",
        "units": {"rho": "bohr^-3", "energy": "hartree/bohr^3"},
        "dtype": "float64",
        "domain": "finite nonnegative rho; finite pointwise result; no clipping",
        "language": "rho-polynomial-degree-8; constant-division; powers-0..4",
        "nodes": nodes,
        "root": root_index,
    }


@dataclass(frozen=True, slots=True)
class UserExpression:
    """Sealed scalar declaration, distinct from scientifically admitted FunctionalSpec.

    The descriptive name never contributes built-in qualification. Graph exports
    are fresh mutable copies; no caller-owned Graph or Python callback is retained.
    """

    source: str
    name: str = "user-expression"
    _canonical: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate all declarations before sealing their canonical graph."""
        if type(self.name) is not str or not self.name.strip() or len(self.name) > 256:
            raise ValueError("name requires 1..256 characters")
        graph, root = _capture(self.source)
        object.__setattr__(
            self,
            "_canonical",
            json.dumps(_payload(graph, root), sort_keys=True, allow_nan=False),
        )

    @property
    def identity(self) -> str:
        """Hash canonical mathematics and contract, excluding name and source spelling."""
        return canonical_hash(json.loads(self._canonical))

    def to_payload(self) -> dict:
        """Return detached declaration and canonical payload for auditing."""
        return {
            "source": self.source,
            "name": self.name,
            "canonical": json.loads(self._canonical),
            "identity": self.identity,
        }

    def dumps(self) -> str:
        """Serialize source, descriptive name, canonical mathematics and identity."""
        return json.dumps(self.to_payload(), sort_keys=True, allow_nan=False)

    @classmethod
    def loads(cls, value: str) -> UserExpression:
        """Revalidate the declaration and its canonical relation on replay."""
        if type(value) is not str or len(value) > 65536:
            raise ValueError("serialized declaration exceeds budget")
        try:
            payload = json.loads(value)
            envelope_hash = canonical_hash(payload)
        except RecursionError as exc:
            raise ValueError(
                "serialized declaration exceeds JSON nesting budget"
            ) from exc
        if type(payload) is not dict or set(payload) != {
            "source",
            "name",
            "canonical",
            "identity",
        }:
            raise ValueError("invalid expression envelope")
        result = cls(payload["source"], name=payload["name"])
        # JSON identity preserves index types; Python equality aliases True/1/1.0.
        if canonical_hash(result.to_payload()) != envelope_hash:
            raise ValueError("expression declaration/canonical identity mismatch")
        return result

    def scalar_graph(self) -> tuple[Graph, Expr]:
        """Export detached canonical mathematics for compiler research, without admission."""
        payload = json.loads(self._canonical)
        graph = Graph()
        values: list[Expr] = []
        for operation, arguments, value in payload["nodes"]:
            operands = [values[i] for i in arguments]
            if operation == "constant":
                node = graph.constant(Fraction(value))
            elif operation == "variable":
                node = graph.variable(value)
            elif operation == "add":
                node = graph.add_many(operands)
            elif operation == "multiply":
                node = graph.multiply_many(operands)
            elif operation == "power":
                node = graph.power(operands[0], float(value))
            else:
                raise ValueError("unsupported sealed graph operation")
            values.append(node)
        return graph, values[payload["root"]]

    def evaluate(self, *, rho_a: float, rho_b: float) -> float:
        """Diagnostic CPU point only, in hartree/bohr^3; never a SCF callback."""
        values = {"rho_a": rho_a, "rho_b": rho_b}
        if any(type(v) not in (int, float) for v in values.values()):
            raise TypeError("point inputs require Python FP64 floats or integers")
        try:
            values = {k: float(v) for k, v in values.items()}
            if any(not math.isfinite(v) or v < 0 for v in values.values()):
                raise ValueError("rho inputs require finite nonnegative values")
            graph, root = self.scalar_graph()
            result = graph.evaluate(root, values)
            if not math.isfinite(result):
                raise ValueError("nonfinite pointwise result")
        except OverflowError as exc:
            raise ValueError("point outside finite FP64 evaluation domain") from exc
        return result

    def capability(self) -> dict:
        """Representation carries no per-expression numerical or endpoint entitlement."""
        return {
            "representable": True,
            "pointwise_validated": "UNQUALIFIED",
            "cpu": {
                "compiled": "UNQUALIFIED",
                "scf": "UNQUALIFIED",
                "forces": "UNQUALIFIED",
            },
            "cuda": {
                "compiled": "UNQUALIFIED",
                "scf": "UNQUALIFIED",
                "forces": "UNQUALIFIED",
            },
        }
