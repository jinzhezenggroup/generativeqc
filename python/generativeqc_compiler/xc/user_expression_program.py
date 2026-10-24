"""Bounded polynomial feature derivatives using the canonical scalar Graph."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from generativeqc_compiler.common.provenance import canonical_hash

from .program import output_set
from .user_expression import FEATURES, UserExpression

if TYPE_CHECKING:
    from pathlib import Path

    from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
    from generativeqc_compiler.common.cuda_runtime import CudaArtifact
    from generativeqc_compiler.integral.expr import Expr, Graph

MAX_DERIVATIVE_GRAPH = 8192
MAX_POINTS = 65536


def _check_graph(graph: Graph) -> None:
    if len(graph.nodes) > MAX_DERIVATIVE_GRAPH:
        raise ValueError("expression derivative graph exceeds node budget")
    for node in graph.nodes:
        if node.operation == "constant":
            value = node.payload
            if (
                not isinstance(value, Fraction)
                or max(value.numerator.bit_length(), value.denominator.bit_length())
                > 512
            ):
                raise ValueError("expression derivative coefficient exceeds budget")


@dataclass(frozen=True)
class UserExpressionProgram:
    """Immutable recipe; exported graphs are detached, never retained handles.

    Outputs are energy, two feature derivatives, and the packed upper Hessian.
    These feature derivatives do not constitute SCF or nuclear-force admission.
    """

    expression: UserExpression
    order: int = 2

    @property
    def features(self) -> tuple[str, ...]:
        """Return the two immutable declared spin-density ingredient names."""
        return FEATURES

    def __post_init__(self) -> None:
        """Validate the sealed declaration, derivative order and graph budgets."""
        if type(self.expression) is not UserExpression:
            raise TypeError("program requires a sealed UserExpression")
        output_set(self, self.order)
        self.scalar_graph()

    @property
    def outputs(self) -> tuple[tuple[int, ...], ...]:
        """Return ordered energy, feature gradient and packed Hessian labels."""
        return output_set(self, self.order)

    def scalar_graph(self) -> tuple[Graph, tuple[Expr, ...]]:
        """Differentiate and canonicalize with the existing mathematical owner."""
        graph, energy = self.expression.scalar_graph()
        variables = tuple(graph.variable(name) for name in FEATURES)
        derivatives = {(): energy}
        for output in self.outputs:
            for depth in range(1, len(output) + 1):
                key = output[:depth]
                if key not in derivatives:
                    derivatives[key] = graph.differentiate(
                        derivatives[key[:-1]], variables[key[-1]]
                    )
                    _check_graph(graph)
        roots = tuple(derivatives[key] for key in self.outputs)
        graph, roots = graph.canonicalize_associative(roots)
        _check_graph(graph)
        graph, roots = graph.lower_small_integer_powers(roots)
        _check_graph(graph)
        return graph, roots

    @property
    def identity(self) -> str:
        """Hash the detached canonical derivative program and output contract."""
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict:
        """Return detached canonical mathematics and bounded output metadata."""
        graph, roots = self.scalar_graph()
        reachable = graph.topological_order(roots)
        indices = {node: i for i, node in enumerate(reachable)}
        return {
            "schema": "generativeqc.xc-expression-program.v1",
            "expression": self.expression.identity,
            "outputs": self.outputs,
            "nodes": [
                (
                    graph.nodes[i].operation,
                    [indices[j] for j in graph.nodes[i].arguments],
                    str(graph.nodes[i].payload),
                )
                for i in reachable
            ],
            "roots": [indices[root.identifier] for root in roots],
            "max_points": MAX_POINTS,
        }

    def compile(
        self, *, compiler: CppCompilerAdapter, cache: str | Path
    ) -> CudaArtifact:
        """Compile a verified CPU artifact without loading or executing it.

        The shared artifact type also serves CPU compilers; its name does not
        imply a CUDA target. Runtime allocation and FFI belong to the public
        runtime adapter constructed by ``extensions.xc.compile_expression``.
        """
        from .user_expression_native import compile_native

        return compile_native(self, compiler=compiler, cache=cache)
