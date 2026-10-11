"""Share scalar work on a producer-proven equal-spin slice of polarized XC."""

from collections.abc import Sequence

from generativeqc_compiler.integral.expr import Expr, Graph

EQUAL_SPIN_BINDINGS = (
    ("rho_b", "rho_a"),
    ("sigma_bb", "sigma_aa"),
    ("tau_b", "tau_a"),
)


def equal_spin_roots(graph: Graph, roots: Sequence[Expr]) -> tuple[Expr, ...]:
    """Bind paired work inputs *after* the polarized derivatives are built.

    Differentiating the restricted energy instead would lose independent spin
    partials and would require a different chain rule. Structural substitution
    retains all original operations and branch policies, sharing only identical
    subgraphs. In particular, sigma_ab remains independent: Libxc's work-input
    floors can make it differ from the equal diagonal sigmas even for RKS.
    Callers must prove equality of the paired sanitized inputs; a method label
    or approximate equality is not sufficient. This is not a response lowering.
    """
    replacements = {
        graph.variable(source): graph.variable(target)
        for source, target in EQUAL_SPIN_BINDINGS
    }
    return graph.replace_subexpressions(roots, replacements)
