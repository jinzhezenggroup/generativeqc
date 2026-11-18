"""Executable contracts for exact cross-graph scalar composition."""

from fractions import Fraction

import pytest
from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.integral.graph_composition import import_roots
from generativeqc_compiler.xc.grid_geometry_response import gga_field_response_program
from generativeqc_compiler.xc.spec import FunctionalSpec


def test_simultaneous_binding_preserves_shared_values_and_target_names() -> None:
    source, target = Graph(), Graph()
    first, second = source.variable("first"), source.variable("second")
    target_first, target_second = target.variable("first"), target.variable("second")
    product = first * second
    roots = import_roots(
        target,
        source,
        (product + first, product, product),
        {"first": target_second, "second": target_first + 1},
    )
    assert roots[1].identifier == roots[2].identifier
    assert [target.evaluate(root, {"first": 2, "second": 5}) for root in roots] == [
        20,
        15,
        15,
    ]


def test_lazy_branch_remains_lazy_in_imported_graph() -> None:
    source, target = Graph(), Graph()
    argument = source.variable("argument")
    root = source.select_le(argument, 0, 7, argument.pow(Fraction(1, 2)))
    (imported,) = import_roots(
        target, source, (root,), {"argument": target.variable("x")}
    )
    assert target.evaluate(imported, {"x": -2}) == 7
    assert target.evaluate(imported, {"x": 4}) == 2


def test_invalid_graphs_and_incomplete_bindings_do_not_mutate_target() -> None:
    source, target, foreign = Graph(), Graph(), Graph()
    argument = source.variable("argument")
    value = target.variable("x")
    wrong = foreign.variable("x")
    count = len(target.nodes)
    for roots, bindings in (
        ((wrong,), {"argument": value}),
        ((argument + 1,), {"argument": wrong}),
        ((argument + 1,), {"argument": value, "unused": wrong}),
        ((argument + 1,), {}),
    ):
        with pytest.raises(ValueError):
            import_roots(target, source, roots, bindings)
        assert len(target.nodes) == count
    assert import_roots(target, source, (), {}) == ()


def test_gga_geometry_program_refuses_a_different_ingredient_contract() -> None:
    functional = FunctionalSpec("LDA control", (("LDA_X", Fraction(1)),))
    with pytest.raises(ValueError, match="exactly rho/sigma"):
        gga_field_response_program(functional)
