"""Exact coefficient identities must survive symbolic multiplication and AD."""

import math
from fractions import Fraction

import pytest
from generativeqc_compiler.integral.expr import Expr, Graph

NEAR_ONE = (Fraction(2**53 + 1, 2**53), Fraction(2**54 - 1, 2**54))


def exact_polynomial_value(graph: Graph, root: Expr, x: Fraction) -> Fraction:
    """Interpret only rational polynomial nodes, independently of float evaluate."""
    values = {}
    for identifier in graph.topological_order((root,)):
        node = graph.nodes[identifier]
        if node.operation == "constant":
            assert isinstance(node.payload, Fraction)
            value = node.payload
        elif node.operation == "variable":
            assert node.payload == "x"
            value = x
        elif node.operation == "add":
            value = sum((values[item] for item in node.arguments), Fraction(0))
        else:
            assert node.operation == "multiply"
            value = math.prod(values[item] for item in node.arguments)
        values[identifier] = value
    return values[root.identifier]


@pytest.mark.parametrize("q", NEAR_ONE)
@pytest.mark.parametrize("constant_first", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_near_unity_product_and_exported_derivatives(
    q: Fraction, constant_first: bool, reverse: bool
) -> None:
    graph = Graph()
    if constant_first:
        c, x = graph.constant(q), graph.variable("x")
    else:
        x, c = graph.variable("x"), graph.constant(q)
    operands = (c, x) if reverse else (x, c)
    for root in (graph.multiply(*operands), graph.multiply_many(operands)):
        derivative = graph.differentiate(root, x)
        assert graph.node(derivative).payload == q
        assert graph.node(graph.differentiate(derivative, x)).payload == 0
        assert graph.differentiate(root, x).identifier == derivative.identifier
        rebuilt, (exported,) = graph.canonicalize_associative((root,))
        exported_derivative = rebuilt.differentiate(exported, rebuilt.variable("x"))
        assert rebuilt.node(exported_derivative).payload == q


@pytest.mark.parametrize("q", (*NEAR_ONE, Fraction(1, 10**400), Fraction(10**400)))
def test_exact_constant_predicates_folding_and_cancellation(q: Fraction) -> None:
    graph = Graph()
    c = graph.constant(q)
    assert graph.is_constant(c)
    assert graph.is_constant(c, q)
    assert not graph.is_constant(c, 0)
    assert not graph.is_constant(c, 1)
    assert c != graph.constant(1)
    for left, right in ((c, graph.constant(3)), (graph.constant(3), c)):
        assert graph.node(graph.multiply(left, right)).payload == 3 * q
    assert graph.node(c + graph.constant(-q)).payload == 0
    assert graph.node(c * graph.constant(1 / q)).payload == 1
    # Exact zero alone annihilates; an underflowing rational must remain live.
    x = graph.variable("x")
    for operands in ((c, x), (x, c)):
        root = graph.multiply(*operands)
        assert graph.node(graph.differentiate(root, x)).payload == q
    assert graph.node(c + graph.constant(2)).payload == q + 2


@pytest.mark.parametrize("q", NEAR_ONE)
def test_polynomial_derivatives_and_rebuild_have_independent_fraction_oracle(
    q: Fraction,
) -> None:
    graph = Graph()
    c = graph.constant(q)
    x = graph.variable("x")
    # Hand derivatives of q*x*x + (q-1)*x: 2*q*x+(q-1), 2*q, 0.
    root = graph.multiply_many((c, x, x)) + (q - 1) * x
    first = graph.differentiate(root, x)
    second = graph.differentiate(first, x)
    third = graph.differentiate(second, x)
    roots = (root, first, second, third)
    for point in (Fraction(-3, 2), Fraction(0), Fraction(7, 3)):
        expected = (
            q * point * point + (q - 1) * point,
            2 * q * point + q - 1,
            2 * q,
            Fraction(0),
        )
        assert tuple(exact_polynomial_value(graph, r, point) for r in roots) == expected
        rebuilt, exported = graph.canonicalize_associative(roots)
        assert (
            tuple(exact_polynomial_value(rebuilt, r, point) for r in exported)
            == expected
        )
        again, repeated = rebuilt.canonicalize_associative(exported)
        assert again.nodes == rebuilt.nodes
        assert tuple(r.identifier for r in repeated) == tuple(
            r.identifier for r in exported
        )


@pytest.mark.parametrize("approximate", (False, True))
def test_real_float_and_exact_zero_one_controls(approximate: bool) -> None:
    graph = Graph()
    make = graph.approximate_constant if approximate else graph.constant
    x = graph.variable("x")
    zero, one = make(0.0), make(1.0)
    for a, b in ((x, one), (one, x)):
        assert graph.multiply(a, b) == x
    for a, b in ((x, zero), (zero, x)):
        assert graph.is_constant(graph.multiply(a, b), 0)
    assert graph.add(x, zero) == x
    adjacent = math.nextafter(1.0, 2.0)
    c = make(adjacent)
    assert not graph.is_constant(c, 1)
    assert graph.is_constant(c, adjacent)
    assert graph.node(graph.differentiate(c * x, x)).payload == graph.node(c).payload
    assert not graph.is_constant(one, NEAR_ONE[0])
    assert not graph.is_constant(x)


def test_decimal_literal_and_approximate_predicate_contract() -> None:
    graph = Graph()
    assert graph.is_constant(graph.constant(0.1), 0.1)
    approximate = graph.approximate_constant(0.1)
    assert graph.is_constant(approximate, 0.1)
    assert not graph.is_constant(approximate, Fraction(1, 10))
    assert not graph.is_constant(graph.constant(NEAR_ONE[0]), 1.0)


def test_distinct_rationals_with_same_binary64_value_keep_distinct_nodes() -> None:
    graph = Graph()
    values = (*NEAR_ONE, Fraction(1))
    assert len({float(q) for q in values}) == 1
    constants = tuple(graph.constant(q) for q in values)
    assert len({c.identifier for c in constants}) == len(values)
    assert tuple(graph.node(c).payload for c in constants) == values


@pytest.mark.parametrize("q", NEAR_ONE)
def test_custom_boys_leaf_rule_retains_exact_multiplier(q: Fraction) -> None:
    graph = Graph()
    c = graph.constant(q)
    argument = graph.variable("T")
    boys = graph.variable("F_n")
    next_boys = graph.variable("F_next")
    derivative = graph.differentiate(c * boys, argument, {"F_n": -next_boys})
    coefficient = graph.differentiate(derivative, next_boys)
    assert graph.node(coefficient).payload == -q
