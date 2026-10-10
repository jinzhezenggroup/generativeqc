"""Symbolic comparisons create bool values without Python branch decisions."""

import operator
import typing

import pytest
from generativeqc_compiler.array_api import VibeArray, input_array, trace
from generativeqc_compiler.tensor import Index, IndexSpace, TensorSpec


def _spec() -> TensorSpec:
    return TensorSpec((Index("i", IndexSpace("ao", "ao", 2)),), role="input")


@pytest.mark.parametrize("comparison", [operator.eq, operator.ne])
@pytest.mark.parametrize("case", ["different", "same", "rewrapped"])
def test_symbolic_comparisons_are_boolean_data(
    comparison: typing.Callable[[object, object], object], case: str
) -> None:
    x = input_array("x", _spec())
    y = input_array("y", _spec())
    operands = {
        "different": (x, y),
        "same": (x, x),
        "rewrapped": (x, VibeArray(x.node)),
    }
    result = comparison(*operands[case])
    assert isinstance(result, VibeArray)
    assert result.dtype == "bool"
    assert not result.node.spec.differentiable


@pytest.mark.parametrize("comparison", [operator.eq, operator.ne])
@pytest.mark.parametrize("other", [0, None])
def test_scientific_scalar_comparison_requires_explicit_domain(
    comparison: typing.Callable[[object, object], object], other: object
) -> None:
    with pytest.raises(TypeError, match="scientific domains"):
        comparison(input_array("x", _spec()), other)


@pytest.mark.parametrize("comparison", [operator.eq, operator.ne])
def test_trace_rejects_comparison_driven_control_flow(
    comparison: typing.Callable[[object, object], object],
) -> None:
    def expression(x: VibeArray, y: VibeArray) -> VibeArray:
        return x if comparison(x, y) else y

    with pytest.raises(TypeError, match="cannot drive Python control flow"):
        trace(expression, {"x": _spec(), "y": _spec()})


def test_explicit_node_identity_remains_available() -> None:
    x = input_array("x", _spec())
    wrapped = VibeArray(x.node)
    assert x.node is wrapped.node
    assert x.shape == (2,)
    assert x.dtype == "float64"
