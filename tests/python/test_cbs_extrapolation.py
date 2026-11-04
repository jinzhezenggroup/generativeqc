"""Real C ABI extrapolation, independent decimal oracles and atomic refusals."""

from __future__ import annotations

import ctypes
import math
import random
from decimal import Decimal, localcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import _native
from generativeqc._cbs_plan import _Descriptor, _Plan, native_cbs_plan

if TYPE_CHECKING:
    from typing import Any


def decimal_plan(
    low: int, high: int, exponent: float, alpha: float, mode: str
) -> tuple[tuple[float, ...], ...]:
    """Solve the two-point error model with independent high-precision arithmetic."""
    with localcontext() as context:
        context.prec = 80
        low_error = Decimal(low) ** -Decimal.from_float(exponent)
        high_error = Decimal(high) ** -Decimal.from_float(exponent)
        mp2 = (
            -high_error / (low_error - high_error),
            low_error / (low_error - high_error),
        )
        if mode == "exponential":
            low_error = (-Decimal.from_float(alpha) * low).exp()
            high_error = (-Decimal.from_float(alpha) * high).exp()
            hf = (
                -high_error / (low_error - high_error),
                low_error / (low_error - high_error),
            )
        else:
            hf = (Decimal(0), Decimal(1))
        coefficients = (hf[0] - mp2[0], hf[1] - mp2[1], *mp2, Decimal(1), Decimal(-1))
        return tuple(tuple(map(float, values)) for values in (hf, mp2, coefficients))


@pytest.mark.parametrize("mode", ("exponential", "highest"))
def test_native_plan_matches_independent_decimal_models(
    policy_library: ctypes.CDLL, mode: str
) -> None:
    """Cover ordinary, close-cardinal and cancellation-sensitive input domains."""
    random_source = random.Random(2227)
    cases = [(2, 3, 3.0, 1.63), (99, 100, 1e-12, 1e-12), (2, 100, 80.0, 400.0)]
    for _ in range(64):
        low = random_source.randint(2, 99)
        high = random_source.randint(low + 1, 100)
        cases.append(
            (
                low,
                high,
                10 ** random_source.uniform(-9, 1),
                10 ** random_source.uniform(-9, 1),
            )
        )
    for low, high, exponent, alpha in cases:
        plan = native_cbs_plan(policy_library, low, high, exponent, alpha, mode)
        expected = decimal_plan(low, high, exponent, alpha, mode)
        for actual, reference in zip(
            (plan.hf_weights, plan.mp2_weights, plan.coefficients),
            expected,
            strict=True,
        ):
            np.testing.assert_allclose(actual, reference, rtol=5e-13, atol=3e-15)
    np.testing.assert_allclose(
        native_cbs_plan(policy_library, 2, 3, 3.0, 1.63, mode).mp2_weights,
        (-8 / 19, 27 / 19),
        rtol=0,
        atol=1e-15,
    )


@pytest.mark.parametrize("mode", ("exponential", "highest"))
def test_native_plan_accepts_finite_exponent_limiting_weights(
    policy_library: ctypes.CDLL, mode: str
) -> None:
    """Overflow in a negative decay exponent is not a divergent extrapolation."""
    plan = native_cbs_plan(
        policy_library, 2, 100, float.fromhex("0x1.fffffffffffffp+1023"), 1e308, mode
    )
    assert plan.hf_weights == plan.mp2_weights == (0.0, 1.0)
    assert plan.coefficients == (0.0, 0.0, 0.0, 1.0, 1.0, -1.0)


def descriptor() -> _Descriptor:
    """A valid copied descriptor; each refusal test mutates only one contract."""
    return _Descriptor(
        ctypes.sizeof(_Descriptor), _native.ABI_VERSION, 2, 3, 0, 0, 3.0, 1.63
    )


def sentinel_output() -> _Plan:
    output = _Plan.from_buffer_copy(b"\xa5" * ctypes.sizeof(_Plan))
    output.struct_size = ctypes.sizeof(_Plan)
    output.abi_version = _native.ABI_VERSION
    return output


def query_function(library: ctypes.CDLL) -> Any:
    query = library.generativeqc_cbs_extrapolation_query_v1
    query.argtypes = [ctypes.POINTER(_Descriptor), ctypes.POINTER(_Plan)]
    query.restype = ctypes.c_int
    return query


@pytest.mark.parametrize(
    "field,value,status",
    (
        ("struct_size", ctypes.sizeof(_Descriptor) - 1, _native.STATUS_ABI_MISMATCH),
        ("abi_version", _native.ABI_VERSION + 1, _native.STATUS_ABI_MISMATCH),
        ("low_cardinal", 1, _native.STATUS_INVALID_ARGUMENT),
        ("low_cardinal", 3, _native.STATUS_INVALID_ARGUMENT),
        ("low_cardinal", 4, _native.STATUS_INVALID_ARGUMENT),
        ("high_cardinal", 101, _native.STATUS_INVALID_ARGUMENT),
        ("hf_extrapolation", 2, _native.STATUS_INVALID_ARGUMENT),
        ("reserved", 1, _native.STATUS_INVALID_ARGUMENT),
        *(
            (field, value, _native.STATUS_INVALID_ARGUMENT)
            for field in ("correlation_exponent", "hf_exponent")
            for value in (0.0, -1.0, math.nan, math.inf, -math.inf, 5e-324)
        ),
    ),
)
def test_native_refusal_preserves_output(
    policy_library: ctypes.CDLL, field: str, value: float, status: int
) -> None:
    inputs = descriptor()
    setattr(inputs, field, value)
    output = sentinel_output()
    original = bytes(output)
    assert (
        query_function(policy_library)(ctypes.byref(inputs), ctypes.byref(output))
        == status
    )
    assert bytes(output) == original


@pytest.mark.parametrize("field,value", (("struct_size", 0), ("abi_version", 1)))
def test_output_header_refusal_is_atomic(
    policy_library: ctypes.CDLL, field: str, value: int
) -> None:
    inputs = descriptor()
    output = sentinel_output()
    setattr(output, field, value)
    original = bytes(output)
    assert (
        query_function(policy_library)(ctypes.byref(inputs), ctypes.byref(output))
        == _native.STATUS_ABI_MISMATCH
    )
    assert bytes(output) == original


def test_null_buffers_and_overlapping_publication(policy_library: ctypes.CDLL) -> None:
    query = query_function(policy_library)
    inputs = descriptor()
    output = sentinel_output()
    original = bytes(output)
    assert query(None, ctypes.byref(output)) == _native.STATUS_INVALID_ARGUMENT
    assert bytes(output) == original
    assert query(ctypes.byref(inputs), None) == _native.STATUS_INVALID_ARGUMENT
    assert query(None, None) == _native.STATUS_INVALID_ARGUMENT
    storage = (ctypes.c_double * (ctypes.sizeof(_Plan) // 8 + 1))()
    overlapping_input = _Descriptor.from_buffer(storage)
    overlapping_output = _Plan.from_buffer(storage)
    ctypes.memmove(
        ctypes.addressof(storage), ctypes.byref(inputs), ctypes.sizeof(inputs)
    )
    overlapping_input.struct_size = ctypes.sizeof(_Plan)
    storage[-1] = 97.0
    assert (
        query(ctypes.byref(overlapping_input), ctypes.byref(overlapping_output))
        == _native.STATUS_SUCCESS
    )
    expected = native_cbs_plan(policy_library, 2, 3, 3.0, 1.63, "exponential")
    assert tuple(overlapping_output.coefficients) == expected.coefficients
    assert storage[-1] == 97.0


@pytest.mark.parametrize("failure", ("missing", "refused", "bad_header", "nonfinite"))
def test_python_transport_never_reconstructs_a_failed_plan(failure: str) -> None:
    """The selected owner must supply a complete, finite receipt."""

    def query(inputs: Any, output: Any) -> int:
        if failure == "refused":
            return _native.STATUS_INVALID_ARGUMENT
        plan = ctypes.cast(output, ctypes.POINTER(_Plan)).contents
        if failure == "bad_header":
            plan.abi_version += 1
        else:
            plan.coefficients[0] = math.nan
        return _native.STATUS_SUCCESS

    library = SimpleNamespace()
    if failure != "missing":
        library.generativeqc_cbs_extrapolation_query_v1 = query
    with pytest.raises(
        (ValueError, RuntimeError), match="extrapolation|receipt|weights"
    ):
        native_cbs_plan(library, 2, 3, 3.0, 1.63, "exponential")


def test_python_transport_rejects_unknown_hf_mode(policy_library: ctypes.CDLL) -> None:
    with pytest.raises(ValueError, match="hf_extrapolation"):
        native_cbs_plan(policy_library, 2, 3, 3.0, 1.63, "guess")
