"""Native CBS output order, rounding and all-or-nothing publication."""

import ctypes
import math
from typing import Any

import numpy as np
import pytest
from generativeqc import _native
from generativeqc._cbs_plan import (
    CBSPlan,
    NativeCbsComposer,
    _Composition,
    _DoublePointer,
    _Observables,
    _Plan,
    native_cbs_plan,
)


@pytest.mark.parametrize("mode", ("exponential", "highest"))
def test_native_cbs_composition_matches_completed_observables(
    policy_library: ctypes.CDLL, mode: str
) -> None:
    """Compute focal-point contributions independently, including repeated roles."""
    plan = native_cbs_plan(policy_library, 2, 3, 3.0, 1.63, mode)
    compose = NativeCbsComposer(policy_library, plan)
    energies = (-75.4, -75.6, -75.64, -75.82, -75.66, -75.64)
    source = np.random.default_rng(9244).normal(size=(5, 3, 3))
    forces = (*source, source[2])
    actual, force_blocks = compose(energies, forces)
    energy_only, omitted = compose(energies, None)
    assert omitted is None and energy_only == actual
    hf = math.fsum(
        weight * value
        for weight, value in zip(plan.hf_weights, energies[:2], strict=True)
    )
    mp2 = math.fsum(
        weight * (correlated - reference)
        for weight, correlated, reference in zip(
            plan.mp2_weights, energies[2:4], energies[:2], strict=True
        )
    )
    delta = energies[4] - energies[5]
    assert actual == (math.fsum((hf, mp2, delta)), hf, mp2, delta)
    hf_force = sum(
        weight * vector
        for weight, vector in zip(plan.hf_weights, forces[:2], strict=True)
    )
    mp2_force = sum(
        weight * (correlated - reference)
        for weight, correlated, reference in zip(
            plan.mp2_weights, forces[2:4], forces[:2], strict=True
        )
    )
    delta_force = forces[4] - forces[5]
    np.testing.assert_array_equal(
        force_blocks,
        (hf_force + mp2_force + delta_force, hf_force, mp2_force, delta_force),
    )


def test_native_cbs_composition_keeps_compensated_energy(
    policy_library: ctypes.CDLL,
) -> None:
    """Keep the residual from three large cancelling contribution terms."""
    plan = CBSPlan((1.0, 0.0), (0.0, 1.0), (1.0, -1.0, 0.0, 1.0, 1.0, -1.0))
    compose = NativeCbsComposer(policy_library, plan)
    assert compose((1e16, 0.0, 1e16, 1.0, -1e16, 0.0), None)[0] == (
        1.0,
        1e16,
        1.0,
        -1e16,
    )


@pytest.mark.parametrize(
    "failure", ("capacity", "extent", "header", "nonfinite", "overflow")
)
def test_native_cbs_composition_refusal_preserves_outputs(
    policy_library: ctypes.CDLL, failure: str
) -> None:
    plan = native_cbs_plan(policy_library, 2, 3, 3.0, 1.63, "exponential")
    native = NativeCbsComposer(policy_library, plan)
    source = np.ones((6, 3), dtype=np.float64)
    destination = np.full(12, 555.0)
    components = _Observables(ctypes.sizeof(_Observables), _native.ABI_VERSION)
    components.energy[:] = (-1.0,) * 6
    components.forces[:] = [item.ctypes.data_as(_DoublePointer) for item in source]
    components.force_count = 3
    output = _Composition(ctypes.sizeof(_Composition), _native.ABI_VERSION)
    output.energy[:] = (987.0,) * 4
    output.force_blocks = destination.ctypes.data_as(_DoublePointer)
    output.force_capacity = destination.size
    expected = _native.STATUS_INVALID_ARGUMENT
    if failure == "capacity":
        output.force_capacity -= 1
    elif failure == "extent":
        components.force_count = 2
    elif failure == "header":
        components.abi_version = _native.ABI_VERSION + 1
        expected = _native.STATUS_ABI_MISMATCH
    elif failure == "nonfinite":
        source[-1, -1] = np.nan
        expected = _native.STATUS_NUMERICAL_FAILURE
    else:
        source[4, -1] = np.finfo(np.float64).max
        source[5, -1] = -source[4, -1]
        expected = _native.STATUS_NUMERICAL_FAILURE
    saved = bytes(output)
    invoke: Any = policy_library.generativeqc_cbs_compose_v1
    assert invoke.argtypes == [
        ctypes.POINTER(_Plan),
        ctypes.POINTER(_Observables),
        ctypes.POINTER(_Composition),
    ]
    assert (
        invoke(
            ctypes.byref(native._plan), ctypes.byref(components), ctypes.byref(output)
        )
        == expected
    )
    assert bytes(output) == saved
    np.testing.assert_array_equal(destination, 555.0)
