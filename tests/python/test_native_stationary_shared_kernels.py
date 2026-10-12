"""Compiler-level parity for one shared CUDA transport and source reducer.

Device numerical qualification belongs to the existing ten native CLI E+F
fixtures. These host checks prove that every promoted MethodIR graph is an
ordered, complete unit source sum before it can use the shared CUDA kernel.
"""

from __future__ import annotations

import struct
from fractions import Fraction

import pytest

from generativeqc_compiler.method.stationary_cuda import (
    _qualified_aot_profile,
    stationary_runtime_sources,
)
from generativeqc_compiler.method.stationary_native_cuda import (
    NATIVE_STATIONARY_CUDA_PROFILES,
    emit_native_stationary_profile_cuda,
)


@pytest.mark.parametrize("profile", NATIVE_STATIONARY_CUDA_PROFILES)
def test_native_shared_reduction_preserves_ir_source_order(profile: str) -> None:
    plan = _qualified_aot_profile(profile).plan
    sources = stationary_runtime_sources(plan)
    program = plan.reduction_program(atoms=1)
    reduction = program.outputs["gradient"]
    assert plan.source_names == sources
    assert reduction.op == "add"
    assert reduction.spec.shape == (1, 3)
    assert reduction.spec.dtype == "float64"
    assert tuple(node.attrs["name"] for node in reduction.inputs) == sources
    assert tuple(reduction.attrs["coefficients"]) == (Fraction(1),) * len(sources)

    # Cover cancellation and underflow in precisely the source order that the
    # original emitted reducer uses, not a reordered parallel tree.
    values = [
        1.0e16,
        1.0,
        -1.0e16,
        0.25,
        -0.125,
        1.0e-200,
        -1.0e-200,
        0.0625,
        0.0,
        -0.0,
    ]
    selected = [values[index % len(values)] for index in range(len(sources))]
    original = 0.0
    for coefficient, value in zip(reduction.attrs["coefficients"], selected, strict=True):
        original += float(coefficient) * value
    shared = 0.0
    for value in selected:
        shared += 1.0 * value
    assert struct.pack("=d", original) == struct.pack("=d", shared)


@pytest.mark.parametrize("profile", NATIVE_STATIONARY_CUDA_PROFILES)
def test_native_emission_uses_one_shared_reducer(profile: str) -> None:
    generated = emit_native_stationary_profile_cuda(profile)
    assert "#define GENERATIVEQC_NATIVE_SHARED_REDUCTION 1" in generated
    # The method graph still supplies its own XC and AO derivatives; only the
    # now-shared integral transport and final reduction are excluded.
    assert "stationary_evaluate_point(" in generated
    assert "__global__ void source_reduce(" not in generated
    assert "__global__ void native_source_transport(" not in generated
