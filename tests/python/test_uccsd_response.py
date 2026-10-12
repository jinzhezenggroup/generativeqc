"""Physical excitation duals and independently checked UCCSD derivative actions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.cc.uccsd_response import (
    build_parameter_vjp,
    build_response_programs,
)
from test_uccsd_equations import (
    _amplitudes,
    _determinant_oracle,
    _feeds,
    _random_hamiltonian,
)

from tools.generativeqc_cc.native_tensor_cpu import NativeCCTensorExecutor


def _seeds(layout: UCCSDLayout, packed: np.ndarray) -> dict:
    return {
        f"bar_{name.replace('t', 'r', 1)}": value
        for name, value in layout.unpack_dual(packed).items()
    }


def _pullback(layout: UCCSDLayout, outputs: dict) -> np.ndarray:
    return layout.pack_dual(
        {block.name: outputs[f"bar_{block.name}"] for block in layout.blocks}
    )


def _physical_residual(
    layout: UCCSDLayout, outputs: dict, prefix: str = ""
) -> np.ndarray:
    return np.concatenate(
        [
            block.pack_dual(outputs[f"{prefix}{block.name.replace('t', 'r', 1)}"])
            / block.full_metric_weight
            for block in layout.blocks
        ]
    )


@pytest.mark.parametrize(
    "populations",
    [
        (3, 2, 3, 2),
        (2, 2, 2, 2),
        (1, 1, 2, 1),
        (0, 2, 0, 2),
        (3, 0, 0, 3),
        (0, 0, 0, 0),
    ],
)
def test_expansion_transpose_and_minimum_norm_dual(populations: tuple) -> None:
    layout = UCCSDLayout(*populations)
    rng = np.random.default_rng(1824)
    packed = rng.normal(size=layout.packed_size)
    cotangents = {
        block.name: rng.normal(size=block.spec.shape) for block in layout.blocks
    }
    expanded = layout.unpack(packed)
    expected = sum(np.sum(expanded[name] * cotangents[name]) for name in expanded)
    np.testing.assert_allclose(
        np.dot(packed, layout.pack_dual(cotangents)), expected, atol=2e-13, rtol=0
    )
    np.testing.assert_array_equal(layout.pack_dual(layout.unpack_dual(packed)), packed)
    for block in layout.blocks:
        lifted = layout.unpack_dual(packed)[block.name]
        np.testing.assert_array_equal(
            lifted * block.full_metric_weight, expanded[block.name]
        )
        if block.same_spin and block.packed_layout.storage_elements:
            bad_amplitude = cotangents[block.name]
            with pytest.raises(ValueError, match="antisymmetric"):
                block.pack(bad_amplitude)


@pytest.mark.parametrize(
    "populations", [(2, 2, 2, 2), (2, 1, 2, 2), (0, 2, 0, 2), (3, 0, 0, 3)]
)
def test_native_jvp_vjp_physical_transpose_and_fermionic_finite_difference(
    populations: tuple,
) -> None:
    layout = UCCSDLayout(*populations)
    programs = build_response_programs(layout)
    amplitudes = _amplitudes(layout)
    fock, integrals = _random_hamiltonian(layout)
    feeds = _feeds(layout, fock, integrals, amplitudes)
    executor = NativeCCTensorExecutor(max_bytes=128 << 20)
    rng = np.random.default_rng(1825)
    direction = rng.normal(scale=0.1, size=layout.packed_size)
    dual = rng.normal(scale=0.1, size=layout.packed_size)
    jvp = executor.execute(
        programs.residual_jvp.program,
        {
            **feeds,
            **{f"d_{name}": value for name, value in layout.unpack(direction).items()},
        },
    )
    jacobian_direction = _physical_residual(layout, jvp, "d_")
    transpose = _pullback(
        layout,
        executor.execute(
            programs.residual_vjp.program, {**feeds, **_seeds(layout, dual)}
        ),
    )
    np.testing.assert_allclose(
        np.dot(dual, jacobian_direction),
        np.dot(transpose, direction),
        atol=2e-13,
        rtol=0,
    )
    lagrangian = _pullback(
        layout,
        executor.execute(
            programs.stationarity_vjp.program,
            {**feeds, **_seeds(layout, dual), "bar_correlation_energy": np.array(1.0)},
        ),
    )
    packed = layout.pack(amplitudes)
    values = []
    for sign in (-1, 1):
        perturbed = layout.unpack(packed + sign * 1e-5 * direction)
        outputs = _determinant_oracle(layout, fock, integrals, perturbed)
        values.append(
            float(outputs["correlation_energy"])
            + np.dot(dual, _physical_residual(layout, outputs))
        )
    np.testing.assert_allclose(
        np.dot(lagrangian, direction), (values[1] - values[0]) / 2e-5, atol=2e-9, rtol=0
    )
    if populations == (2, 2, 2, 2):
        wrong_seeds = {
            f"bar_{name.replace('t', 'r', 1)}": value
            for name, value in layout.unpack(dual).items()
        }
        wrong = _pullback(
            layout,
            executor.execute(programs.residual_vjp.program, {**feeds, **wrong_seeds}),
        )
        assert abs(np.dot(wrong - transpose, direction)) > 1e-5


def test_all_canonical_parameter_vjps_close_against_fermionic_hamiltonian() -> None:
    layout = UCCSDLayout(2, 1, 2, 2)
    programs = build_response_programs(layout)
    amplitudes = _amplitudes(layout)
    fock, integrals = _random_hamiltonian(layout)
    feeds = _feeds(layout, fock, integrals, amplitudes)
    rng = np.random.default_rng(1826)
    dual = rng.normal(scale=0.1, size=layout.packed_size)
    spins = np.repeat((0, 1, 0, 1), (layout.oa, layout.ob, layout.va, layout.vb))
    fock_direction = rng.normal(scale=0.1, size=fock.shape)
    fock_direction = (fock_direction + fock_direction.T) * (
        spins[:, None] == spins[None, :]
    )
    factors = rng.normal(scale=0.1, size=(*fock.shape, 3))
    factors = (factors + factors.swapaxes(0, 1)) * (spins[:, None] == spins[None, :])[
        :, :, None
    ]
    coulomb = np.einsum("prx,qsx->pqrs", factors, factors)
    integral_direction = coulomb - coulomb.swapaxes(2, 3)
    directions = _feeds(layout, fock_direction, integral_direction, amplitudes)
    executor = NativeCCTensorExecutor(max_bytes=128 << 20)
    derivative = 0.0
    for parameter in programs.parameter_names:
        action = build_parameter_vjp(programs, parameter)
        values = executor.execute(
            action.program,
            {**feeds, **_seeds(layout, dual), "bar_correlation_energy": np.array(1.0)},
        )[f"bar_{parameter}"]
        derivative += float(np.sum(values * directions[parameter]))
    values = []
    for sign in (-1, 1):
        outputs = _determinant_oracle(
            layout,
            fock + sign * 1e-5 * fock_direction,
            integrals + sign * 1e-5 * integral_direction,
            amplitudes,
        )
        values.append(
            float(outputs["correlation_energy"])
            + np.dot(dual, _physical_residual(layout, outputs))
        )
    np.testing.assert_allclose(
        derivative, (values[1] - values[0]) / 2e-5, atol=3e-9, rtol=0
    )


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((2, 2), dtype=np.float32),
        np.array([[float("nan")]]),
        np.ma.array(np.zeros((2, 2))),
    ],
)
def test_dual_maps_reject_non_fp64_nonfinite_and_masked_inputs(value: Any) -> None:
    layout = UCCSDLayout(2, 2, 2, 2)
    fields = {block.name: np.zeros(block.spec.shape) for block in layout.blocks}
    fields["t1a"] = value
    with pytest.raises(ValueError):
        layout.pack_dual(fields)
