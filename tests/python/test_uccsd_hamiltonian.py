"""Independent physical binding, composed CC adjoints and shared UHF actions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.cc.gradient_equations import (
    build_ao_eri_weight_block_program,
)
from generativeqc_compiler.cc.uccsd_hamiltonian import (
    RAW_INPUTS,
    build_lagrangian_programs,
    build_raw_hamiltonian,
)
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from test_uccsd_equations import _amplitudes, _determinant_oracle, _feeds
from test_uccsd_response import _physical_residual

from tools.generativeqc_cc.native_tensor_cpu import NativeCCTensorExecutor


def test_shared_shell_transform_uses_explicit_admission_beyond_oracle_frontier() -> (
    None
):
    """A production bound, not the historical 12-MO oracle, controls the shape."""
    rng = np.random.default_rng(18245)
    extent, shape = 13, (1, 2, 1, 1)
    with pytest.raises(ValueError):
        build_ao_eri_weight_block_program(extent, shape)
    with pytest.raises(ValueError, match="admission"):
        build_ao_eri_weight_block_program(extent, shape, max_elements=extent**4 - 1)
    program = build_ao_eri_weight_block_program(extent, shape, max_elements=extent**4)
    coefficients = [rng.normal(size=(size, extent)) for size in shape]
    integrals = rng.normal(size=(extent,) * 4)
    executor = NativeCCTensorExecutor(max_bytes=128 << 20)
    actual = executor.execute(
        program,
        {
            "eri": integrals,
            **{
                f"coefficients_{slot}": value for slot, value in enumerate(coefficients)
            },
        },
    )["eri"]
    expected = np.einsum(
        "up,vq,wr,xs,pqrs->uvwx", *coefficients, integrals, optimize=True
    )
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)


def _inputs(layout: UCCSDLayout) -> dict[str, np.ndarray]:
    """Build physical within-pair Coulomb symmetries from independent spin factors."""
    rng = np.random.default_rng(18242)
    counts = {"a": layout.oa + layout.va, "b": layout.ob + layout.vb}
    factors, values = {}, {}
    for spin in "ab":
        extent = counts[spin]
        factor = rng.normal(scale=0.1, size=(extent, extent, 3))
        factors[spin] = factor + factor.swapaxes(0, 1)
        matrix = rng.normal(scale=0.1, size=(extent, extent))
        values[f"h_{spin}"] = matrix + matrix.T
        values[f"rotation_{spin}"] = np.eye(extent)
    for pair in ("aa", "ab", "bb"):
        values[f"g_{pair}"] = np.einsum(
            "pqP,rsP->pqrs", factors[pair[0]], factors[pair[1]]
        )
    return values


def _reference(layout: UCCSDLayout, values: dict) -> tuple:
    """Explicit full spin-orbital direct/exchange, independent of compiler views."""
    occupied = {"a": layout.oa, "b": layout.ob}
    counts = {"a": layout.oa + layout.va, "b": layout.ob + layout.vb}
    rotation = {spin: values[f"rotation_{spin}"] for spin in "ab"}
    hcore = {
        spin: rotation[spin].T @ values[f"h_{spin}"] @ rotation[spin] for spin in "ab"
    }
    integrals = {
        pair: np.einsum(
            "up,vq,wr,xs,uvwx->pqrs",
            rotation[pair[0]],
            rotation[pair[0]],
            rotation[pair[1]],
            rotation[pair[1]],
            values[f"g_{pair}"],
            optimize=True,
        )
        for pair in ("aa", "ab", "bb")
    }
    integrals["ba"] = integrals["ab"].transpose(2, 3, 0, 1)
    density = {
        spin: np.diag([1.0] * occupied[spin] + [0.0] * (counts[spin] - occupied[spin]))
        for spin in "ab"
    }
    fock = {
        spin: hcore[spin]
        + sum(
            np.einsum("pqrs,rs->pq", integrals[spin + source], density[source])
            for source in "ab"
        )
        - np.einsum("prqs,rs->pq", integrals[spin * 2], density[spin])
        for spin in "ab"
    }
    owners = [*"a" * layout.oa, *"b" * layout.ob, *"a" * layout.va, *"b" * layout.vb]
    local = [
        *range(layout.oa),
        *range(layout.ob),
        *range(layout.oa, counts["a"]),
        *range(layout.ob, counts["b"]),
    ]
    extent = len(owners)
    spin_fock = np.zeros((extent, extent))
    spin_integrals = np.zeros((extent,) * 4)
    for first, second in np.ndindex(spin_fock.shape):
        if owners[first] == owners[second]:
            spin_fock[first, second] = fock[owners[first]][local[first], local[second]]
    for first, second, third, fourth in np.ndindex(spin_integrals.shape):
        if owners[first] == owners[third] and owners[second] == owners[fourth]:
            spin_integrals[first, second, third, fourth] += integrals[
                owners[first] + owners[second]
            ][local[first], local[third], local[second], local[fourth]]
        if owners[first] == owners[fourth] and owners[second] == owners[third]:
            spin_integrals[first, second, third, fourth] -= integrals[
                owners[first] + owners[second]
            ][local[first], local[fourth], local[second], local[third]]
    result = _feeds(
        layout, spin_fock, spin_integrals, layout.unpack(np.zeros(layout.packed_size))
    )
    result["reference_electronic_energy"] = np.array(
        0.5 * sum(np.sum(density[spin] * (hcore[spin] + fock[spin])) for spin in "ab")
    )
    return result, spin_fock, spin_integrals


def _direction(name: str, shape: tuple, rng: Any) -> np.ndarray:
    """Respect raw h/g physical symmetries; keep rotations general for overlap AD."""
    direction = rng.normal(scale=0.1, size=shape)
    if name.startswith("h"):
        direction += direction.T
    elif name.startswith("g"):
        direction = direction + direction.swapaxes(0, 1)
        direction = direction + direction.swapaxes(2, 3)
        if name in ("g_aa", "g_bb"):
            direction = direction + direction.transpose(2, 3, 0, 1)
    return direction


@pytest.mark.parametrize(
    "populations", [(2, 1, 2, 2), (2, 2, 2, 2), (3, 0, 0, 3), (0, 0, 0, 0)]
)
def test_native_raw_bindings_and_each_input_pullback(populations: tuple) -> None:
    layout = UCCSDLayout(*populations)
    programs, values = build_raw_hamiltonian(layout), _inputs(layout)
    executor = NativeCCTensorExecutor(max_bytes=128 << 20)
    actual = executor.execute(programs.primal, values)
    expected, _, _ = _reference(layout, values)
    for name, value in actual.items():
        np.testing.assert_allclose(value, expected[name], atol=2e-13, rtol=0)
    rng = np.random.default_rng(18243)
    seeds = {
        f"bar_{name}": rng.normal(scale=0.1, size=value.shape)
        for name, value in actual.items()
    }
    bars = executor.execute(programs.pullback.program, {**values, **seeds})
    for name, value in values.items():
        direction = _direction(name, value.shape, rng)
        contractions = []
        for sign in (-1, 1):
            outputs, _, _ = _reference(
                layout, {**values, name: value + sign * 1e-5 * direction}
            )
            contractions.append(
                sum(
                    np.sum(outputs[key.removeprefix("bar_")] * seed)
                    for key, seed in seeds.items()
                )
            )
        np.testing.assert_allclose(
            np.sum(bars[f"bar_{name}"] * direction),
            (contractions[1] - contractions[0]) / 2e-5,
            atol=5e-9,
            rtol=0,
        )


@pytest.mark.parametrize("populations", [(2, 1, 2, 2), (2, 2, 2, 2), (3, 0, 0, 3)])
def test_composed_physical_lagrangian_against_fermionic_oracle(
    populations: tuple,
) -> None:
    layout = UCCSDLayout(*populations)
    raw = build_raw_hamiltonian(layout)
    programs = build_lagrangian_programs(layout, raw)
    values, amplitudes = _inputs(layout), _amplitudes(layout)
    executor = NativeCCTensorExecutor(max_bytes=256 << 20)
    actual = executor.execute(programs.primal, {**values, **amplitudes})
    _, fock, integrals = _reference(layout, values)
    expected = _determinant_oracle(layout, fock, integrals, amplitudes)
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name], value, atol=2e-12, rtol=0)
    rng = np.random.default_rng(18244)
    dual = rng.normal(scale=0.1, size=layout.packed_size)
    seeds = {
        f"bar_{name.replace('t', 'r', 1)}": value
        for name, value in layout.unpack_dual(dual).items()
    }
    seeds.update(
        bar_correlation_energy=np.array(1.0),
        bar_reference_electronic_energy=np.array(0.7),
        bar_fov_aa=rng.normal(scale=0.1, size=(layout.oa, layout.va)),
        bar_fov_bb=rng.normal(scale=0.1, size=(layout.ob, layout.vb)),
    )
    bars = executor.execute(
        programs.pullback.program, {**values, **amplitudes, **seeds}
    )
    for name in RAW_INPUTS:
        direction = _direction(name, values[name].shape, rng)
        contractions = []
        for sign in (-1, 1):
            fields, fock, integrals = _reference(
                layout, {**values, name: values[name] + sign * 1e-5 * direction}
            )
            oracle = _determinant_oracle(layout, fock, integrals, amplitudes)
            contractions.append(
                float(oracle["correlation_energy"])
                + np.dot(dual, _physical_residual(layout, oracle))
                + 0.7 * float(fields["reference_electronic_energy"])
                + sum(
                    np.sum(fields[field] * seeds[f"bar_{field}"])
                    for field in ("fov_aa", "fov_bb")
                )
            )
        np.testing.assert_allclose(
            np.sum(bars[f"bar_{name}"] * direction),
            (contractions[1] - contractions[0]) / 2e-5,
            atol=5e-9,
            rtol=0,
        )
