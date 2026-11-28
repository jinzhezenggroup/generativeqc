"""Independent full-rank fitted-primary derivatives, including metric motion."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.method.stationary_df_second import df_second_programs
from generativeqc_compiler.method.wb97mv_df_response import df_response_maps
from generativeqc_compiler.tensor import execute


def df_oracle(
    density: np.ndarray,
    values: np.ndarray,
    metric: np.ndarray,
    fraction: float = 0.15,
    *,
    coulomb: bool = True,
    inverse: np.ndarray | None = None,
) -> dict:
    """Expanded fitted ERIs and independently written source pullbacks.

    Only tests form the four-orbital tensor. Production programs must retain
    the factorized contraction schedule, including every AD validation node.
    """
    if inverse is None:
        inverse = np.linalg.inv(metric)
    total = density.sum(axis=0)
    charge = np.einsum("ij,ijp->p", total, values)
    fitted = np.einsum("pq,q->p", inverse, charge)
    eri = np.einsum("ijp,pq,klq->ijkl", values, inverse, values)
    exchange = fraction / (4 if len(density) == 1 else 2)
    energy = (0.5 if coulomb else 0) * np.einsum("ij,kl,ijkl->", total, total, eri)
    energy -= exchange * np.einsum("sik,sjl,ijkl->", density, density, eri)
    weight_inverse = (0.5 if coulomb else 0) * np.outer(charge, charge)
    weight_inverse -= exchange * np.einsum(
        "sik,sjl,ijp,klq->pq", density, density, values, values
    )
    return {
        "energy": np.asarray(energy),
        "fock": coulomb * np.einsum("kl,ijkl->ij", total, eri)[None]
        - 2 * exchange * np.einsum("skl,ikjl->sij", density, eri),
        "weight_three_center": coulomb * total[:, :, None] * fitted
        - 2
        * exchange
        * np.einsum("sik,sjl,klq,qp->ijp", density, density, values, inverse),
        "weight_metric": -inverse @ weight_inverse @ inverse,
    }


def df_case(spins: int, nao: int = 2, naux: int = 3) -> dict[str, np.ndarray]:
    """Signed spin densities and a deliberately nonsymmetric whitening gauge."""
    rng = np.random.default_rng(1631 + spins + 7 * nao + naux)
    density = rng.normal(size=(spins, nao, nao))
    density = (density + density.transpose(0, 2, 1)) / 2
    values = rng.normal(scale=0.3, size=(nao, nao, naux))
    values = (values + values.transpose(1, 0, 2)) / 2
    factor = rng.normal(size=(naux, naux))
    metric = factor @ factor.T + np.eye(naux)
    rotation, _ = np.linalg.qr(rng.normal(size=(naux, naux)))
    whitening = rotation @ np.linalg.cholesky(np.linalg.inv(metric)).T
    result = {"density": density, "three_center": values, "metric": metric}
    for name, value in tuple(result.items()):
        motion = rng.normal(scale=0.2, size=value.shape)
        if name == "three_center":
            motion = (motion + motion.transpose(1, 0, 2)) / 2
        else:
            motion = (motion + motion.swapaxes(-1, -2)) / 2
        result[f"d_{name}"] = motion
    result.update(
        inverse=np.linalg.inv(metric),
        whitening=whitening,
        bar_coulomb=np.asarray(1.0),
        bar_exchange=np.asarray(1.0),
        bar_energy=np.asarray(1.0),
        exchange_fraction=np.asarray(0.15),
    )
    return result


def test_spectral_frame_rotation_uses_scf_column_major_convention() -> None:
    """Independent NumPy products for a nonsymmetric orthogonal frame."""
    generator = np.random.default_rng(51873)
    frame, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    panel = generator.normal(size=(3, 3))
    symmetric = (panel + panel.T) / 2
    programs = df_response_maps()
    rotated = execute(
        programs["spectral_enter"], {"frame_transposed": frame.T, "panel": panel}
    ).outputs["rotated"]
    np.testing.assert_allclose(rotated, frame.T @ symmetric @ frame, atol=2e-15)
    recovered = execute(
        programs["spectral_leave"], {"frame_transposed": frame.T, "panel": rotated}
    ).outputs["rotated"]
    np.testing.assert_allclose(recovered, symmetric, atol=2e-15)


def test_packed_spectral_fit_motion_matches_full_panel() -> None:
    """Triangular rows have the same tangent as the full symmetric panel."""
    case = df_case(1)
    raw, motion = (case[name] for name in ("three_center", "d_three_center"))
    inverse = case["inverse"]
    inverse_motion = -inverse @ case["d_metric"] @ inverse
    pairs = np.tril_indices(raw.shape[0])
    result = execute(
        df_response_maps()["packed_spectral_fit_motion"],
        {
            "raw": raw[pairs],
            "raw_motion": motion[pairs],
            "inverse": inverse,
            "inverse_motion": inverse_motion,
        },
    ).outputs["motion"]
    step = 1e-22j
    fitted = (raw + step * motion) @ np.linalg.inv(
        case["metric"] + step * case["d_metric"]
    )
    np.testing.assert_allclose(result, fitted.imag[pairs] / step.imag, atol=2e-15)


@pytest.mark.parametrize("spins", (1, 2))
def test_fitted_charge_reference_and_response(spins: int) -> None:
    """Check charge-level schedules against expanded ERIs and complex-step AD."""
    maps = df_response_maps()
    prefix = "rks" if spins == 1 else "uks"
    case = df_case(spins)
    case["accepted_charge"] = execute(maps[f"{prefix}_fit_charge"], case).outputs[
        "fitted_charge"
    ]
    reference = execute(maps[f"{prefix}_reference_coulomb"], case).outputs
    response = execute(maps[f"{prefix}_response_coulomb"], case).outputs
    split = {
        **execute(maps[f"{prefix}_response_coulomb_fock"], case).outputs,
        **execute(maps[f"{prefix}_response_coulomb_weights"], case).outputs,
    }
    rhs = split.pop("frozen_rhs")
    charge_motion = split.pop("frozen_charge_motion")
    density = case["density"].sum(axis=0)
    expected_rhs = np.einsum("ij,ijp->p", density, case["d_three_center"]) - (
        case["d_metric"] @ case["accepted_charge"]
    )
    np.testing.assert_allclose(rhs, expected_rhs, atol=3e-14, rtol=2e-13)
    np.testing.assert_allclose(
        charge_motion,
        np.linalg.solve(case["metric"], expected_rhs),
        atol=3e-14,
        rtol=2e-13,
    )
    assert split.keys() == response.keys()
    for name, value in split.items():
        np.testing.assert_allclose(value, response[name], atol=3e-14, rtol=2e-13)
    expected = df_oracle(case["density"], case["three_center"], case["metric"], 0)
    for name, value in expected.items():
        np.testing.assert_allclose(reference[name], value, atol=3e-14, rtol=2e-13)
    step = 1e-18j
    values = case["three_center"] + step * case["d_three_center"]
    metric = case["metric"] + step * case["d_metric"]
    frozen = df_oracle(case["density"], values, metric, 0)
    moving = df_oracle(case["density"] + step * case["d_density"], values, metric, 0)
    np.testing.assert_allclose(
        response["fock_direction"],
        frozen["fock"].imag / step.imag,
        atol=3e-14,
        rtol=2e-13,
    )
    for field in ("three_center", "metric"):
        np.testing.assert_allclose(
            response[f"weight_direction_{field}"],
            moving[f"weight_{field}"].imag / step.imag,
            atol=3e-14,
            rtol=2e-13,
        )


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("fraction", (0, 0.15, 0.2, 0.25))
def test_parameterized_fitted_graph_and_coulomb_only(
    spins: int, fraction: float
) -> None:
    feeds = df_case(spins)
    energies, sources, frozen, fock = df_second_programs(
        2, 3, spins, None if fraction else Fraction(0)
    )
    if fraction:
        feeds["exchange_fraction"] = np.asarray(fraction)
    energy = sum(execute(energies, feeds).outputs.values())
    reference = df_oracle(
        feeds["density"], feeds["three_center"], feeds["metric"], fraction
    )
    np.testing.assert_allclose(energy, reference["energy"], atol=3e-14, rtol=2e-13)
    np.testing.assert_allclose(
        execute(fock, feeds).outputs["bar_density"],
        reference["fock"],
        atol=3e-14,
        rtol=2e-13,
    )
    for field in ("three_center", "metric"):
        value = sum(execute(fixed, feeds).outputs[field] for fixed, _ in sources)
        np.testing.assert_allclose(
            value, reference[f"weight_{field}"], atol=3e-14, rtol=2e-13
        )
    step = 2e-4
    plus, minus = (
        df_oracle(
            feeds["density"],
            feeds["three_center"] + shift * feeds["d_three_center"],
            feeds["metric"] + shift * feeds["d_metric"],
            fraction,
        )["fock"]
        for shift in (step, -step)
    )
    np.testing.assert_allclose(
        execute(frozen, feeds).outputs["d_bar_density"],
        (plus - minus) / (2 * step),
        atol=2e-7,
        rtol=3e-6,
    )


@pytest.mark.parametrize("spins", (1, 2))
def test_coulomb_metric_pullback_avoids_expanded_outer_product_cancellation(
    spins: int,
) -> None:
    """Compare an ill-conditioned full-rank metric to independent solves."""
    rng = np.random.default_rng(4563)
    count = 18
    frame, _ = np.linalg.qr(rng.normal(size=(count, count)))
    eigenvalues = np.geomspace(1e-6, 1.0, count)
    inverse = (frame / eigenvalues) @ frame.T
    inverse = 0.5 * (inverse + inverse.T)
    metric = (frame * eigenvalues) @ frame.T
    density = np.ones((spins, 1, 1)) / spins
    charge = metric @ rng.normal(size=count)
    values = charge.reshape(1, 1, count)
    delta_charge = metric @ rng.normal(size=count)
    _, sources, _, _ = df_second_programs(1, count, spins, Fraction(0))
    ((reference, response),) = sources
    feeds = {
        "density": density,
        "three_center": values,
        "inverse": inverse,
        "bar_coulomb": np.asarray(1.0),
        "d_three_center": delta_charge.reshape(values.shape),
        "d_density": np.zeros_like(density),
        "d_metric": np.zeros_like(metric),
    }
    fitted = np.linalg.solve(metric, charge)
    fitted_dot = np.linalg.solve(metric, delta_charge)
    weight = execute(reference, feeds).outputs["metric"]
    action = execute(response, feeds).outputs["d_metric"]
    np.testing.assert_allclose(
        weight, -0.5 * np.outer(fitted, fitted), atol=3e-10, rtol=0
    )
    np.testing.assert_allclose(
        action,
        -0.5 * (np.outer(fitted, fitted_dot) + np.outer(fitted_dot, fitted)),
        atol=3e-10,
        rtol=0,
    )
    np.testing.assert_allclose(weight, weight.T, atol=1e-12, rtol=0)
    np.testing.assert_allclose(action, action.T, atol=1e-12, rtol=0)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("coulomb", (True, False), ids=("hybrid", "exchange-only"))
def test_fitted_reference_and_moving_sources_against_expanded_oracle(
    spins: int,
    coulomb: bool,
) -> None:
    maps = df_response_maps()
    prefix = "rks" if spins == 1 else "uks"
    feeds = df_case(spins)
    inverse = execute(maps["inverse"], feeds).outputs["inverse"]
    np.testing.assert_allclose(inverse, feeds["inverse"], atol=2e-15)
    assert np.max(abs(inverse - feeds["whitening"] @ feeds["whitening"].T)) > 0.05
    fit = execute(maps[f"{prefix}_fit_exchange"], dict(feeds, inverse=inverse)).outputs
    feeds.update(accepted_fit=fit["fitted"], accepted_dressed=fit["half_dressed"])
    suffix = "" if coulomb else "_exchange"
    reference = execute(maps[f"{prefix}_reference{suffix}"], feeds).outputs
    oracle = df_oracle(
        feeds["density"], feeds["three_center"], feeds["metric"], coulomb=coulomb
    )
    for name, expected in oracle.items():
        np.testing.assert_allclose(reference[name], expected, atol=3e-14, rtol=2e-13)
    motion = execute(maps[f"{prefix}_response{suffix}"], feeds).outputs
    for step in (1e-3, 2e-4):
        plus, minus, fixed_plus, fixed_minus = {}, {}, {}, {}
        for sign in (1, -1):
            shifted = {
                name: feeds[name] + sign * step * feeds[f"d_{name}"]
                for name in ("density", "three_center", "metric")
            }
            moving = df_oracle(
                shifted["density"],
                shifted["three_center"],
                shifted["metric"],
                coulomb=coulomb,
            )
            fixed = df_oracle(
                feeds["density"],
                shifted["three_center"],
                shifted["metric"],
                coulomb=coulomb,
            )
            (plus if sign == 1 else minus).update(moving)
            (fixed_plus if sign == 1 else fixed_minus).update(fixed)
        for field in ("three_center", "metric"):
            finite = (plus[f"weight_{field}"] - minus[f"weight_{field}"]) / (2 * step)
            np.testing.assert_allclose(
                motion[f"weight_direction_{field}"], finite, atol=2e-7, rtol=3e-6
            )
        finite_fock = (fixed_plus["fock"] - fixed_minus["fock"]) / (2 * step)
        np.testing.assert_allclose(
            motion["fock_direction"], finite_fock, atol=2e-7, rtol=3e-6
        )
    full_fock_motion = (plus["fock"] - minus["fock"]) / (2 * step)
    induced = df_oracle(
        feeds["d_density"], feeds["three_center"], feeds["metric"], coulomb=coulomb
    )["fock"]
    assert np.max(abs(induced)) > 1e-4
    np.testing.assert_allclose(
        full_fock_motion - motion["fock_direction"], induced, atol=2e-7, rtol=3e-6
    )
    derivative = np.sum(reference["fock"] * feeds["d_density"])
    derivative += np.sum(reference["weight_three_center"] * feeds["d_three_center"])
    derivative += np.sum(reference["weight_metric"] * feeds["d_metric"])
    np.testing.assert_allclose(
        derivative,
        (plus["energy"] - minus["energy"]) / (2 * step),
        atol=3e-7,
        rtol=2e-6,
    )


def test_factorized_schedule_has_no_quartic_orbital_or_auxiliary_work() -> None:
    """Bound every emitted node, not only the live energy contraction."""
    for program in df_response_maps().values():
        for node in program.nodes:
            if node.op == "einsum":
                labels = {}
                for source, indices in zip(
                    node.inputs, node.attrs["labels"], strict=True
                ):
                    labels.update(zip(indices, source.spec.indices, strict=True))
                axes = labels.values()
            elif node.op == "reduce":
                axes = node.inputs[0].spec.indices
            else:
                axes = node.spec.indices
            work = Counter(axis.space.name for axis in axes)
            assert work["orbital"] <= 3
            assert work["auxiliary"] <= 3
            assert work["orbital"] + work["auxiliary"] <= 4
            storage = Counter(axis.space.name for axis in node.spec.indices)
            assert storage["orbital"] <= 2
            assert storage["orbital"] + storage["auxiliary"] <= 3


def test_df_aot_generation_needs_no_runtime_or_gpu(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = """
import runpy, sys
class Deny:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'generativeqc', 'cupy', 'torch', 'pyscf'}:
            raise RuntimeError('forbidden build-time runtime import: ' + fullname)
sys.meta_path.insert(0, Deny())
sys.argv = ['generate_wb97mv_df_response.py', '--output', sys.argv[1]]
runpy.run_path('tools/generate_wb97mv_df_response.py', run_name='__main__')
"""
    outputs = [tmp_path / name for name in ("first.cuh", "second.cuh")]
    for output in outputs:
        subprocess.run(
            [sys.executable, "-c", script, str(output)],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
