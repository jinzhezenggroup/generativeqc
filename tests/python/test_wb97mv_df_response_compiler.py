"""Independent full-rank fitted-primary derivatives, including metric motion."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.method.wb97mv_df_response import df_response_maps
from generativeqc_compiler.tensor import execute


def df_oracle(density: np.ndarray, values: np.ndarray, metric: np.ndarray) -> dict:
    """Expanded fitted ERIs and independently written source pullbacks.

    Only tests form the four-orbital tensor. Production programs must retain
    the factorized contraction schedule, including every AD validation node.
    """
    inverse = np.linalg.inv(metric)
    total = density.sum(axis=0)
    charge = np.einsum("ij,ijp->p", total, values)
    fitted = np.einsum("pq,q->p", inverse, charge)
    eri = np.einsum("ijp,pq,klq->ijkl", values, inverse, values)
    exchange = 0.15 / (4 if len(density) == 1 else 2)
    energy = 0.5 * np.einsum("ij,kl,ijkl->", total, total, eri)
    energy -= exchange * np.einsum("sik,sjl,ijkl->", density, density, eri)
    weight_inverse = 0.5 * np.outer(charge, charge)
    weight_inverse -= exchange * np.einsum(
        "sik,sjl,ijp,klq->pq", density, density, values, values
    )
    return {
        "energy": np.asarray(energy),
        "fock": np.einsum("kl,ijkl->ij", total, eri)[None]
        - 2 * exchange * np.einsum("skl,ikjl->sij", density, eri),
        "weight_three_center": total[:, :, None] * fitted
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
        metric_direction=result["d_metric"],
        bar_coulomb=np.asarray(1.0),
        bar_exchange=np.asarray(1.0),
        bar_energy=np.asarray(1.0),
    )
    return result


@pytest.mark.parametrize("spins", (1, 2))
def test_fitted_reference_and_moving_sources_against_expanded_oracle(
    spins: int,
) -> None:
    maps = df_response_maps()
    prefix = "rks" if spins == 1 else "uks"
    feeds = df_case(spins)
    inverse = execute(maps["inverse"], feeds).outputs["inverse"]
    np.testing.assert_allclose(inverse, feeds["inverse"], atol=2e-15)
    assert np.max(abs(inverse - feeds["whitening"] @ feeds["whitening"].T)) > 0.05
    feeds["d_inverse"] = execute(maps["inverse_direction"], feeds).outputs["direction"]
    reference = execute(maps[f"{prefix}_reference"], feeds).outputs
    oracle = df_oracle(feeds["density"], feeds["three_center"], feeds["metric"])
    for name, expected in oracle.items():
        np.testing.assert_allclose(reference[name], expected, atol=3e-14, rtol=2e-13)
    motion = execute(maps[f"{prefix}_response"], feeds).outputs
    for step in (1e-3, 2e-4):
        plus, minus, fixed_plus, fixed_minus = {}, {}, {}, {}
        for sign in (1, -1):
            shifted = {
                name: feeds[name] + sign * step * feeds[f"d_{name}"]
                for name in ("density", "three_center", "metric")
            }
            moving = df_oracle(
                shifted["density"], shifted["three_center"], shifted["metric"]
            )
            fixed = df_oracle(
                feeds["density"], shifted["three_center"], shifted["metric"]
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
    assert np.max(abs(full_fock_motion - motion["fock_direction"])) > 0.01
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
