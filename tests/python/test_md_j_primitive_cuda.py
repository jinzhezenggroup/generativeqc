"""Complete primitive J has its own cumulative screening contract.

Independent unscreened libcint ERIs qualify signed/general contractions and
both density orientations. Graph replays change density magnitude in both
directions so a stale active prefix cannot pass through an ordinary warm test.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from _md_j_engine_oracle import WATER, general_contraction_basis, public_ao_oracle

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TEST_MD_J_ENGINE") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires MD qualification in a finite Slurm GPU allocation",
)


def _spec(spin: str, exchange: bool = False) -> Any:
    from generativeqc.fock import FockBuildSpec, FockTerm

    return replace(
        FockBuildSpec.hf(spin, derivative_order=0), exchange=FockTerm(exchange)
    )


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
@pytest.mark.parametrize("general", [False, True])
@pytest.mark.parametrize("cutoff", [0.0, 1e-12, 0.02])
def test_primitive_complete_j_and_changed_density_graph(
    monkeypatch: pytest.MonkeyPatch,
    representation: str,
    spin: str,
    general: bool,
    cutoff: float,
    record_property: Any,
) -> None:
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe

    # Exercise the ordinary unset selector against an independent unscreened
    # oracle; the capacity test below also covers explicit primitive selection.
    monkeypatch.delenv("GENERATIVEQC_MD_J_ENGINE", raising=False)
    monkeypatch.delenv("GENERATIVEQC_DISABLE_MD_J", raising=False)
    monkeypatch.delenv("GENERATIVEQC_MD_J_WORK_COUNTS", raising=False)
    atoms, shells = general_contraction_basis() if general else (WATER, "def2-svp")
    with NativeAO(atoms, shells, representation=representation) as basis:
        _, _, eri, _ = public_ao_oracle(basis)
        density = (
            np.random.default_rng(81013).normal(size=(basis.nao, basis.nao)) * 0.03
        )
        if spin == "unrestricted":
            density = np.stack((density, -0.6 * density.T))
        with ResidentMdJProbe(basis, _spec(spin), screening_tolerance=cutoff) as plan:
            assert plan.diagnostics["direct_schedule"].startswith("md-j-primitive/")
            plan.capture_graph()
            maximum = 0.0
            for scale in (1.0, 0.0, 1e-15, -3.0, 100.0, 1e-6):
                changed = density * scale
                total = changed.sum(axis=0) if changed.ndim == 3 else changed
                expected = np.einsum("ijkl,kl->ij", eri, total, optimize=True)
                actual = plan.evaluate_graph(changed).coulomb
                maximum = max(maximum, float(np.max(np.abs(actual - expected))))
                np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=3e-12)
                if scale == 0:
                    np.testing.assert_array_equal(actual, 0)
            record_property("max_complete_j_error", maximum)


@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
def test_primitive_source_cancellation_and_restoration(
    monkeypatch: pytest.MonkeyPatch, spin: str
) -> None:
    """Replay must refresh cancellation-sensitive bounds, even at similar |D|.

    Antisymmetric density and opposing spin densities cancel in the Coulomb
    source. Restoring the original density through the same captured owner
    must restore its work as well; a stale zero source bound would omit J.
    """
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe

    monkeypatch.delenv("GENERATIVEQC_MD_J_ENGINE", raising=False)
    monkeypatch.delenv("GENERATIVEQC_DISABLE_MD_J", raising=False)
    atoms, shells = general_contraction_basis()
    with NativeAO(atoms, shells, representation="spherical") as basis:
        _, _, eri, _ = public_ao_oracle(basis)
        d = np.random.default_rng(1743).normal(size=(basis.nao, basis.nao)) * 0.03
        if spin == "unrestricted":
            ordinary = np.stack((d, 0.6 * d.T))
            cancelled = np.stack((d, -d))
            nearly_cancelled = np.stack((d, -d + 1e-10 * d.T))
        else:
            ordinary = d
            cancelled = d - d.T
            nearly_cancelled = cancelled + 1e-10 * (d + d.T)
        with ResidentMdJProbe(basis, _spec(spin), screening_tolerance=1e-12) as plan:
            assert plan.diagnostics["direct_schedule"].startswith("md-j-primitive/")
            plan.capture_graph()
            for changed in (ordinary, cancelled, nearly_cancelled, ordinary):
                total = changed.sum(axis=0) if changed.ndim == 3 else changed
                expected = np.einsum("ijkl,kl->ij", eri, total, optimize=True)
                actual = plan.evaluate_graph(changed).coulomb
                np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=3e-12)


@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
@pytest.mark.parametrize("cutoff", [1e-12, 0.02])
def test_primitive_cumulative_omission_with_unequal_contractions(
    monkeypatch: pytest.MonkeyPatch, spin: str, cutoff: float
) -> None:
    """The absolute omission cap survives unequal target sizes and sparse replay.

    Scale independent unscreened libcint results around the omission cap, so
    ordinary FP64 accumulation error is negligible compared with the gate.
    Signed contractions of one, two and three primitives forbid substituting
    one convenient shell's multiplicity for the output's actual multiplicity.
    """
    from generativeqc import Primitive, Shell
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe

    monkeypatch.delenv("GENERATIVEQC_MD_J_ENGINE", raising=False)
    monkeypatch.delenv("GENERATIVEQC_DISABLE_MD_J", raising=False)
    atoms, shells = general_contraction_basis()
    shells = (
        *shells,
        Shell(
            0, 0, (Primitive(3.2, 0.41), Primitive(0.82, -0.18), Primitive(0.23, 0.6))
        ),
        Shell(1, 2, (Primitive(0.28, 0.7),)),
    )
    # The independent converter deliberately rejects implicit PySCF shell
    # reordering. Supply both engines the same explicit atom/angular order.
    shells = tuple(sorted(shells, key=lambda shell: (shell.atom_index, shell.angular_momentum)))
    cap = min(cutoff, 1e-12)
    with NativeAO(atoms, shells, representation="spherical") as basis:
        _, _, eri, _ = public_ao_oracle(basis)
        dense = np.random.default_rng(20261020).normal(size=(basis.nao, basis.nao))
        sparse = np.zeros_like(dense)
        sparse[-1, 0] = 1.0
        with ResidentMdJProbe(basis, _spec(spin), screening_tolerance=cutoff) as plan:
            assert plan.diagnostics["direct_schedule"].startswith("md-j-primitive/")
            plan.capture_graph()
            for raw in (dense, sparse, np.zeros_like(dense), dense):
                density = (
                    np.stack((raw, -0.6 * raw.T)) if spin == "unrestricted" else raw
                )
                total = density.sum(axis=0) if density.ndim == 3 else density
                oracle = np.einsum("ijkl,kl->ij", eri, total, optimize=True)
                norm = max(float(np.max(np.abs(oracle))), 1e-100)
                for factor in (0.01, 1.0, 20.0, 1e4):
                    scale = cap * factor / norm
                    actual = plan.evaluate_graph(density * scale).coulomb
                    np.testing.assert_allclose(
                        actual, oracle * scale, atol=cap * 1.0001, rtol=0
                    )
                    if not np.any(raw):
                        np.testing.assert_array_equal(actual, 0.0)


@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
def test_primitive_capacity_fallback_and_independent_k(
    monkeypatch: pytest.MonkeyPatch, spin: str
) -> None:
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe

    with NativeAO(WATER, "def2-svp", representation="spherical") as basis:
        d = np.random.default_rng(6735).normal(size=(basis.nao, basis.nao)) * 0.03
        if spin == "unrestricted":
            d = np.stack((d, -0.3 * d.T))
        monkeypatch.setenv("GENERATIVEQC_DISABLE_MD_J", "1")
        with ResidentMdJProbe(
            basis, _spec(spin, True), screening_tolerance=0
        ) as normal:
            expected = normal.evaluate(d)
            minimum = normal.diagnostics["device_bytes"]
        monkeypatch.delenv("GENERATIVEQC_DISABLE_MD_J")
        monkeypatch.setenv("GENERATIVEQC_MD_J_ENGINE", "incumbent")
        with ResidentMdJProbe(
            basis, _spec(spin, True), screening_tolerance=0
        ) as incumbent:
            incumbent_bytes = incumbent.diagnostics["device_bytes"]
        monkeypatch.setenv("GENERATIVEQC_MD_J_ENGINE", "primitive")
        for budget in (0, minimum, incumbent_bytes):
            with ResidentMdJProbe(
                basis,
                _spec(spin, True),
                screening_tolerance=0,
                device_budget_bytes=budget,
            ) as plan:
                schedule = plan.diagnostics["direct_schedule"]
                if budget == 0:
                    assert schedule.startswith("md-j-primitive/")
                elif budget == minimum:
                    assert not schedule.startswith("md-j-")
                else:
                    assert schedule.startswith("md-j-hermite/")
                actual = plan.evaluate(d)
                np.testing.assert_allclose(
                    actual.coulomb, expected.coulomb, atol=3e-11, rtol=3e-11
                )
                np.testing.assert_allclose(
                    actual.exchange, expected.exchange, atol=3e-11, rtol=3e-11
                )
