"""Positive and adversarial issue-specific certificate checks (no native solve)."""

from __future__ import annotations

import copy
import hashlib
import json
import lzma
from pathlib import Path

import numpy as np
import pytest

from tools.oh_uhf_symmetry import (
    AO_LABELS,
    axial_rotation,
    certify_oh,
    fit_axial_angle,
    physical_state,
    raw_comparison,
)


def fixture() -> dict:
    # Synthetic algebra fixture; not molecular CPU acceptance evidence.
    s = np.eye(6)
    h = np.diag([-10.0, -5.0, -2.0, -2.0, -1.0, 1.0])
    g = np.zeros((6, 6, 6, 6))
    ref = np.array(
        [
            np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            np.diag([1.0, 1.0, 1.0, 0.0, 1.0, 0.0]),
        ]
    )
    u = axial_rotation(0.63)[1]
    d = u @ ref @ u.T
    xyz = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.834]]
    e = physical_state(ref, h, g, 8 / 1.834)[1]
    return {
        "coordinates": xyz,
        "ao_labels": AO_LABELS,
        "overlap": s,
        "hcore": h,
        "eri": g,
        "density": d,
        "reference": ref,
        "endpoint_energy": e,
        "reference_energy": e,
        "angle": fit_axial_angle(d, ref),
    }


def test_raw_fail_and_separate_spatial_certificate() -> None:
    f = fixture()
    before = f["density"].copy()
    result = certify_oh(**f)
    assert result["raw"]["status"] == "FAIL"
    assert result["symmetry"]["status"] == "PASS"
    assert np.array_equal(f["density"], before)
    assert result["production_density_modified"] is False
    assert "unresolved" in result["canonical_determinant_policy"]


def test_non_equivalent_integer_state_rejected() -> None:
    f = fixture()
    # Promote an occupied z electron to the H AO: same spin counts/idempotency,
    # stationary for this diagonal algebra fixture, physically different energy.
    f["density"][1, 4, 4] = 0
    f["density"][1, 5, 5] = 1
    result = certify_oh(**f)["symmetry"]
    assert result["status"] == "FAIL"
    assert result["errors"]["electron_count"] < 1e-9
    assert result["errors"]["idempotency"] < 1e-9
    assert "energy" in result["failed_gates"]
    assert "density" in result["failed_gates"]


@pytest.mark.parametrize("term", ["overlap", "hcore", "eri"])
def test_non_symmetric_hamiltonian_cannot_license_rotation(term: str) -> None:
    f = fixture()
    if term == "eri":
        f[term][2, 2, 2, 2] = 0.01
    else:
        f[term][2, 2] += 0.01
    result = certify_oh(**f)["symmetry"]
    assert result["status"] == "FAIL"
    assert f"{term}_invariance" in result["failed_gates"]


def test_separate_spin_rotations_are_not_spatial_equivalence() -> None:
    f = fixture()
    # Give both spins an anisotropic pi block, then rotate them differently.
    # Each spin is a valid stationary integer projector for the synthetic H,
    # but no *single* spatial transformation maps both to this reference.
    f["reference"][0] = np.diag([1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    for spin, angle in enumerate((0.27, 0.63)):
        u = axial_rotation(angle)[1]
        f["density"][spin] = u @ f["reference"][spin] @ u.T
    energy = physical_state(f["reference"], f["hcore"], f["eri"], 8 / 1.834)[1]
    f["endpoint_energy"] = f["reference_energy"] = energy
    f["angle"] = fit_axial_angle(f["density"], f["reference"])
    result = certify_oh(**f)
    assert result["symmetry"]["status"] == "FAIL"
    assert "density" in result["symmetry"]["failed_gates"]
    assert result["symmetry"]["errors"]["physical_residual"] < 1e-8


def test_wrong_angle_and_non_axial_geometry_rejected() -> None:
    f = fixture()
    f["angle"] = 0.11
    assert certify_oh(**f)["symmetry"]["status"] == "FAIL"
    f["coordinates"][1][0] = 0.001
    with pytest.raises(ValueError, match="only supports"):
        certify_oh(**f)


def test_nondegenerate_fixture_keeps_strict_raw_gate() -> None:
    s = np.eye(2)
    d = np.array([np.diag([1.0, 0.0])] * 2)
    u = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert raw_comparison(d, d, s)["status"] == "PASS"
    assert raw_comparison(u @ d @ u.T, d, s)["status"] == "FAIL"
    # A two-AO H2 state has no six-AO OH certificate; no equivalence fallback.
    f = fixture()
    f.update(density=d, reference=d, overlap=s, coordinates=[[0, 0, 0], [0, 0, 1.4]])
    with pytest.raises(ValueError, match="only supports"):
        certify_oh(**f)


def test_raw_gate_is_not_relaxed_for_small_errors() -> None:
    s = np.eye(2)
    d = np.zeros((2, 2, 2))
    changed = d.copy()
    changed[0, 0, 0] = 0.99e-7
    assert raw_comparison(changed, d, s)["status"] == "PASS"
    changed[0, 0, 0] = 1.01e-7
    assert raw_comparison(changed, d, s)["status"] == "FAIL"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_fails_closed(bad: float) -> None:
    f = fixture()
    f["density"][0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        certify_oh(**f)


def test_exact_ao_order_and_spd_overlap_required() -> None:
    f = fixture()
    f["ao_labels"] = tuple(reversed(AO_LABELS))
    with pytest.raises(ValueError, match="AO frame"):
        certify_oh(**f)
    f = fixture()
    f["overlap"][0, 0] = -1
    with pytest.raises(ValueError, match="positive definite"):
        certify_oh(**f)


def test_retained_historical_failures_are_not_promoted() -> None:
    root = Path(__file__).resolve().parents[2]
    capsule = root / "benchmarks/results/cpu-bounded-scalar-hf-20261003/capsule.json.xz"
    data = json.loads(lzma.decompress(capsule.read_bytes()))["components"]["molecular"]
    rows = [r for r in data["raw_failures"] if r["case"] == "oh6_exact_uhf"]
    assert len(rows) == 4
    assert {(r["role"], r["phase"]) for r in rows} == {
        (role, phase)
        for role in ("baseline", "candidate")
        for phase in ("cold_forces", "changed_geometry_forces")
    }
    for row in rows:
        assert set(row["failed_gates"]) == {"density_error", "metric_projector_error"}
        assert row["metrics"]["density_error"] > 1e-7
        assert row["metrics"]["metric_projector_error"] > 1e-7


def test_recorded_molecular_cpu_states() -> None:
    path = Path(__file__).parents[1] / "data/oh_uhf_symmetry_1791.json"
    evidence = json.loads(path.read_text())
    assert evidence["schema"] == "oh-uhf-symmetry-1791-v1"
    assert len(evidence["rows"]) == 4
    assert {(r["provider"], r["phase"]) for r in evidence["rows"]} == {
        (provider, phase)
        for provider in ("scalar", "openblas")
        for phase in ("original", "moved")
    }
    root = Path(__file__).resolve().parents[2]
    for provenance in evidence["providers"].values():
        for field, tool in (
            ("runner_sha256", "run_oh_uhf_symmetry.py"),
            ("certificate_sha256", "oh_uhf_symmetry.py"),
        ):
            # Git's canonical text is LF even in an autocrlf Windows checkout.
            canonical = (root / "tools" / tool).read_bytes().replace(b"\r\n", b"\n")
            assert provenance[field] == hashlib.sha256(canonical).hexdigest()
    capsule = root / "benchmarks/results/cpu-bounded-scalar-hf-20261003/capsule.json.xz"
    historical = json.loads(lzma.decompress(capsule.read_bytes()))["components"][
        "molecular"
    ]["raw_failures"]
    assert evidence["historical_raw_failures"] == [
        row for row in historical if row["case"] == "oh6_exact_uhf"
    ]
    h2 = evidence["nondegenerate_fixture"]
    assert h2["orbital_gap_hartree"] > 1e-3
    s = np.asarray(h2["overlap"])
    d = np.asarray(h2["density"])
    assert raw_comparison(d, d, s)["status"] == "PASS"
    w, v = np.linalg.eigh(s)
    root_s, invroot_s = (v * np.sqrt(w)) @ v.T, (v / np.sqrt(w)) @ v.T
    u = np.array([[0.0, -1.0], [1.0, 0.0]])
    other = invroot_s @ u @ root_s @ d @ root_s @ u.T @ invroot_s
    assert raw_comparison(other, d, s)["status"] == "FAIL"
    for row in evidence["rows"]:
        assert row["converged"] and row["oracle_converged"]
        # Public name of the native C++ CPU HF backend enum (not a PySCF solve).
        assert row["executed_backend"] == "cpu_reference"
        assert row["force_error"] <= row["force_gate"] == 1e-7
        f = copy.deepcopy(row["certificate_inputs"])
        assert f["coordinates"][1][2] == (
            1.834 if row["phase"] == "original" else 1.85234
        )
        result = certify_oh(**f)
        assert result["raw"]["status"] == row["diagnostic"]["raw"]["status"]
        for key, value in result["symmetry"]["errors"].items():
            assert value == pytest.approx(
                row["diagnostic"]["symmetry"]["errors"][key], abs=1e-12
            )
        assert result["symmetry"]["status"] == "PASS"
        assert result["raw"]["status"] == row["observed_raw_status"]
        assert result["raw"]["status"] == "FAIL"
        # Adversarial molecular tensor mutation; distinct from synthetic probes.
        f["hcore"][2][2] += 1e-3
        assert certify_oh(**f)["symmetry"]["status"] == "FAIL"

        # Molecular non-equivalent determinant: promote a low-energy beta
        # occupied direction to a high-energy virtual direction, retaining
        # counts/idempotency. Select within the metric-projector eigenspaces.
        f = copy.deepcopy(row["certificate_inputs"])
        s = np.asarray(f["overlap"])
        w, v = np.linalg.eigh(s)
        root_s = (v * np.sqrt(w)) @ v.T
        invroot_s = (v / np.sqrt(w)) @ v.T
        d = np.asarray(f["density"])
        occ, c = np.linalg.eigh(root_s @ d[1] @ root_s)
        fock = physical_state(
            d, np.asarray(f["hcore"]), np.asarray(f["eri"]), 8 / f["coordinates"][1][2]
        )[0][1]
        expectations = np.diag(c.T @ invroot_s @ fock @ invroot_s @ c)
        occupied_indices = np.flatnonzero(occ > 0.5)
        virtual_indices = np.flatnonzero(occ < 0.5)
        occupied = c[:, occupied_indices[np.argmin(expectations[occupied_indices])]]
        virtual = c[:, virtual_indices[np.argmax(expectations[virtual_indices])]]
        d[1] += (
            invroot_s
            @ (np.outer(virtual, virtual) - np.outer(occupied, occupied))
            @ invroot_s
        )
        f["density"] = d
        rejected = certify_oh(**f)["symmetry"]
        assert rejected["status"] == "FAIL"
        assert rejected["errors"]["electron_count"] < 1e-9
        assert rejected["errors"]["idempotency"] < 1e-9
        assert "density" in rejected["failed_gates"]
        assert (
            "energy" in rejected["failed_gates"]
            or "physical_residual" in rejected["failed_gates"]
        )
