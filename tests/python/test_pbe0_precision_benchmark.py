"""Host-only guards for precision benchmark semantics and acceptance gates."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import GridSpec

from benchmarks.pbe0_precision import (
    CASES,
    MODES,
    errors,
    protocol,
    reference,
    summarize,
)


@pytest.mark.parametrize(
    ("case_name", "cartesian"),
    (("water-def2-svp-spherical", False), ("water-def2-svp", True)),
)
def test_reference_preserves_basis_representation(
    case_name: str, cartesian: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A force oracle must not silently replace six Cartesian d AOs by five."""

    def molecule(**options: object) -> None:
        assert options["cart"] is cartesian
        raise RuntimeError("stop before SCF")

    monkeypatch.setitem(
        sys.modules,
        "pyscf",
        SimpleNamespace(
            dft=SimpleNamespace(),
            gto=SimpleNamespace(M=molecule),
            lib=SimpleNamespace(num_threads=lambda _: None),
        ),
    )
    with pytest.raises(RuntimeError, match="stop before SCF"):
        reference(case_name, GridSpec(), 1)


def test_protocol_fixes_guess_and_separates_science_from_precision() -> None:
    """A tighter audit changes solver controls, not its moving-grid target."""
    spec = GridSpec()
    scientific = protocol(CASES[0], spec)
    assert scientific["initial_guess"] == "hcore"
    assert scientific["method"] == "pbe0-rks"
    assert scientific["properties"] == ["energy", "forces"]
    assert scientific["density_fitting"] is False
    assert "precision" not in scientific
    assert scientific != protocol(CASES[0], replace(spec, radial_points=24))
    assert scientific != protocol(CASES[1], spec)


def test_force_error_norms_reject_broadcasts_and_nonfinite_results() -> None:
    """Component maxima and RMS cannot silently broadcast one atom to many."""
    oracle = {"energy": -1.0, "forces": [[0.0, 0.0, 0.0]]}
    observed = {"energy": -0.75, "forces": [[0.0, 3.0, 4.0]]}
    measured = errors(observed, oracle)
    assert measured["energy_Eh"] == 0.25
    assert measured["max_force_Eh_per_bohr"] == 4.0
    assert measured["rms_force_Eh_per_bohr"] == pytest.approx(5 / np.sqrt(3))
    with pytest.raises(ValueError, match="dimensions"):
        errors({**observed, "forces": [[0.0, 3.0, 4.0]] * 2}, oracle)
    with pytest.raises(ValueError, match="nonfinite"):
        errors({**observed, "energy": float("nan")}, oracle)


def test_summary_excludes_first_calls_and_preserves_audit_slowdowns() -> None:
    """Report complete-policy time rather than just its candidate solve."""
    records = [
        {"mode": mode, "phase": phase, "seconds": seconds, "oracle_gate": True}
        for mode, elapsed in zip(MODES, (10.0, 8.0, 20.0), strict=True)
        for phase, seconds in (("first-call-after-probe", 100.0), ("measured", elapsed))
    ]
    summary = summarize(records)
    assert summary["median_seconds"]["audit_auto"] == 20.0
    assert summary["speedup_vs_fp64"] == {"fp64": 1.0, "auto": 1.25, "audit_auto": 0.5}
    records[-1]["oracle_gate"] = False
    with pytest.raises(ValueError, match="rejected"):
        summarize(records)
