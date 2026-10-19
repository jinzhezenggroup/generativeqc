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
@pytest.mark.parametrize("backend", ("pyscf", "gpu4pyscf"))
def test_reference_preserves_basis_representation(
    case_name: str, cartesian: bool, backend: str, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setenv("SLURM_JOB_ID", "host-only-molecule-port")
    with pytest.raises(RuntimeError, match="stop before SCF"):
        reference(case_name, GridSpec(), 1, backend=backend)


def test_gpu_reference_requires_scheduling_and_matched_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unsupported requests before importing either numerical engine."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="reference backend"):
        reference(CASES[0], GridSpec(), 1, backend="unknown")
    with pytest.raises(RuntimeError, match="Slurm"):
        reference(CASES[0], GridSpec(), 1, backend="gpu4pyscf")
    with pytest.raises(NotImplementedError, match="matched GPU"):
        reference(CASES[0], GridSpec(), 1, functional="WB97M_V", auxiliary_basis={})


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


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf")])
def test_reference_rejects_invalid_vv10_thresholds(threshold: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        reference(CASES[0], GridSpec(), 1, vv10_density_threshold=threshold)


def test_cpu_reference_keeps_its_fixed_vv10_domain() -> None:
    with pytest.raises(NotImplementedError, match="require GPU4PySCF"):
        reference(CASES[0], GridSpec(), 1, vv10_density_threshold=1e-10)


@pytest.mark.parametrize("threshold", [None, 1e-10])
@pytest.mark.parametrize("fail", [None, "scf", "forces"])
def test_gpu_reference_scopes_both_vv10_legs_and_restores_defaults(
    threshold: float | None, fail: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real adapter and scoping helper without numerical engines."""
    expected = 1e-8 if threshold is None else threshold
    numint = SimpleNamespace(NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD=6e-9)
    grad_rks = SimpleNamespace(NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD=7e-9)
    observed = []

    def record(stage: str) -> None:
        observed.append(stage)
        assert numint.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD == expected
        assert grad_rks.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD == expected

    def kernel() -> float:
        record("scf")
        if fail == "scf":
            raise RuntimeError("synthetic scf failure")
        return -1.0

    def gradient_kernel() -> np.ndarray:
        record("forces")
        if fail == "forces":
            raise RuntimeError("synthetic forces failure")
        return np.zeros((3, 3))

    molecule = SimpleNamespace(natm=3, nao=24, atom_symbol=lambda i: ("O", "H", "H")[i])
    engine = SimpleNamespace(
        grids=SimpleNamespace(weights=np.ones(2)),
        nlcgrids=SimpleNamespace(weights=np.ones(2)),
        do_nlc=lambda: True,
        get_veff=lambda *a, **k: None,
        kernel=kernel,
        converged=True,
        cycles=1,
        nuc_grad_method=lambda: SimpleNamespace(kernel=gradient_kernel),
    )
    monkeypatch.setitem(
        sys.modules,
        "pyscf",
        SimpleNamespace(
            __version__="host-fixture",
            dft=SimpleNamespace(),
            gto=SimpleNamespace(M=lambda **k: molecule),
            lib=SimpleNamespace(num_threads=lambda _: None),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            asarray=np.asarray,
            asnumpy=np.asarray,
            cuda=SimpleNamespace(
                Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None))
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "gpu4pyscf", SimpleNamespace(__version__="host-fixture")
    )
    monkeypatch.setitem(
        sys.modules,
        "gpu4pyscf.dft",
        SimpleNamespace(
            gen_grid=SimpleNamespace(original_becke=object()),
            rks=SimpleNamespace(RKS=lambda _: engine),
            numint=numint,
        ),
    )
    monkeypatch.setitem(sys.modules, "gpu4pyscf.grad", SimpleNamespace(rks=grad_rks))
    monkeypatch.setenv("SLURM_JOB_ID", "host-only-adapter-test")
    options = {} if threshold is None else {"vv10_density_threshold": threshold}
    if fail:
        with pytest.raises(RuntimeError, match=f"synthetic {fail} failure"):
            reference(
                CASES[0],
                GridSpec(2, 2, 4),
                1,
                functional="WB97M-V",
                backend="gpu4pyscf",
                **options,
            )
        assert observed == (["scf"] if fail == "scf" else ["scf", "forces"])
    else:
        result = reference(
            CASES[0],
            GridSpec(2, 2, 4),
            1,
            functional="WB97M-V",
            backend="gpu4pyscf",
            **options,
        )
        assert result["vv10_density_threshold"] == {"scf": expected, "forces": expected}
        assert observed == ["scf", "forces"]
    assert numint.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD == 6e-9
    assert grad_rks.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD == 7e-9
