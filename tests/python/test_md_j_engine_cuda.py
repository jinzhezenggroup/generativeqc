"""Opt-in, finite-Slurm scientific gates for the selected resident MD-J engine.

Run via benchmarks/md_j_engine_acceptance.py --gpu. Collection without an
explicit allocation skips CUDA tests and cannot count as integration evidence.
The resident probe executes the native KS Fock seam, not the host Fock facade.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _md_j_engine_oracle import (
    WATER,
    general_contraction_basis,
    public_ao_oracle,
    screened_j,
    work_rows,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TEST_MD_J_ENGINE") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires explicit MD-engine qualification in a finite Slurm GPU allocation",
)
REFERENCE_SCHEDULE = "md-j-reference/"


@pytest.fixture(scope="module", autouse=True)
def _cached_resident_bridge() -> None:
    """The inherited bridge must never silently compile without its cache."""
    from generativeqc_compiler.common.compiler_cache import resolve_compiler_cache

    resolve_compiler_cache(timeout=15)


def _spec(spin: str, exchange: bool = False) -> Any:
    from generativeqc.fock import FockBuildSpec, FockTerm

    return replace(
        FockBuildSpec.hf(spin, derivative_order=0), exchange=FockTerm(exchange)
    )


def _select(monkeypatch: Any, engine: str) -> None:
    monkeypatch.setenv("GENERATIVEQC_MD_J_ENGINE", engine)
    monkeypatch.delenv("GENERATIVEQC_DISABLE_MD_J", raising=False)


def _assert_selected(plan: Any, engine: str = "reference") -> None:
    schedule = plan.diagnostics["direct_schedule"]
    assert schedule.startswith(
        REFERENCE_SCHEDULE if engine == "reference" else "md-j-hermite/"
    ), schedule


def test_reference_r2scan3c_large_owner_admission(
    monkeypatch: Any, capfd: Any, record_property: Any
) -> None:
    """A budgeted 96-atom owner admits the hierarchy above the old fixed cap.

    Compare a nonsymmetric one-density-tail J against the exact-mask incumbent;
    full E/F acceptance supplies the independent large-system scientific oracle.
    The explicit budget belongs to this standalone probe, not a public ledger.
    """
    from generativeqc import Atom
    from generativeqc.basis import load_basis
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe

    from benchmarks.md_j_engine_benchmark import MolecularCase

    atoms = [
        Atom.from_value(atom)
        for atom in MolecularCase("r2scan3c-budget", "def2-mtzvpp", 1088, 2609).atoms()
    ]
    basis_file = (
        Path(__file__).resolve().parents[2]
        / "python/generativeqc/data/r2scan3c/def2-mtzvpp-h-ar.json"
    )
    shells = load_basis(basis_file).shells_for(atoms)
    budget = 1 << 30
    monkeypatch.setenv("GENERATIVEQC_MD_J_WORK_COUNTS", "1")
    results = {}
    with NativeAO(atoms, shells, representation="spherical") as basis:
        assert basis.nao == 1088
        density = np.zeros((basis.nao, basis.nao))
        density[-1, -2] = 0.83
        for engine in ("incumbent", "reference"):
            _select(monkeypatch, engine)
            with ResidentMdJProbe(
                basis,
                _spec("restricted"),
                screening_tolerance=1e-12,
                device_budget_bytes=budget,
            ) as plan:
                _assert_selected(plan, engine)
                assert plan.diagnostics["device_bytes"] <= budget
                results[engine] = plan.evaluate(density).coulomb.copy()
                np.testing.assert_array_equal(
                    plan.evaluate(np.zeros_like(density)).coulomb, 0.0
                )
                record_property(
                    f"{engine}_device_bytes", plan.diagnostics["device_bytes"]
                )
            log = capfd.readouterr().err
            assert "MD_J_WORK " in log
            if engine == "reference":
                assert "MD_J_ENGINE_WORK " in log
    np.testing.assert_allclose(
        results["reference"], results["incumbent"], atol=3e-11, rtol=3e-11
    )
    record_property("owner_budget_bytes", budget)


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
@pytest.mark.parametrize("basis_kind", ["def2-svp", "general-contractions"])
@pytest.mark.parametrize("cutoff", [0.0, 1e-12, 0.02])
def test_reference_raw_j_public_ao_oracle(
    representation: str,
    spin: str,
    basis_kind: str,
    cutoff: float,
    monkeypatch: Any,
    capfd: Any,
    record_property: Any,
) -> None:
    """Independent full-AO mask, nonsymmetric spins and changed-density replay."""
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe

    pytest.importorskip("pyscf")
    atoms, shells = (
        general_contraction_basis()
        if basis_kind == "general-contractions"
        else (WATER, "def2-svp")
    )
    _select(monkeypatch, "reference")
    monkeypatch.setenv("GENERATIVEQC_MD_J_WORK_COUNTS", "1")
    with NativeAO(atoms, shells, representation=representation) as basis:
        _, _, eri, bounds = public_ao_oracle(basis)
        rng = np.random.default_rng(90731)
        density = rng.normal(size=(basis.nao, basis.nao)) * 0.04
        if spin == "unrestricted":
            density = np.stack((density, rng.normal(size=density.shape) * 0.03))
        expected = screened_j(eri, bounds, density, cutoff)
        with ResidentMdJProbe(basis, _spec(spin), screening_tolerance=cutoff) as plan:
            _assert_selected(plan)
            capfd.readouterr()
            result = plan.evaluate(density).coulomb
            measured_seconds = plan.last_device_seconds
            rows = work_rows(capfd.readouterr().err)
            assert rows and sum(row["uniform_roots"] for row in rows) > 0
            assert {(row["bra"], row["ket"]) for row in rows} == {
                (a, b) for a in range(5) for b in range(5)
            }
            np.testing.assert_allclose(result, expected, atol=3e-11, rtol=3e-11)
            # A constructed owner freezes its engine; toggling the environment
            # must not mutate a retained plan or authorize stale density reuse.
            monkeypatch.setenv("GENERATIVEQC_MD_J_ENGINE", "incumbent")
            changed = np.swapaxes(density, -1, -2) * -0.7
            np.testing.assert_allclose(
                plan.evaluate(changed).coulomb,
                screened_j(eri, bounds, changed, cutoff),
                atol=3e-11,
                rtol=3e-11,
            )
            _assert_selected(plan)
            capfd.readouterr()
            np.testing.assert_allclose(
                plan.evaluate(np.zeros_like(density)).coulomb, 0.0, atol=1e-14, rtol=0
            )
            zero = work_rows(capfd.readouterr().err)
            assert zero and all(
                row["uniform_roots"] == row["residual_roots"] == 0 for row in zero
            )
            record_property("selected_schedule", plan.diagnostics["direct_schedule"])
            record_property("execution_work", json.dumps(rows, sort_keys=True))
            record_property("initial_device_seconds", measured_seconds)
            record_property("maximum_j_error", float(np.max(np.abs(result - expected))))
            record_property(
                "timing_scope",
                "diagnostic resident J only; excludes preparation, copies and full endpoint",
            )


@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
@pytest.mark.parametrize(
    "orientation",
    ["same-shell-diagonal", "same-shell-offdiagonal", "off-shell", "one-density-tail"],
)
def test_reference_pair_factors_and_one_density_tail(
    spin: str, orientation: str, monkeypatch: Any
) -> None:
    """Isolate same-shell, off-shell and final partial-tile density consumers."""
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe

    pytest.importorskip("pyscf")
    _select(monkeypatch, "reference")
    atoms, shells = general_contraction_basis()
    if orientation == "one-density-tail":
        from generativeqc import Primitive, Shell

        atoms = [("He", (0.0, 0.0, 0.0))]
        shells = tuple(
            Shell(0, angular, (Primitive(0.6 + 0.2 * angular, 1.0),))
            for angular in range(3)
        )
        # All but the angular-two class now have exactly one primitive pair:
        # the last admitted leaf occupies a genuinely one-element tile tail.
    with NativeAO(atoms, shells, representation="spherical") as basis:
        molecule, _, eri, bounds = public_ao_oracle(basis)
        offsets = molecule.ao_loc_nr()
        p_shell = next(i for i in range(molecule.nbas) if molecule.bas_angular(i) == 1)
        first = int(offsets[p_shell])
        indices = {
            "same-shell-diagonal": (first, first),
            "same-shell-offdiagonal": (first, first + 1),
            "off-shell": (0, first),
            "one-density-tail": (basis.nao - 1, basis.nao - 2),
        }
        density = np.zeros((basis.nao, basis.nao))
        density[indices[orientation]] = 0.83
        if spin == "unrestricted":
            density = np.stack((density, -0.3 * density.T))
        with ResidentMdJProbe(basis, _spec(spin), screening_tolerance=0) as plan:
            _assert_selected(plan)
            np.testing.assert_allclose(
                plan.evaluate(density).coulomb,
                screened_j(eri, bounds, density, 0),
                atol=3e-11,
                rtol=3e-11,
            )
            if spin == "unrestricted":
                cancelling = np.stack((density[0], -density[0]))
                np.testing.assert_allclose(
                    plan.evaluate(cancelling).coulomb, 0.0, atol=3e-12, rtol=0
                )


def test_reference_residual_mask_and_molecular_threshold_neighbors(
    monkeypatch: Any, capfd: Any, record_property: Any
) -> None:
    """Compare both native MD engines at neighbors of a libcint-derived cutoff.

    These are not claimed to be exact native equality boundaries: native and
    libcint Schwarz evaluations can differ by ulps. Synthetic direction and
    hierarchy tests separately qualify exact equality. This molecular test
    checks identical native J values at three neighboring inputs, plus an
    independent complete libcint-mask comparison at the separated 0.02 cutoff.
    """
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe

    pytest.importorskip("pyscf")
    monkeypatch.setenv("GENERATIVEQC_MD_J_WORK_COUNTS", "1")
    with NativeAO(WATER, "def2-svp", representation="spherical") as basis:
        molecule, _, eri, bounds = public_ao_oracle(basis)
        density = np.random.default_rng(547).normal(size=(basis.nao, basis.nao)) * 0.04
        offsets = molecule.ao_loc_nr()
        blocks = [
            bounds[offsets[a] : offsets[a + 1], offsets[b] : offsets[b + 1]]
            for a in range(molecule.nbas)
            for b in range(a + 1)
        ]
        assert any(
            a.min() * b.min() < 0.02 < a.max() * b.max() for a in blocks for b in blocks
        )
        products = np.sort(
            np.unique((bounds[:, :, None, None] * bounds[None, None, :, :]).ravel())
        )
        boundary = float(products[np.argmin(abs(products - 0.02))])
        thresholds = [
            0.02,
            np.nextafter(boundary, 0.0),
            boundary,
            np.nextafter(boundary, np.inf),
        ]
        for cutoff in thresholds:
            # The disabled-MD provider has a different nonzero-screening
            # contract. Only the incumbent MD path shares this exact AO mask.
            _select(monkeypatch, "incumbent")
            with ResidentMdJProbe(
                basis, _spec("restricted"), screening_tolerance=cutoff
            ) as baseline:
                _assert_selected(baseline, "incumbent")
                expected = baseline.evaluate(density).coulomb
            _select(monkeypatch, "reference")
            with ResidentMdJProbe(
                basis, _spec("restricted"), screening_tolerance=cutoff
            ) as plan:
                _assert_selected(plan)
                capfd.readouterr()
                actual = plan.evaluate(density).coulomb
                rows = work_rows(capfd.readouterr().err)
                assert sum(row["residual_roots"] for row in rows) > 0
                np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-11)
                if cutoff == 0.02:
                    np.testing.assert_allclose(
                        actual,
                        screened_j(eri, bounds, density, cutoff),
                        atol=3e-11,
                        rtol=3e-11,
                    )
                record_property(
                    f"residual_work_{cutoff.hex()}", json.dumps(rows, sort_keys=True)
                )


@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
def test_reference_incumbent_k_and_tiny_budget_fallback(
    spin: str, monkeypatch: Any
) -> None:
    """A refusal of optional MD storage must retain complete J and untouched K."""
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe

    with NativeAO(WATER, "def2-svp", representation="spherical") as basis:
        density = np.random.default_rng(721).normal(size=(basis.nao, basis.nao)) * 0.04
        if spin == "unrestricted":
            density = np.stack((density, -0.3 * density.T))
        monkeypatch.setenv("GENERATIVEQC_DISABLE_MD_J", "1")
        with ResidentMdJProbe(basis, _spec(spin, True)) as normal:
            expected = normal.evaluate(density)
            minimum = normal.diagnostics["device_bytes"]
        for engine in ("incumbent", "reference"):
            _select(monkeypatch, engine)
            for budget in (0, minimum):
                with ResidentMdJProbe(
                    basis, _spec(spin, True), device_budget_bytes=budget
                ) as plan:
                    if budget:
                        assert not plan.diagnostics["direct_schedule"].startswith(
                            "md-j-"
                        )
                    else:
                        _assert_selected(plan, engine)
                    observed = plan.evaluate(density)
                    np.testing.assert_allclose(
                        observed.coulomb, expected.coulomb, atol=3e-11, rtol=3e-11
                    )
                    np.testing.assert_allclose(
                        observed.exchange, expected.exchange, atol=3e-11, rtol=3e-11
                    )


def test_unset_selector_selects_primitive(monkeypatch: Any) -> None:
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe

    monkeypatch.delenv("GENERATIVEQC_MD_J_ENGINE", raising=False)
    monkeypatch.delenv("GENERATIVEQC_DISABLE_MD_J", raising=False)
    with (
        NativeAO(WATER, "def2-svp", representation="spherical") as basis,
        ResidentMdJProbe(basis, _spec("restricted")) as plan,
    ):
        assert plan.diagnostics["direct_schedule"].startswith("md-j-primitive/")


@pytest.mark.parametrize("method", ["rhf", "uhf", "pbe0-rks", "pbe0-uks"])
@pytest.mark.parametrize("policy", ["strict", "fast"])
@pytest.mark.parametrize("engine", ["reference", "default"])
def test_reference_complete_energy_force_endpoint(
    method: str,
    policy: str,
    engine: str,
    monkeypatch: Any,
    capfd: Any,
    record_property: Any,
) -> None:
    """Matched whole-endpoint parity with disabled MD, with execution provenance."""
    from generativeqc import Calculator, GridSpec, KsOptions, PrecisionCalculator

    monkeypatch.setenv("GENERATIVEQC_MD_J_COUNTS", "1")
    monkeypatch.setenv("GENERATIVEQC_MD_J_WORK_COUNTS", "1")
    options = {
        "method": method,
        "basis": "def2-svp",
        "basis_representation": "spherical",
        "device": "cuda",
        "initial_guess": None,
        "energy_tolerance": 1e-11,
        "density_tolerance": 1e-9,
        "screening_tolerance": 1e-12,
        "max_iterations": 180,
    }
    if "ks" in method:
        options["ks_options"] = KsOptions(grid=GridSpec(24, 8, 16))
    # OH is a genuinely open-shell doublet; closed-shell UHF alone does not
    # qualify spin-separated density consumers.
    atoms = WATER[:2] if method in ("uhf", "pbe0-uks") else WATER
    multiplicity = 2 if len(atoms) == 2 else 1
    outputs = {}
    for mode in ("disabled", engine):
        if mode == "disabled":
            monkeypatch.setenv("GENERATIVEQC_DISABLE_MD_J", "1")
            monkeypatch.setenv("GENERATIVEQC_CUDA_XC_SPATIAL_ORDER", "0")
        else:
            monkeypatch.delenv("GENERATIVEQC_CUDA_XC_SPATIAL_ORDER", raising=False)
            if mode == "default":
                monkeypatch.delenv("GENERATIVEQC_MD_J_ENGINE", raising=False)
                monkeypatch.delenv("GENERATIVEQC_DISABLE_MD_J", raising=False)
            else:
                _select(monkeypatch, engine)
        capfd.readouterr()
        started = time.perf_counter()
        calculator = (
            PrecisionCalculator(policy="fast", **options)
            if policy == "fast"
            else Calculator(precision="fp64", **options)
        )
        try:
            if method == "pbe0-rks":
                # RKS singlepoint closes a non-retained owner and may bypass
                # resident MD. Exercise the complete retained-owner route used
                # by full-size acceptance, retaining the execution assertion.
                with calculator.prepare_batch(
                    [atoms], multiplicities=[multiplicity], warm_start=True
                ) as owner:
                    outputs[mode] = owner.execute(
                        properties=("energy", "forces"), strict=True
                    ).items[0]
            else:
                outputs[mode] = calculator.singlepoint(
                    atoms, multiplicity=multiplicity, properties=("energy", "forces")
                )
        finally:
            calculator.clear_cache()
        seconds = time.perf_counter() - started
        stderr = capfd.readouterr().err
        counts = [
            int(line.split("=", 1)[1])
            for line in stderr.splitlines()
            if line.startswith("MD_J_NORMAL_COUNTS=")
        ]
        rows = work_rows(stderr)
        if mode == "disabled":
            assert not sum(counts) and not rows
        # Retain the actual observation: endpoint equality alone cannot prove
        # that an endpoint routes through resident MD. The KS endpoint must.
        if mode != "disabled" and "ks" in method:
            assert sum(counts) > 0 and rows, "KS endpoint bypassed resident MD-J"
        record_property(
            f"{mode}_endpoint",
            json.dumps(
                {
                    "seconds": seconds,
                    "md_j_calls": sum(counts),
                    "uniform_roots": sum(row["uniform_roots"] for row in rows),
                    "residual_roots": sum(row["residual_roots"] for row in rows),
                    "energy": outputs[mode].energy,
                    "iterations": outputs[mode].iterations,
                    "backend": outputs[mode].executed_backend,
                    "precision_policy": policy,
                    "precision_validation": getattr(
                        outputs[mode].precision_validation, "status", None
                    ),
                    "mixed_fock_builds": (outputs[mode].precision or {}).get(
                        "mixed_stage_fock_builds"
                    ),
                    "strict_fock_builds": (outputs[mode].precision or {}).get(
                        "strict_stage_fock_builds"
                    ),
                    "timing_scope": "fresh calculator + complete energy/force endpoint + owner teardown/clear_cache; execution counters enabled",
                },
                sort_keys=True,
            ),
        )
    expected, actual = outputs["disabled"], outputs[engine]
    assert expected.converged and actual.converged
    assert expected.executed_backend == actual.executed_backend == "cuda"
    energy_gate, force_gate = (3e-9, 3e-8) if policy == "strict" else (1e-7, 1e-5)
    assert abs(actual.energy - expected.energy) <= energy_gate
    np.testing.assert_allclose(actual.forces, expected.forces, atol=force_gate, rtol=0)
    delta = np.asarray(actual.forces) - np.asarray(expected.forces)
    record_property(
        "endpoint_errors",
        json.dumps(
            {
                "energy_absolute": abs(actual.energy - expected.energy),
                "force_maximum": float(np.max(np.abs(delta))),
                "force_rms": float(np.sqrt(np.mean(delta * delta))),
                "energy_gate": energy_gate,
                "force_maximum_gate": force_gate,
            },
            sort_keys=True,
        ),
    )


def test_reference_pinned_gpu4pyscf_differential(
    monkeypatch: Any, record_property: Any
) -> None:
    """Call the pinned GPU4PySCF MD J engine, never its generic J/K provider."""
    if not os.environ.get("GENERATIVEQC_MD_J_UPSTREAM_ROOT"):
        pytest.skip("optional pinned upstream checkout was not supplied")
    import cupy as cp
    import gpu4pyscf
    from generativeqc_compiler.dft import NativeAO
    from gpu4pyscf.scf import j_engine
    from md_j_resident_probe import ResidentMdJProbe

    from benchmarks.md_j_engine_acceptance import verify_upstream

    root = Path(os.environ["GENERATIVEQC_MD_J_UPSTREAM_ROOT"]).resolve()
    commit = verify_upstream(root)
    assert Path(gpu4pyscf.__file__).resolve().is_relative_to(root), (
        "imported GPU4PySCF differs from pinned checkout"
    )
    _select(monkeypatch, "reference")
    with NativeAO(WATER, "def2-svp", representation="spherical") as basis:
        molecule, scales, _, _ = public_ao_oracle(basis)
        density = np.random.default_rng(9102).normal(size=(basis.nao, basis.nao)) * 0.04
        transformed = density * scales[:, None] * scales[None, :]
        upstream_cutoff = 1e-30
        upstream_owner = j_engine._VHFOpt(
            molecule, direct_scf_tol=upstream_cutoff
        ).build()
        upstream = j_engine.get_j(
            molecule, cp.asarray(transformed), hermi=0, vhfopt=upstream_owner
        ).get()
        expected = upstream * scales[:, None] * scales[None, :]
        with ResidentMdJProbe(
            basis, _spec("restricted"), screening_tolerance=0.0
        ) as plan:
            _assert_selected(plan)
            np.testing.assert_allclose(
                plan.evaluate(density).coulomb, expected, atol=3e-10, rtol=3e-10
            )
    binary = Path(j_engine.libvhf_md._name).resolve()
    assert binary.is_file(), "upstream MD shared-library identity is unavailable"
    import hashlib

    record_property("upstream_library", str(binary))
    record_property(
        "upstream_library_sha256", hashlib.sha256(binary.read_bytes()).hexdigest()
    )
    record_property("upstream_screening_tolerance", upstream_cutoff)
    record_property("native_screening_tolerance", 0.0)
    record_property(
        "screening_scope",
        "near-unscreened differential; upstream requires a positive logarithmic cutoff; not a matched public-AO screening-mask gate",
    )
    record_property("upstream_commit", commit)
    record_property("upstream_entrypoint", "gpu4pyscf.scf.j_engine.get_j(hermi=0)")


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
@pytest.mark.parametrize("basis_kind", ["general-contractions", "one-primitive-tail"])
@pytest.mark.parametrize("cutoff", [0.0, 0.02])
def test_reference_graph_capture_changed_density(
    representation: str,
    spin: str,
    basis_kind: str,
    cutoff: float,
    monkeypatch: Any,
    record_property: Any,
) -> None:
    """Replay actual captured native resident J with fresh hierarchy/input data.

    Capture contains kernels, memsets and verified device copies. Density upload,
    final result readback, and the publishing owner-stream fence are outside the captured
    region. This is a scientific/capture gate, not an end-to-end timing claim.
    """
    import cupy as cp
    from generativeqc import Primitive, Shell
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe

    pytest.importorskip("pyscf")
    _select(monkeypatch, "reference")
    monkeypatch.delenv("GENERATIVEQC_MD_J_WORK_COUNTS", raising=False)
    atoms, shells = general_contraction_basis()
    if basis_kind == "one-primitive-tail":
        atoms = [("He", (0.0, 0.0, 0.0))]
        shells = tuple(
            Shell(0, angular, (Primitive(0.6 + 0.2 * angular, 1.0),))
            for angular in range(3)
        )
    with NativeAO(atoms, shells, representation=representation) as basis:
        _, _, eri, bounds = public_ao_oracle(basis)
        density = np.random.default_rng(2609).normal(size=(basis.nao, basis.nao)) * 0.04
        if spin == "unrestricted":
            density = np.stack((density, -0.31 * density.T))
        tail = np.zeros_like(density)
        tail[..., -1, -2] = 0.83
        cases = [
            ("changed", -0.7 * np.swapaxes(density, -1, -2)),
            ("zero", np.zeros_like(density)),
            ("one-density-tail", tail),
            ("original-after-tail", density),
        ]
        if spin == "unrestricted":
            cases.append(("spin-cancellation", np.stack((density[0], -density[0]))))
        with ResidentMdJProbe(basis, _spec(spin), screening_tolerance=cutoff) as plan:
            _assert_selected(plan)
            plan.evaluate(density)  # Finish lazy runtime setup outside capture.
            with pytest.raises(RuntimeError, match="has not been captured"):
                plan.evaluate_graph(density)
            plan.capture_graph()
            assert (
                plan.diagnostics["graph_nodes"]
                >= plan.diagnostics["graph_kernel_nodes"]
                > 0
            )
            errors = {}
            # Also exercise ordering from a nondefault upload stream. Native
            # capture/replay remains on the retained Fock owner's own stream.
            with cp.cuda.Stream(non_blocking=True):
                for label, changed in cases:
                    observed = plan.evaluate_graph(changed).coulomb
                    ordinary = plan.evaluate(changed).coulomb
                    expected = screened_j(eri, bounds, changed, cutoff)
                    np.testing.assert_allclose(
                        observed, ordinary, atol=3e-11, rtol=3e-11
                    )
                    np.testing.assert_allclose(
                        observed, expected, atol=3e-11, rtol=3e-11
                    )
                    if label in ("zero", "spin-cancellation"):
                        np.testing.assert_allclose(observed, 0.0, atol=3e-12, rtol=0)
                    errors[label] = {
                        "graph_vs_ordinary": float(np.max(np.abs(observed - ordinary))),
                        "graph_vs_oracle": float(np.max(np.abs(observed - expected))),
                    }
            # A rejected native request must end capture while preserving the
            # previous executable and leaving ordinary execution available.
            invalid = list(plan._buffer_arguments())
            invalid[3] = basis.nao**2 - 1
            assert plan._probe.md_j_probe_capture(*invalid) != 0
            assert "dimensions" in plan._probe.md_j_probe_error().decode()
            np.testing.assert_allclose(
                plan.evaluate_graph(cases[0][1]).coulomb,
                plan.evaluate(cases[0][1]).coulomb,
                atol=3e-11,
                rtol=3e-11,
            )
            # Replacing a graph cannot release borrowed owner/buffer state or
            # leave the owner stream in capture mode.
            plan.capture_graph()
            np.testing.assert_allclose(
                plan.evaluate_graph(cases[0][1]).coulomb,
                screened_j(eri, bounds, cases[0][1], cutoff),
                atol=3e-11,
                rtol=3e-11,
            )
            record_property("selected_schedule", plan.diagnostics["direct_schedule"])
            record_property("graph_nodes", plan.diagnostics["graph_nodes"])
            record_property(
                "graph_kernel_nodes", plan.diagnostics["graph_kernel_nodes"]
            )
            record_property(
                "graph_node_policy", "kernels, memsets and verified device copies only"
            )
            record_property("graph_replay_errors", json.dumps(errors, sort_keys=True))
            record_property(
                "capture_scope", "native resident enqueue on prepared owner stream"
            )
        with pytest.raises(RuntimeError, match="closed"):
            plan.evaluate_graph(density)


def test_reference_graph_capture_diagnostics_fail_closed(monkeypatch: Any) -> None:
    """Reject opt-in host polling before capture and preserve ordinary execution."""
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe

    _select(monkeypatch, "reference")
    monkeypatch.setenv("GENERATIVEQC_MD_J_WORK_COUNTS", "1")
    with (
        NativeAO(WATER, "def2-svp", representation="spherical") as basis,
        ResidentMdJProbe(basis, _spec("restricted"), screening_tolerance=0) as plan,
    ):
        _assert_selected(plan)
        density = np.eye(basis.nao)
        expected = plan.evaluate(density).coulomb
        # The native plan froze diagnostic storage at construction; changing
        # the environment later cannot make that plan graph-safe.
        monkeypatch.delenv("GENERATIVEQC_MD_J_WORK_COUNTS")
        with pytest.raises(RuntimeError, match="WORK_COUNTS disabled"):
            plan.capture_graph()
        np.testing.assert_allclose(
            plan.evaluate(-0.7 * density).coulomb,
            -0.7 * expected,
            atol=3e-11,
            rtol=3e-11,
        )


@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
def test_reference_graph_capture_preserves_retained_k(
    spin: str, monkeypatch: Any, record_property: Any
) -> None:
    """Capture complete J/K, including native device-side spin/AO staging."""
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe

    _select(monkeypatch, "reference")
    monkeypatch.delenv("GENERATIVEQC_MD_J_WORK_COUNTS", raising=False)
    with NativeAO(WATER, "def2-svp", representation="spherical") as basis:
        density = np.random.default_rng(911).normal(size=(basis.nao, basis.nao)) * 0.04
        if spin == "unrestricted":
            density = np.stack((density, -0.3 * density.T))
        with ResidentMdJProbe(basis, _spec(spin, True), screening_tolerance=0) as plan:
            _assert_selected(plan)
            plan.evaluate(density)
            plan.capture_graph()
            for changed in (
                -0.7 * np.swapaxes(density, -1, -2),
                np.zeros_like(density),
                density,
            ):
                observed = plan.evaluate_graph(changed)
                expected = plan.evaluate(changed)
                np.testing.assert_allclose(
                    observed.coulomb, expected.coulomb, atol=3e-11, rtol=3e-11
                )
                np.testing.assert_allclose(
                    observed.exchange, expected.exchange, atol=3e-11, rtol=3e-11
                )
            record_property("selected_schedule", plan.diagnostics["direct_schedule"])
            record_property("graph_nodes", plan.diagnostics["graph_nodes"])
            record_property(
                "graph_kernel_nodes", plan.diagnostics["graph_kernel_nodes"]
            )
            record_property(
                "graph_device_copy_nodes", plan.diagnostics["graph_device_copy_nodes"]
            )
