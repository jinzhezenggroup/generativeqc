"""Host-only executable endpoint contracts; no CUDA/native library is loaded."""

from __future__ import annotations

import copy
import os
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from benchmarks import _md_j_engine_endpoints as endpoints


def _case() -> dict[str, Any]:
    # Test-double shells exercise the receipt/ownership contract only. Actual
    # campaigns resolve and freeze the named molecular basis in the driver.
    return {
        "name": "water32-sto-3g",
        "basis": "sto-3g",
        "nao": 224,
        "atoms": [
            (symbol, [float(index), 0.0, 0.0])
            for index in range(32)
            for symbol in ("O", "H", "H")
        ],
        "charge": 0,
        "multiplicity": 1,
        "shells": [
            {
                "atom_index": index % 96,
                "angular_momentum": 0,
                "primitives": [[1.0, 1.0]],
            }
            for index in range(224)
        ],
        "density_file": "must-never-enter-scf.npy",
    }


def _work(engine: str, fock: int = 1) -> str:
    lines = [
        f"MD_J_WORK fock={fock} bra={a} ket={b} uniform_roots=1 uniform_directions=1 uniform_summands=1 residual_candidates=2 residual_tasks=1 residual_roots=1"
        for a in range(5)
        for b in range(5)
    ]
    if engine == "reference":
        lines.extend(
            f"MD_J_ENGINE_WORK fock={fock} bra={a} ket={b} source_tiles=1 screened_tiles=0 bra_active_primitives=2 ket_active_primitives=2"
            for a in range(5)
            for b in range(a, 5)
        )
    return "\n".join(lines) + "\n"


def _sequence(
    engine: str, instrumented: bool, *, fail: bool = False
) -> tuple[dict[str, Any], list[Any]]:
    events: list[Any] = []
    ticks = [0.0]

    class Owner:
        calls = 0

        def execute(self, coordinates: Any, **options: Any) -> Any:
            self.calls += 1
            events.append(("execute", coordinates, options))
            ticks[0] += 7.0 if coordinates is not None else 5.0
            if fail:
                raise RuntimeError("injected force failure")
            if instrumented:
                os.write(2, _work(engine, self.calls).encode())
            diagnostic = {
                "physical_residual_max": 1e-11,
                "initial_density_used": False,
                "grid_points": 294912,
                "scf_domain": "physical-test-domain",
            }
            return SimpleNamespace(
                items=[
                    SimpleNamespace(
                        energy=-100.0 + (0.001 if coordinates is not None else 0.0),
                        forces=np.zeros((96, 3)),
                        converged=True,
                        succeeded=True,
                        status=0,
                        status_message="success",
                        iterations=1,
                        fock_builds=1,
                        executed_backend="cuda",
                        energy_change=1e-12,
                        density_rms=1e-11,
                        warm_start_used=False,
                        basis_metadata={
                            "orbital": {"mathematical_identity": "frozen-basis"}
                        },
                        precision={
                            "requested_mode": "fp64",
                            "effective_bits": 64,
                            "mixed_stage_fock_builds": 0,
                        },
                        ks_diagnostic=SimpleNamespace(
                            **diagnostic, to_payload=lambda: dict(diagnostic)
                        ),
                    )
                ]
            )

        def close(self) -> None:
            events.append("close")
            ticks[0] += 11.0
            if instrumented:
                os.write(2, f"MD_J_NORMAL_COUNTS={self.calls}\n".encode())

    class Calculator:
        def prepare_batch(self, atoms: Any, **options: Any) -> Owner:
            events.append(("prepare", atoms, options))
            ticks[0] += 3.0
            return Owner()

        def clear_cache(self) -> None:
            events.append("clear_cache")
            ticks[0] += 13.0

    def factory() -> Calculator:
        events.append("construct")
        ticks[0] += 2.0
        return Calculator()

    try:
        result = endpoints._run_sequence(
            endpoints._scientific_input(_case()),
            engine=engine,
            instrumented=instrumented,
            warm_repeats=3,
            factory=factory,
            clock=lambda: ticks[0],
        )
    except RuntimeError:
        assert events[-2:] == ["close", "clear_cache"]
        raise
    return result, events


@pytest.fixture(scope="module")
def campaign() -> list[dict[str, Any]]:
    samples = []
    for engine in ("incumbent", "reference"):
        for repetition in range(4):
            sample, _ = _sequence(engine, repetition == 3)
            sample.update(
                repetition=repetition,
                pid=1000 + len(samples),
                library_sha256="a" * 64,
                source_identity={"revision": "source"},
                input_sha256="b" * 64,
                run_id="unique-campaign",
                process_status="passed",
            )
            samples.append(sample)
    return samples


def test_complete_cold_warm_moved_timing_and_owner_contract() -> None:
    receipt, events = _sequence("reference", False)
    assert events[0] == "construct"
    assert events[1][0] == "prepare"
    assert events[1][2] == {"charges": [0], "multiplicities": [1], "warm_start": False}
    assert [row["seconds"] for row in receipt["phases"]] == [10.0, 5.0, 5.0, 5.0, 7.0]
    assert receipt["teardown_seconds"] == 24.0
    assert events[-2:] == ["close", "clear_cache"]
    executions = [
        event for event in events if isinstance(event, tuple) and event[0] == "execute"
    ]
    assert all(
        event[2] == {"properties": ("energy", "forces"), "strict": True}
        for event in executions
    )
    assert all(event[1] is None for event in executions[:-1])
    assert executions[-1][1] == [receipt["geometries"]["moved"]["coordinates"]]
    assert "density_file" not in receipt["scientific_input"]
    assert (
        receipt["geometries"]["original"]["sha256"]
        != receipt["geometries"]["moved"]["sha256"]
    )


def test_failure_closes_owner_and_calculator() -> None:
    with pytest.raises(RuntimeError, match="injected force failure"):
        _sequence("reference", False, fail=True)


def test_instrumented_source_evidence_is_per_phase_and_untimed() -> None:
    receipt, _ = _sequence("reference", True)
    assert receipt["teardown_seconds"] is None
    assert receipt["execution"]["destroyed_owner_md_calls"] == [5]
    for phase in receipt["phases"]:
        assert phase["seconds"] is None
        assert len(phase["execution"]["work_rows"]) == 25
        assert len(phase["execution"]["selected_source_rows"]) == 15


def test_matched_campaign_reports_complete_endpoint_distributions(
    campaign: Any,
) -> None:
    summary = endpoints.validate_endpoint_receipts(campaign)
    assert summary["status"] == "passed", summary["failures"]
    assert summary["incumbent_over_reference"] == {
        "cold": 1.0,
        "warm": 1.0,
        "moved": 1.0,
    }
    assert len(summary["timing"]["reference"]["cold"]["samples"]) == 3
    assert len(summary["timing"]["reference"]["warm"]["samples"]) == 9
    assert len(summary["comparisons"]) == 40
    assert not summary["default_promotion_qualified"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema",), "other"),
        (("process_status",), "timeout"),
        (("settings", "precision"), "auto"),
        (("settings_sha256",), "stale"),
        (("scientific_input_sha256",), "stale"),
        (("natoms",), 3),
        (("nao",), 24),
        (("process_cold",), False),
        (("context_primed",), True),
        (("supplied_density",), True),
        (("warm_repeats",), 1),
        (("library_sha256",), "different"),
        (("source_identity",), "different"),
        (("input_sha256",), "different"),
        (("run_id",), "different"),
        (("pid",), 1000),
        (("repetition",), 0),
        (("phases", 0, "converged"), False),
        (("phases", 0, "backend"), "cpu"),
        (("phases", 0, "forces"), None),
        (("phases", 0, "forces_sha256"), "wrong"),
        (("phases", 0, "energy"), float("nan")),
        (("phases", 0, "energy"), -99.0),
        (("phases", 0, "physical_residual_max"), 1e-5),
        (("phases", 0, "physical_residual_max"), None),
        (("phases", 0, "density_rms"), 1e-5),
        (("phases", 0, "energy_change"), 1e-5),
        (("phases", 0, "fock_builds"), None),
        (("phases", 0, "precision", "requested_mode"), "auto"),
        (("phases", 0, "precision", "effective_bits"), 32),
        (("phases", 0, "precision", "mixed_stage_fock_builds"), 1),
        (("phases", 0, "iterations"), 0),
        (("phases", 0, "warm_start_used"), True),
        (("phases", 0, "initial_density_used"), True),
        (("phases", 0, "grid_points"), 100),
        (("phases", 0, "geometry_sha256"), "other"),
        (("phases", 0, "seconds"), -1),
        (("phases", 0, "seconds"), float("inf")),
        (
            ("phases", 0, "basis_metadata", "orbital", "mathematical_identity"),
            "different",
        ),
        (("phases", 0, "ks_diagnostic", "scf_domain"), "other"),
    ],
)
def test_malformed_or_unmatched_receipt_fails_closed(
    campaign: Any, path: Any, value: Any
) -> None:
    samples = copy.deepcopy(campaign)
    target = samples[1]
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value
    result = endpoints.validate_endpoint_receipts(samples)
    assert result["status"] == "failed"
    assert result["failures"]
    assert result["timing"] == result["incumbent_over_reference"] == {}


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "fallback",
        "angular",
        "source-identity",
        "zero-work",
        "count",
        "instrumented-time",
        "diagnostic-error",
        "clean-counters",
    ],
)
def test_executed_selection_and_clean_diagnostic_parity_fail_closed(
    campaign: Any, mutation: str
) -> None:
    samples = copy.deepcopy(campaign)
    diagnostic = samples[-1]
    phase = diagnostic["phases"][0]
    if mutation == "missing":
        samples.pop()
    elif mutation == "fallback":
        phase["execution"]["selected_source_rows"] = []
    elif mutation == "angular":
        phase["execution"]["work_rows"][0]["bra"] = 99
    elif mutation == "source-identity":
        phase["execution"]["selected_source_rows"][0]["fock"] = 9
    elif mutation == "zero-work":
        for row in phase["execution"]["selected_source_rows"]:
            row["source_tiles"] = 0
    elif mutation == "count":
        diagnostic["execution"]["destroyed_owner_md_calls"] = [6]
    elif mutation == "instrumented-time":
        phase["seconds"] = 1.0
    elif mutation == "diagnostic-error":
        phase["energy"] += 1e-3
    else:
        samples[0]["phases"][0]["execution"] = copy.deepcopy(phase["execution"])
    assert endpoints.validate_endpoint_receipts(samples)["status"] == "failed"


def test_missing_or_insufficient_evidence_never_qualifies(campaign: Any) -> None:
    for samples in ([], campaign[:4], campaign[1:], [{"schema": endpoints.SCHEMA}]):
        assert endpoints.validate_endpoint_receipts(samples)["status"] == "failed"
    assert (
        endpoints.validate_endpoint_receipts(campaign, minimum_samples=1)["status"]
        == "failed"
    )


def test_both_engines_agreeing_on_wrong_warm_state_fails(campaign: Any) -> None:
    samples = copy.deepcopy(campaign)
    for sample in samples:
        sample["phases"][1]["energy"] += 1e-3
    result = endpoints.validate_endpoint_receipts(samples)
    assert result["status"] == "failed"
    assert "same-geometry warm" in result["failures"][0]


def test_worker_does_not_import_or_execute_cuda_without_finite_allocation(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="finite Slurm"):
        endpoints.run_endpoint_worker(_case(), engine="reference", instrumented=False)


def test_worker_uses_explicit_shells_and_exact_scientific_options(
    monkeypatch: Any,
) -> None:
    from benchmarks import md_j_engine_acceptance

    monkeypatch.setattr(
        md_j_engine_acceptance, "_finite_slurm", lambda: {"job_id": "test"}
    )
    monkeypatch.setitem(
        sys.modules,
        "generativeqc",
        SimpleNamespace(
            Calculator=lambda **options: options,
            GridSpec=lambda **options: options,
            KsOptions=lambda **options: options,
            Primitive=lambda *values: values,
            Shell=lambda *values: values,
        ),
    )
    monkeypatch.setattr(
        endpoints,
        "_run_sequence",
        lambda inputs, **options: {
            "inputs": inputs,
            "options": options["factory"](),
            "instrumented": options["instrumented"],
        },
    )
    monkeypatch.setenv("GENERATIVEQC_DISABLE_MD_J", "1")
    monkeypatch.setenv("GENERATIVEQC_MD_J_ENGINE", "incumbent")
    monkeypatch.setenv("GENERATIVEQC_MD_J_COUNTS", "1")
    monkeypatch.setenv("GENERATIVEQC_MD_J_WORK_COUNTS", "1")
    result = endpoints.run_endpoint_worker(
        _case(), engine="reference", instrumented=False
    )
    options = result["options"]
    for key in (
        "method",
        "precision",
        "device",
        "basis_representation",
        "density_fitting",
        "initial_guess",
        "max_iterations",
        "energy_tolerance",
        "density_tolerance",
        "screening_tolerance",
        "diis_history",
    ):
        assert options[key] == endpoints.endpoint_settings()[key]
    assert options["ks_options"] == {
        "grid": {"radial_points": 24, "angular_polar": 8, "angular_azimuth": 16},
        "tile_points": 256,
        "xc_schedule": "device_fused",
    }
    assert len(options["basis"]) == 224
    assert "density_file" not in result["inputs"]
    assert os.environ["GENERATIVEQC_MD_J_ENGINE"] == "reference"
    assert "GENERATIVEQC_DISABLE_MD_J" not in os.environ
    assert "GENERATIVEQC_MD_J_COUNTS" not in os.environ
    assert "GENERATIVEQC_MD_J_WORK_COUNTS" not in os.environ
    endpoints.run_endpoint_worker(_case(), engine="incumbent", instrumented=True)
    assert os.environ["GENERATIVEQC_MD_J_ENGINE"] == "incumbent"
    assert os.environ["GENERATIVEQC_MD_J_COUNTS"] == "1"
    assert os.environ["GENERATIVEQC_MD_J_WORK_COUNTS"] == "1"


def test_frozen_settings_and_geometry_do_not_alias_inputs() -> None:
    first, second = endpoints.endpoint_settings(), endpoints.endpoint_settings()
    first["grid"]["radial_points"] = 1
    assert second["grid"]["radial_points"] == 24
    case = _case()
    frozen = endpoints._scientific_input(case)
    case["atoms"][0][1][0] = 999.0
    assert frozen["atoms"][0][1][0] == 0.0
