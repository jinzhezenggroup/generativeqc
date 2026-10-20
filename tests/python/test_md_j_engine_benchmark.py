"""Host-only contracts for the bounded full-size MD-J performance driver."""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from benchmarks import md_j_engine_benchmark as benchmark


def arguments(tmp_path: Path, *extra: str) -> Any:
    return benchmark.parse_args(["--output", str(tmp_path / "receipt.json"), *extra])


def work(engine: str) -> dict[str, Any]:
    return {
        "work_rows": [
            {
                "fock": 1,
                "bra": a,
                "ket": b,
                "uniform_roots": 1,
                "uniform_directions": 1,
                "uniform_summands": 1,
                "residual_candidates": 0,
                "residual_tasks": 0,
                "residual_roots": 0,
            }
            for a in range(5)
            for b in range(5)
        ],
        "selected_source_rows": [
            {
                "fock": 1,
                "bra": a,
                "ket": b,
                "source_tiles": 1,
                "screened_tiles": 0,
                "bra_active_primitives": 1,
                "ket_active_primitives": 1,
            }
            for a in range(5)
            for b in range(a, 5)
        ]
        if engine == "reference"
        else [],
    }


def cohort() -> list[dict[str, Any]]:
    """Synthetic receipts model successful observations, never GPU evidence."""
    rows = []
    for engine, seconds in (("reference", 1.0), ("incumbent", 2.0)):
        for phase, repeat in [("cold", i) for i in range(3)] + [
            ("warm", 0),
            ("diagnostic", 0),
        ]:
            rows.append(
                {
                    "status": "passed",
                    "engine": engine,
                    "phase": phase,
                    "repetition": repeat,
                    "pid": 100 + len(rows),
                    "backend": "cuda",
                    "instrumented": phase == "diagnostic",
                    "execution": work(engine)
                    if phase == "diagnostic"
                    else {"work_rows": [], "selected_source_rows": []},
                    "run_id": "run",
                    "case": "water32-sto-3g",
                    "input_file_sha256": "input",
                    "library_sha256": "library",
                    "slurm_job_id": "123",
                    "cuda_visible_devices": "assigned",
                    "comparison": {"passed": True},
                    "sampled_peak_process_gpu_bytes": 1000,
                    "peak_host_rss_bytes": 2000,
                    "process_wall_seconds": seconds + 1,
                    "timings": {
                        "j_calls": 8 if phase == "warm" else 1,
                        "cold_complete_j_seconds": seconds,
                        "warm_complete_j_seconds": [seconds] * 5
                        if phase == "warm"
                        else [],
                        "resident_device_seconds": [seconds / 2] * 5
                        if phase == "warm"
                        else [],
                    },
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return benchmark.summarize_case(
        rows,
        cold_repeats=3,
        warm_repeats=5,
        engines=["reference", "incumbent"],
        minimum_speedup=1.0,
    )


def test_default_full_size_prescriptions_are_deterministic() -> None:
    assert {case.expected_nao for case in benchmark.CASES.values()} == {224, 768}
    for case in benchmark.CASES.values():
        assert case.atoms() == case.atoms()
        assert len(case.atoms()) == 96
        assert sum(symbol == "O" for symbol, _ in case.atoms()) == 32
        assert len({position for _, position in case.atoms()}) == 96
        assert case.prescription()["native_screening_tolerance"] == 0
        assert case.prescription()["reference_screening_tolerance"] == 1e-30


@pytest.mark.parametrize(
    "options",
    [
        ["--cold-repeats", "2"],
        ["--cold-repeats", "21"],
        ["--warm-repeats", "4"],
        ["--warmup", "0"],
        ["--timeout", "0"],
        ["--total-timeout", "86401"],
        ["--atol", "nan"],
        ["--rtol", "inf"],
        ["--minimum-speedup", "0.9"],
        ["--require-upstream"],
        ["--cases", "water32-sto-3g", "water32-sto-3g"],
        ["--worker", "reference"],
        ["--scope", "energy-force", "--require-upstream", "--upstream-root", "/absent"],
    ],
)
def test_invalid_or_unbounded_options_fail_before_runtime(
    tmp_path: Path, options: list[str]
) -> None:
    with pytest.raises(SystemExit):
        arguments(tmp_path, *options)


def test_raw_output_cannot_overwrite_reviewed_evidence() -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            ["--output", str(benchmark.ROOT / "benchmarks/results/raw.json")]
        )


def test_dry_run_is_explicitly_not_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("dry-run touched native/CUDA prerequisites")

    monkeypatch.setattr(benchmark, "preflight", unexpected)
    report = benchmark.run_campaign(arguments(tmp_path, "--dry-run"))
    assert report["status"] == "not-run"
    assert report["scope"] == "all"
    assert not report["performance_acceptance_passed"]
    assert not report["default_promotion_qualified"]
    assert report["samples"] == []
    assert len(report["prescriptions"]) == 2
    assert json.loads((tmp_path / "receipt.json").read_text()) == json.loads(
        json.dumps(report)
    )
    with pytest.raises(ValueError, match="already exists"):
        benchmark.run_campaign(arguments(tmp_path, "--dry-run"))


def test_complete_matched_cohort_separates_timing_boundaries() -> None:
    result = summarize(cohort())
    assert result["performance_acceptance_passed"]
    assert result["incumbent_over_reference_ratio"]["cold_complete_j"] == 2
    assert result["incumbent_over_reference_ratio"]["cold_process_wall"] == 1.5
    assert result["engines"]["reference"]["warm_complete_j"]["count"] == 5
    assert result["engines"]["reference"]["resident_device"]["median_seconds"] == 0.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "timeout"),
        ("backend", "cpu"),
        ("input_file_sha256", "changed"),
        ("library_sha256", "changed"),
        ("run_id", "stale"),
        ("pid", 101),
        ("comparison", {"passed": False}),
        ("sampled_peak_process_gpu_bytes", None),
        ("peak_host_rss_bytes", 0),
        ("cuda_visible_devices", "different"),
    ],
)
def test_bad_receipt_cannot_qualify(field: str, value: Any) -> None:
    rows = deepcopy(cohort())
    rows[0][field] = value
    assert not summarize(rows)["performance_acceptance_passed"]


def test_partial_cohort_and_hidden_cold_repeats_fail() -> None:
    assert not summarize(cohort()[:-1])["performance_acceptance_passed"]
    rows = cohort()
    rows[0]["timings"]["j_calls"] = 2
    assert not summarize(rows)["performance_acceptance_passed"]
    rows = cohort()
    rows[3]["timings"]["warm_complete_j_seconds"].pop()
    assert not summarize(rows)["performance_acceptance_passed"]


def test_slow_reference_is_retained_as_failed_performance() -> None:
    rows = cohort()
    for row in rows:
        if row["engine"] == "reference":
            row["timings"]["cold_complete_j_seconds"] = 10
    result = summarize(rows)
    assert not result["performance_acceptance_passed"]
    assert result["engines"]["reference"]["cold_complete_j"]["raw_seconds"] == [10] * 3


@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [float("nan")], [float("inf")]])
def test_invalid_timing_is_never_ranked(values: list[float]) -> None:
    with pytest.raises(ValueError):
        benchmark.timing_summary(values)


def test_median_and_dispersion_retain_outliers() -> None:
    summary = benchmark.timing_summary([1, 2, 100])
    assert summary["raw_seconds"] == [1, 2, 100]
    assert summary["median_seconds"] == 2
    assert summary["median_absolute_deviation_seconds"] == 1


def test_numerical_gate_uses_full_matrix_and_declared_tolerance() -> None:
    expected = np.eye(3)
    actual = expected.copy()
    actual[1, 2] += 1e-8
    assert not benchmark.numerical_comparison(actual, expected, 1e-9, 1e-10)["passed"]
    assert benchmark.numerical_comparison(actual, expected, 2e-8, 1e-10)["passed"]
    actual[1, 2] = float("nan")
    assert not benchmark.numerical_comparison(actual, expected, 1e-9, 1e-10)["passed"]
    assert not benchmark.numerical_comparison(np.ones(2), expected, 1e-9, 1e-10)[
        "passed"
    ]


def make_receipt(tmp_path: Path) -> Path:
    source, matrix, output = (
        tmp_path / "input.json",
        tmp_path / "matrix.npy",
        tmp_path / "worker.json",
    )
    source.write_text('{"frozen":true}')
    np.save(matrix, np.eye(2), allow_pickle=False)
    benchmark.write_json(
        output,
        {
            "schema": benchmark.SCHEMA,
            "run_id": "run",
            "status": "passed",
            "case": "case",
            "engine": "reference",
            "phase": "cold",
            "input_file": str(source),
            "input_file_sha256": benchmark.sha256(source),
            "matrix_file": str(matrix),
            "matrix_file_sha256": benchmark.sha256(matrix),
        },
    )
    return output


def read_receipt(path: Path) -> dict[str, Any]:
    return benchmark.load_receipt(
        path, run_id="run", case="case", engine="reference", phase="cold"
    )


def test_receipt_loader_checks_fresh_identity_and_file_bytes(tmp_path: Path) -> None:
    path = make_receipt(tmp_path)
    assert read_receipt(path)["status"] == "passed"
    (tmp_path / "matrix.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        read_receipt(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "stale"),
        ("status", "failed"),
        ("phase", "warm"),
        ("engine", "incumbent"),
    ],
)
def test_stale_receipts_fail(tmp_path: Path, field: str, value: Any) -> None:
    path = make_receipt(tmp_path)
    data = json.loads(path.read_text())
    data[field] = value
    benchmark.write_json(path, data)
    with pytest.raises(ValueError, match="identity/status"):
        read_receipt(path)


def test_missing_truncated_nonfinite_and_escaped_receipts_fail(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_receipt(tmp_path / "missing")
    path = tmp_path / "bad.json"
    path.write_text('{"x":')
    with pytest.raises(ValueError):
        read_receipt(path)
    path.write_text('{"x":NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        read_receipt(path)
    path = make_receipt(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped = outside / "matrix.npy"
    escaped.write_bytes((tmp_path / "matrix.npy").read_bytes())
    data = json.loads(path.read_text())
    data["matrix_file"] = str(escaped)
    benchmark.write_json(path, data)
    with pytest.raises(ValueError, match="outside"):
        read_receipt(path)


def test_failed_preflight_leaves_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(args: Any) -> None:
        raise RuntimeError("finite Slurm allocation required")

    monkeypatch.setattr(benchmark, "preflight", fail)
    report = benchmark.run_campaign(arguments(tmp_path))
    assert report["status"] == "failed"
    assert not report["performance_acceptance_passed"]
    assert "Slurm" in report["gate_failures"][0]
    assert json.loads((tmp_path / "receipt.json").read_text())["status"] == "failed"


def test_bounded_process_retains_output_and_kills_timeout(tmp_path: Path) -> None:
    execution = benchmark.run_process(
        [
            sys.executable,
            "-c",
            "import time; print('entered', flush=True); time.sleep(10)",
        ],
        tmp_path / "slow.json",
        dict(os.environ),
        0.2,
        measure_gpu=False,
    )
    assert execution["status"] == "timeout"
    assert execution["returncode"] != 0
    assert "entered" in Path(execution["stdout"]).read_text()
    assert execution["process_wall_seconds"] < 5


def test_worker_environment_disables_instrumentation_preserves_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-exact-assignment")
    monkeypatch.setenv("GENERATIVEQC_MD_J_WORK_COUNTS", "1")
    monkeypatch.setenv("GENERATIVEQC_MD_J_ENGINE", "reference")
    monkeypatch.setenv("GENERATIVEQC_PROFILE", "/different/cached/native-profile")
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", "/pinned/native.so")
    monkeypatch.setenv("GENERATIVEQC_DIRECT_FORCE_CENSUS", "1")
    monkeypatch.setenv("GENERATIVEQC_UNREVIEWED_SCIENTIFIC_CONTROL", "1")
    result = benchmark._worker_environment(None)
    assert result["CUDA_VISIBLE_DEVICES"] == "GPU-exact-assignment"
    assert "GENERATIVEQC_MD_J_WORK_COUNTS" not in result
    assert "GENERATIVEQC_MD_J_ENGINE" not in result
    assert result["OMP_NUM_THREADS"] == "1"
    assert result["GENERATIVEQC_PROFILE"] == "off"
    assert result["GENERATIVEQC_LIBRARY"] == "/pinned/native.so"
    assert "GENERATIVEQC_DIRECT_FORCE_CENSUS" not in result
    assert "GENERATIVEQC_UNREVIEWED_SCIENTIFIC_CONTROL" not in result


def test_native_source_binding_rejects_unmatched_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "native.so"
    library.write_bytes(b"native")
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", str(library))
    native = SimpleNamespace(
        _name=str(library), generativeqc_get_source_identity=lambda: b"actual"
    )
    monkeypatch.setitem(
        sys.modules,
        "generativeqc",
        SimpleNamespace(_native=SimpleNamespace(load_library=lambda: native)),
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        benchmark.native_identity("other")
    result = benchmark.native_identity("actual")
    assert result["loaded_library_sha256"] == benchmark.sha256(library)


@pytest.mark.parametrize("mutate_source", [False, True])
def test_fake_campaign_checks_final_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate_source: bool
) -> None:
    library = tmp_path / "native.so"
    library.write_bytes(b"native")
    provenance = {
        "source": {"native_source_identity": "expected"},
        "library": str(library),
        "library_sha256": benchmark.sha256(library),
    }
    monkeypatch.setattr(benchmark, "preflight", lambda args: deepcopy(provenance))
    monkeypatch.setattr(
        benchmark,
        "source_identity",
        lambda: {"native_source_identity": "changed" if mutate_source else "expected"},
    )
    next_pid = 100

    def fake_run(
        command: list[str],
        output: Path,
        environment: dict[str, str],
        timeout: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal next_pid
        next_pid += 1
        worker_args = benchmark.parse_args(command[2:])
        if worker_args.worker == "prepare":
            benchmark.write_json(output, {"run_id": worker_args.run_id})
        else:
            matrix = output.with_suffix(".matrix.npy")
            np.save(matrix, np.eye(2), allow_pickle=False)
            template = next(
                row
                for row in cohort()
                if row["engine"]
                == (
                    worker_args.worker
                    if worker_args.worker != "oracle"
                    else "reference"
                )
                and row["phase"] == worker_args.phase
            )
            record = dict(
                template,
                schema=benchmark.SCHEMA,
                case=worker_args.cases[0],
                engine=worker_args.worker,
                run_id=worker_args.run_id,
                pid=next_pid,
                phase=worker_args.phase,
                input_file=str(worker_args.input),
                input_file_sha256=benchmark.sha256(worker_args.input),
                matrix_file=str(matrix),
                matrix_file_sha256=benchmark.sha256(matrix),
                library_sha256=provenance["library_sha256"],
                native_identity={
                    "source_identity": "expected",
                    "loaded_library_sha256": provenance["library_sha256"],
                },
            )
            benchmark.write_json(output, record)
        stderr = output.with_suffix(".stderr.txt")
        text = ""
        if worker_args.phase == "diagnostic":
            census = work(worker_args.worker)
            for field, prefix in (
                ("work_rows", "MD_J_WORK"),
                ("selected_source_rows", "MD_J_ENGINE_WORK"),
            ):
                for row in census[field]:
                    text += (
                        prefix
                        + " "
                        + " ".join(f"{key}={value}" for key, value in row.items())
                        + "\n"
                    )
        stderr.write_text(text)
        return {
            "status": "passed",
            "pid": next_pid,
            "process_wall_seconds": 2.0,
            "sampled_peak_process_gpu_bytes": 1000,
            "stderr": str(stderr),
        }

    monkeypatch.setattr(benchmark, "run_process", fake_run)
    result = benchmark.run_campaign(
        arguments(
            tmp_path, "--scope", "j", "--cases", "water32-sto-3g", "--warm-repeats", "5"
        )
    )
    assert result["performance_acceptance_passed"] is not mutate_source
    if mutate_source:
        assert "changed during" in result["gate_failures"][0]
    assert not result["default_promotion_qualified"]


def test_direct_oracle_never_uses_incore_eri_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyscf = pytest.importorskip("pyscf")
    molecule = pyscf.gto.M(
        atom="H 0 0 0; H 0 0 1.4", unit="Bohr", basis="sto-3g", verbose=0
    )
    density = np.asarray([[0.7, 0.2], [0.2, 0.5]])
    expected = np.einsum("ijkl,kl->ij", molecule.intor("int2e"), density)
    original = molecule.intor

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("int2e"):
            pytest.fail("full-size oracle must never materialize four-index ERIs")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(molecule, "intor", guarded)
    np.testing.assert_allclose(
        benchmark.direct_pyscf_j(molecule, density), expected, atol=1e-13, rtol=1e-13
    )


def test_missing_or_instrumented_raw_source_evidence_cannot_rank() -> None:
    rows = cohort()
    rows[4]["execution"]["selected_source_rows"] = []
    assert not summarize(rows)["performance_acceptance_passed"]
    rows = cohort()
    rows[0]["instrumented"] = True
    assert not summarize(rows)["performance_acceptance_passed"]


def test_upstream_binary_must_remain_identical_across_samples() -> None:
    rows = cohort()
    upstream = [
        deepcopy(row)
        for row in rows
        if row["engine"] == "incumbent" and row["phase"] != "diagnostic"
    ]
    for index, row in enumerate(upstream):
        row.update(
            engine="upstream",
            pid=500 + index,
            metadata={
                "upstream_commit": benchmark.UPSTREAM_COMMIT,
                "upstream_library": "/pinned/lib.so",
                "upstream_library_sha256": "same",
            },
        )
    values = {
        "cold_repeats": 3,
        "warm_repeats": 5,
        "engines": ["reference", "incumbent", "upstream"],
        "minimum_speedup": 1.0,
    }
    assert benchmark.summarize_case(rows + upstream, **values)[
        "performance_acceptance_passed"
    ]
    upstream[-1]["metadata"]["upstream_library_sha256"] = "different"
    assert not benchmark.summarize_case(rows + upstream, **values)[
        "performance_acceptance_passed"
    ]


@pytest.mark.parametrize(
    "phase,ratio", [("cold", 0.8), ("warm", 0.8), ("moved", 0.8), ("all", 1.2)]
)
def test_full_endpoint_orchestration_applies_every_predeclared_speed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, ratio: float
) -> None:
    from benchmarks import _md_j_engine_endpoints as endpoints

    library = tmp_path / "native.so"
    library.write_bytes(b"native")
    source = {"native_source_identity": "expected"}
    provenance = {
        "source": source,
        "library": str(library),
        "library_sha256": benchmark.sha256(library),
    }
    monkeypatch.setattr(benchmark, "preflight", lambda args: deepcopy(provenance))
    monkeypatch.setattr(benchmark, "source_identity", lambda: source)
    observed: list[dict[str, Any]] = []
    pid = 200

    def fake_run(
        command: list[str],
        output: Path,
        environment: dict[str, str],
        timeout: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal pid
        pid += 1
        args = benchmark.parse_args(command[2:])
        if args.worker == "prepare":
            benchmark.write_json(output, {"run_id": args.run_id})
        else:
            benchmark.write_json(
                output,
                {
                    "schema": benchmark.SCHEMA,
                    "status": "passed",
                    "run_id": args.run_id,
                    "case": args.cases[0],
                    "engine": args.engine,
                    "scope": "energy-force",
                    "pid": pid,
                    "input_file": str(args.input),
                    "input_file_sha256": benchmark.sha256(args.input),
                    "library_sha256": provenance["library_sha256"],
                    "peak_host_rss_bytes": 2000,
                    "native_identity": {
                        "source_identity": "expected",
                        "loaded_library_sha256": provenance["library_sha256"],
                    },
                    "endpoint_result": {
                        "engine": args.engine,
                        "instrumented": args.instrumented,
                    },
                },
            )
        return {
            "status": "passed",
            "pid": pid,
            "process_wall_seconds": 2.0,
            "sampled_peak_process_gpu_bytes": 1000,
            "stderr": "not-read-for-endpoint",
        }

    def validate(
        samples: list[dict[str, Any]], *, minimum_samples: int
    ) -> dict[str, Any]:
        observed.extend(samples)
        assert minimum_samples == 3
        for engine in benchmark.NATIVE_ENGINES:
            chosen = [row for row in samples if row["engine"] == engine]
            assert len([row for row in chosen if not row["instrumented"]]) == 3
            assert len([row for row in chosen if row["instrumented"]]) == 1
        assert all(
            row["source_identity"] == source and row["process_status"] == "passed"
            for row in samples
        )
        return {
            "status": "passed",
            "failures": [],
            "numerical_parity_passed": True,
            "incumbent_over_reference": {
                key: ratio if phase in (key, "all") else 1.2
                for key in ("cold", "warm", "moved")
            },
        }

    monkeypatch.setattr(benchmark, "run_process", fake_run)
    monkeypatch.setattr(endpoints, "validate_endpoint_receipts", validate)
    report = benchmark.run_campaign(
        arguments(tmp_path, "--scope", "energy-force", "--cases", "water32-sto-3g")
    )
    result = report["cases"]["water32-sto-3g"]["energy-force"]
    assert result["numerical_parity_passed"]
    assert len(observed) == 8
    assert report["performance_acceptance_passed"] is (ratio >= 1)
    assert not report["default_promotion_qualified"]
    if ratio < 1:
        assert phase in result["failures"][0]


def test_blocking_cuda_launches_fail_before_performance_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    with pytest.raises(RuntimeError, match="CUDA_LAUNCH_BLOCKING"):
        benchmark.preflight(arguments(tmp_path))


def test_environment_provenance_never_records_arbitrary_secret_values() -> None:
    result = benchmark.environment_provenance(
        {
            "SCCACHE_REDIS": "redis://user:secret-password@host",
            "GENERATIVEQC_API_TOKEN": "private-token-value",
            "CCACHE_REMOTE_STORAGE": "https://user:another-secret@cache",
            "CUDA_VISIBLE_DEVICES": "GPU-assigned",
            "CCACHE_DIR": "/safe/cache",
        }
    )
    serialized = json.dumps(result)
    for secret in (
        "secret-password",
        "private-token-value",
        "another-secret",
        "redis://",
    ):
        assert secret not in serialized
    assert result["ambient_runtime_controls"] == {
        "CUDA_VISIBLE_DEVICES": "GPU-assigned",
        "CCACHE_DIR": "/safe/cache",
    }
    assert set(result["unrecorded_ambient_control_names"]) == {
        "SCCACHE_REDIS",
        "GENERATIVEQC_API_TOKEN",
        "CCACHE_REMOTE_STORAGE",
    }
