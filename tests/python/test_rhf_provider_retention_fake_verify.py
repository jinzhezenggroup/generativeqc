from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[2]
    / "benchmarks"
    / "rhf_provider_retention_fake"
    / "verify_results.py"
)
SPEC = importlib.util.spec_from_file_location(
    "rhf_provider_retention_fake_verify", MODULE_PATH
)
assert SPEC and SPEC.loader
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


CASES = (
    "positive.baseline",
    "positive.shim",
    "provider_failure.baseline",
    "provider_failure.shim",
    "overflow.shim",
    "resolution_failure.shim",
    "o_excl.first",
    "o_excl.second",
    "write_failure.shim",
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def event(
    sequence: int,
    name: str,
    *,
    phase: str = "end",
    origin: str = "application",
    status: int = 0,
    **extra: object,
) -> dict:
    return {
        "sequence": sequence,
        "pid": 7,
        "tid": 11,
        "event": name,
        "phase": phase,
        "origin": origin,
        "status": status,
        **extra,
    }


def terminal(
    sequence: int,
    *,
    written: int,
    dropped: int = 0,
    resolution: int = 0,
    writes: int = 0,
) -> dict:
    return {
        "sequence": sequence,
        "pid": 7,
        "tid": 11,
        "event": "terminal",
        "phase": "end",
        "status": 0,
        "events_written": written,
        "dropped_events": dropped,
        "resolution_errors": resolution,
        "write_errors": writes,
    }


def positive_trace(*, failure: bool = False) -> list[dict]:
    values = [
        event(0, "cuda_mem_get_info", observed=True, free_bytes=1000, total_bytes=2000),
        event(1, "cublas_create", phase="begin"),
        event(
            2,
            "cuda_mem_get_info",
            origin="provider_internal",
            observed=True,
            free_bytes=950,
            total_bytes=2000,
        ),
        event(3, "cublas_create"),
        event(4, "cublas_set_stream", phase="begin"),
        event(5, "cublas_set_stream"),
        event(6, "cublas_set_pointer_mode", phase="begin"),
        event(7, "cublas_set_pointer_mode", status=13 if failure else 0),
        event(8, "cusolver_create", phase="begin"),
        event(9, "cusolver_create"),
        event(10, "cusolver_set_stream", phase="begin"),
        event(11, "cusolver_set_stream"),
        event(12, "cusolver_create_params", phase="begin"),
        event(13, "cusolver_create_params"),
        event(14, "cuda_mem_get_info", observed=True, free_bytes=800, total_bytes=2000),
    ]
    return [*values, terminal(15, written=15)]


def provider_output(*, failure: bool = False) -> dict:
    return {
        "case": "provider_failure" if failure else "positive",
        "before_status": 0,
        "before_free": 1000,
        "before_total": 2000,
        "cublas_create_status": 0,
        "cublas_handle": 12628389,
        "cublas_stream_status": 0,
        "cublas_pointer_status": 13 if failure else 0,
        "cusolver_create_status": 0,
        "cusolver_handle": 787713,
        "cusolver_stream_status": 0,
        "cusolver_params_status": 0,
        "cusolver_params": 51786,
        "after_status": 0,
        "after_free": 800,
        "after_total": 2000,
        "audit": {
            "expected_calls": [3, 1, 1, 1, 1, 1, 1],
            "default_calls": [0, 0, 0, 0, 0, 0, 0],
            "argument_failures": 0,
        },
    }


def write_fixture(root: Path) -> None:
    root.joinpath("case-exits.tsv").write_text(
        "".join(f"{name}\t0\n" for name in CASES), encoding="utf-8"
    )
    for name in CASES:
        root.joinpath(f"{name}.stderr.txt").write_text("", encoding="utf-8")

    positive = provider_output()
    failure = provider_output(failure=True)
    for name in ("positive.baseline", "positive.shim", "o_excl.first", "o_excl.second"):
        write_json(root / f"{name}.stdout.json", positive)
    for name in ("provider_failure.baseline", "provider_failure.shim"):
        write_json(root / f"{name}.stdout.json", failure)
    write_json(root / "overflow.shim.stdout.json", {"case": "overflow"})
    write_json(
        root / "resolution_failure.shim.stdout.json",
        {"case": "resolution_failure", "lookup": "found", "status": 7, "parameters": 0},
    )
    write_json(root / "write_failure.shim.stdout.json", positive)
    write_json(
        root / "stdout-parity.json",
        {
            "positive": {
                "baseline": {"size": 10, "sha256": "a" * 64},
                "shim": {"size": 10, "sha256": "a" * 64},
                "byte_identical": True,
            },
            "provider_failure": {
                "baseline": {"size": 11, "sha256": "b" * 64},
                "shim": {"size": 11, "sha256": "b" * 64},
                "byte_identical": True,
            },
        },
    )

    write_jsonl(root / "positive.trace.jsonl", positive_trace())
    write_jsonl(root / "provider_failure.trace.jsonl", positive_trace(failure=True))
    write_jsonl(root / "o_excl.trace.jsonl", positive_trace())
    write_jsonl(
        root / "resolution_failure.trace.jsonl",
        [
            event(0, "resolution_error", origin="interposer", status=-1),
            event(1, "cusolver_create_params", phase="begin"),
            event(2, "cusolver_create_params", status=7),
            terminal(3, written=3, resolution=1),
        ],
    )
    overflow = [
        event(
            sequence,
            "cuda_mem_get_info",
            observed=True,
            free_bytes=800,
            total_bytes=2000,
        )
        for sequence in range(512)
    ]
    write_jsonl(
        root / "overflow.trace.jsonl",
        [*overflow, terminal(520, written=512, dropped=8)],
    )
    write_jsonl(
        root / "write_failure.trace.jsonl",
        [*positive_trace()[:-1][1:], terminal(15, written=15, writes=1)],
    )
    write_json(
        root / "o_excl.meta.json",
        {
            "mode": "600",
            "sha256_before": "a" * 64,
            "sha256_after": "a" * 64,
            "size_before": 123,
            "size_after": 123,
        },
    )

    driver_symbols = []
    provider_symbols = []
    shim_symbols = []
    for symbol, version in verify.TARGET_VERSIONS.items():
        driver_symbols.append(f"{symbol}@{version}")
        provider_symbols.extend((f"{symbol}@{version}", f"{symbol}@@FAKE_PROVIDER_9.9"))
        shim_symbols.append(f"{symbol}@@{version}")
    root.joinpath("driver.symbols.txt").write_text(
        "\n".join(driver_symbols), encoding="utf-8"
    )
    root.joinpath("fake_provider.symbols.txt").write_text(
        "\n".join(provider_symbols), encoding="utf-8"
    )
    root.joinpath("shim.symbols.txt").write_text(
        "\n".join(shim_symbols), encoding="utf-8"
    )

    write_json(
        root / "positive.report.json",
        {
            "status": "COMPLETE",
            "measurement": {
                "retained_delta_bytes": 200,
                "effective_allowance_bytes": 250,
                "exceeds_allowance": False,
                "provider_internal_mem_queries": 1,
            },
        },
    )
    root.joinpath("positive.report.exit.txt").write_text("0\n", encoding="utf-8")
    write_json(
        root / "identity.json",
        {**verify.PRODUCTION_IDENTITIES, "observer_commit": "f" * 40},
    )
    for name in (
        "source.sha256",
        "headers.sha256",
        "binaries.sha256",
        "compiler-version.txt",
        "launcher-version.txt",
        "launcher-stats-before.txt",
        "launcher-stats-after.txt",
        "compile-commands.txt",
    ):
        root.joinpath(name).write_text("evidence\n", encoding="utf-8")


def test_independent_verifier_accepts_hand_authored_complete_capture(
    tmp_path: Path,
) -> None:
    write_fixture(tmp_path)
    assert verify.verify_all(tmp_path)["outcome"] == "PASS"


def test_independent_verifier_rejects_default_version_forwarding(
    tmp_path: Path,
) -> None:
    write_fixture(tmp_path)
    value = json.loads(
        (tmp_path / "positive.shim.stdout.json").read_text(encoding="utf-8")
    )
    value["audit"]["expected_calls"][0] = 0
    value["audit"]["default_calls"][0] = 1
    write_json(tmp_path / "positive.shim.stdout.json", value)
    with pytest.raises(verify.VerificationError, match="baseline/shim output differs"):
        verify.verify_all(tmp_path)


def test_independent_verifier_requires_raw_stdout_parity(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    value = json.loads((tmp_path / "stdout-parity.json").read_text(encoding="utf-8"))
    value["positive"]["byte_identical"] = False
    write_json(tmp_path / "stdout-parity.json", value)
    with pytest.raises(verify.VerificationError, match="stdout is not byte-identical"):
        verify.verify_all(tmp_path)


def test_independent_verifier_rejects_missing_nested_origin(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    values = [
        json.loads(line)
        for line in (tmp_path / "positive.trace.jsonl").read_text().splitlines()
    ]
    values[2]["origin"] = "application"
    write_jsonl(tmp_path / "positive.trace.jsonl", values)
    with pytest.raises(verify.VerificationError, match="event protocol mismatch"):
        verify.verify_all(tmp_path)
