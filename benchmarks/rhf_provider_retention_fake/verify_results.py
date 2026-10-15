"""Independent verifier for the versioned RHF provider observer fake harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGET_VERSIONS = {
    "cudaMemGetInfo": "libcudart.so.12",
    "cublasCreate_v2": "libcublas.so.12",
    "cublasSetStream_v2": "libcublas.so.12",
    "cublasSetPointerMode_v2": "libcublas.so.12",
    "cusolverDnCreate": "libcusolver.so.11",
    "cusolverDnSetStream": "libcusolver.so.11",
    "cusolverDnCreateParams": "libcusolver.so.11",
}

PRODUCTION_IDENTITIES = {
    "runtime_source_commit": "a3088c5d2825de719123682facc36e8327a155e9",
    "production_library_sha256": (
        "e39cbf740cd81ace24c56fd0c1e7bdb2bcaeb8dc93aca02c7f37c70d20c0e001"
    ),
    "production_endpoint_sha256": (
        "19b34468267ef52174e14afb950cb80e08c33fd3d861b89b068819886120cc65"
    ),
}


class VerificationError(RuntimeError):
    """Raised when captured runtime evidence violates the hand-authored oracle."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == 1, f"{path.name} must contain exactly one JSON row")
    value = json.loads(lines[0])
    require(isinstance(value, dict), f"{path.name} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require(rows, f"{path.name} is empty")
    require(
        all(isinstance(row, dict) for row in rows), f"{path.name} has a non-object row"
    )
    return rows


def split_trace(path: Path) -> tuple[list[dict], dict]:
    rows = load_jsonl(path)
    terminals = [row for row in rows if row.get("event") == "terminal"]
    require(len(terminals) == 1, f"{path.name} requires exactly one terminal")
    require(rows[-1] is terminals[0], f"{path.name} terminal must be last")
    return rows[:-1], rows[-1]


def verify_clean_terminal(ordinary: list[dict], terminal: dict, name: str) -> None:
    require(
        [row["sequence"] for row in ordinary] == list(range(len(ordinary))),
        f"{name} ordinary event sequence is not contiguous",
    )
    require(terminal["sequence"] == len(ordinary), f"{name} terminal sequence mismatch")
    require(terminal["events_written"] == len(ordinary), f"{name} write count mismatch")
    for field in ("dropped_events", "resolution_errors", "write_errors"):
        require(terminal[field] == 0, f"{name} unexpectedly reports {field}")


def signature(row: dict) -> tuple[str, str, str]:
    return row["event"], row["phase"], row.get("origin", "")


def expected_positive_signatures() -> list[tuple[str, str, str]]:
    return [
        ("cuda_mem_get_info", "end", "application"),
        ("cublas_create", "begin", "application"),
        ("cuda_mem_get_info", "end", "provider_internal"),
        ("cublas_create", "end", "application"),
        ("cublas_set_stream", "begin", "application"),
        ("cublas_set_stream", "end", "application"),
        ("cublas_set_pointer_mode", "begin", "application"),
        ("cublas_set_pointer_mode", "end", "application"),
        ("cusolver_create", "begin", "application"),
        ("cusolver_create", "end", "application"),
        ("cusolver_set_stream", "begin", "application"),
        ("cusolver_set_stream", "end", "application"),
        ("cusolver_create_params", "begin", "application"),
        ("cusolver_create_params", "end", "application"),
        ("cuda_mem_get_info", "end", "application"),
    ]


def verify_positive_trace(path: Path, *, provider_failure: bool = False) -> None:
    ordinary, terminal = split_trace(path)
    verify_clean_terminal(ordinary, terminal, path.name)
    require(
        [signature(row) for row in ordinary] == expected_positive_signatures(),
        f"{path.name} event protocol mismatch",
    )
    nonzero = [row for row in ordinary if row["status"] != 0]
    if provider_failure:
        require(len(nonzero) == 1, "provider failure case requires one failing event")
        require(
            nonzero[0]["event"] == "cublas_set_pointer_mode"
            and nonzero[0]["phase"] == "end",
            "provider failure was attributed to the wrong event",
        )
    else:
        require(not nonzero, f"{path.name} contains an unexpected provider failure")
    nested = [row for row in ordinary if row.get("origin") == "provider_internal"]
    require(len(nested) == 1, f"{path.name} requires one nested provider memory query")
    require(
        (ordinary[0]["free_bytes"], nested[0]["free_bytes"], ordinary[-1]["free_bytes"])
        == (1000, 950, 800),
        f"{path.name} memory sentinels do not identify the expected version",
    )


def verify_outputs(root: Path) -> None:
    positive_baseline = load_json(root / "positive.baseline.stdout.json")
    positive_shim = load_json(root / "positive.shim.stdout.json")
    require(positive_baseline == positive_shim, "positive baseline/shim output differs")
    require(positive_shim["before_free"] == 1000, "positive before output mismatch")
    require(positive_shim["after_free"] == 800, "positive after output mismatch")
    require(positive_shim["before_total"] == 2000, "positive total output mismatch")
    require(positive_shim["after_total"] == 2000, "positive total changed")
    status_fields = [key for key in positive_shim if key.endswith("_status")]
    require(
        all(positive_shim[key] == 0 for key in status_fields),
        "positive status was not forwarded exactly",
    )
    require(
        positive_shim["audit"]
        == {
            "expected_calls": [3, 1, 1, 1, 1, 1, 1],
            "default_calls": [0, 0, 0, 0, 0, 0, 0],
            "argument_failures": 0,
        },
        "positive provider audit did not use exact versions and arguments",
    )

    failure_baseline = load_json(root / "provider_failure.baseline.stdout.json")
    failure_shim = load_json(root / "provider_failure.shim.stdout.json")
    require(failure_baseline == failure_shim, "failure baseline/shim output differs")
    require(
        failure_shim["cublas_pointer_status"] != 0,
        "provider failure case did not preserve the failing status",
    )
    require(
        failure_shim["audit"]["default_calls"] == [0, 0, 0, 0, 0, 0, 0]
        and failure_shim["audit"]["argument_failures"] == 0,
        "provider failure used a default version or changed arguments",
    )

    require(
        load_json(root / "o_excl.first.stdout.json") == positive_baseline,
        "O_EXCL first run changed forwarded output",
    )
    require(
        load_json(root / "o_excl.second.stdout.json") == positive_baseline,
        "O_EXCL second run changed forwarded output",
    )


def verify_traces(root: Path) -> None:
    verify_positive_trace(root / "positive.trace.jsonl")
    verify_positive_trace(root / "provider_failure.trace.jsonl", provider_failure=True)
    verify_positive_trace(root / "o_excl.trace.jsonl")

    resolution, terminal = split_trace(root / "resolution_failure.trace.jsonl")
    require(
        [signature(row) for row in resolution]
        == [
            ("resolution_error", "end", "interposer"),
            ("cusolver_create_params", "begin", "application"),
            ("cusolver_create_params", "end", "application"),
        ],
        "resolution failure event protocol mismatch",
    )
    require(resolution[-1]["status"] != 0, "resolution failure returned success")
    require(terminal["resolution_errors"] == 1, "resolution error counter mismatch")
    require(terminal["write_errors"] == 0, "resolution case write failed")
    require(terminal["dropped_events"] == 0, "resolution case dropped events")

    overflow, terminal = split_trace(root / "overflow.trace.jsonl")
    require(len(overflow) == 512, "overflow journal did not retain exactly 512 events")
    require(
        [row["sequence"] for row in overflow] == list(range(512)),
        "overflow retained sequences are not bounded and contiguous",
    )
    require(
        all(row["event"] == "cuda_mem_get_info" for row in overflow),
        "overflow event mismatch",
    )
    require(terminal["sequence"] == 520, "overflow terminal sequence mismatch")
    require(terminal["events_written"] == 512, "overflow written count mismatch")
    require(terminal["dropped_events"] == 8, "overflow loss counter mismatch")
    require(terminal["resolution_errors"] == 0, "overflow resolution failed")
    require(terminal["write_errors"] == 0, "overflow write failed")

    write_fault, terminal = split_trace(root / "write_failure.trace.jsonl")
    require(
        [row["sequence"] for row in write_fault] == list(range(1, 15)),
        "write failure did not preserve the expected sequence gap",
    )
    require(terminal["sequence"] == 15, "write failure terminal sequence mismatch")
    require(terminal["events_written"] == 15, "write failure attempted count mismatch")
    require(terminal["write_errors"] == 1, "write failure counter mismatch")
    require(terminal["dropped_events"] == 0, "write failure dropped events")
    require(terminal["resolution_errors"] == 0, "write failure resolution failed")


def verify_o_excl(root: Path) -> None:
    metadata = json.loads((root / "o_excl.meta.json").read_text(encoding="utf-8"))
    require(metadata["mode"] == "600", "observer journal mode is not 0600")
    require(
        metadata["sha256_before"] == metadata["sha256_after"], "O_EXCL file changed"
    )
    require(
        metadata["size_before"] == metadata["size_after"], "O_EXCL file size changed"
    )


def verify_symbols(root: Path) -> None:
    driver = (root / "driver.symbols.txt").read_text(encoding="utf-8")
    provider = (root / "fake_provider.symbols.txt").read_text(encoding="utf-8")
    shim = (root / "shim.symbols.txt").read_text(encoding="utf-8")
    for symbol, version in TARGET_VERSIONS.items():
        require(f"{symbol}@{version}" in driver, f"driver lacks {symbol}@{version}")
        require(
            f"{symbol}@{version}" in provider, f"fake provider lacks {symbol}@{version}"
        )
        require(
            f"{symbol}@@FAKE_PROVIDER_9.9" in provider,
            f"fake provider lacks different default for {symbol}",
        )
        require(
            f"{symbol}@@{version}" in shim,
            f"shim lacks default export {symbol}@@{version}",
        )


def verify_report(root: Path) -> None:
    require(
        (root / "positive.report.exit.txt").read_text(encoding="utf-8").strip() == "0",
        "positive report command did not exit successfully",
    )
    report = json.loads((root / "positive.report.json").read_text(encoding="utf-8"))
    require(report["status"] == "COMPLETE", "positive report is not COMPLETE")
    measurement = report["measurement"]
    require(measurement is not None, "positive report omitted measurement")
    require(
        measurement["retained_delta_bytes"] == 200, "positive retained delta mismatch"
    )
    require(
        measurement["effective_allowance_bytes"] == 250, "positive allowance mismatch"
    )
    require(
        not measurement["exceeds_allowance"], "positive fake delta exceeds allowance"
    )
    require(
        measurement["provider_internal_mem_queries"] == 1, "nested query count mismatch"
    )


def verify_provenance(root: Path) -> None:
    identity = json.loads((root / "identity.json").read_text(encoding="utf-8"))
    for name, value in PRODUCTION_IDENTITIES.items():
        require(identity.get(name) == value, f"production identity mismatch: {name}")
    observer_commit = identity.get("observer_commit", "")
    require(
        isinstance(observer_commit, str)
        and len(observer_commit) == 40
        and all(character in "0123456789abcdef" for character in observer_commit),
        "observer commit is not a full Git identity",
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
        require(
            (root / name).stat().st_size > 0, f"missing provenance evidence: {name}"
        )


def verify_exits(root: Path) -> None:
    rows = [
        line.split("\t") for line in (root / "case-exits.tsv").read_text().splitlines()
    ]
    expected = {
        "positive.baseline",
        "positive.shim",
        "provider_failure.baseline",
        "provider_failure.shim",
        "overflow.shim",
        "resolution_failure.shim",
        "o_excl.first",
        "o_excl.second",
        "write_failure.shim",
    }
    require({row[0] for row in rows} == expected, "case exit ledger is incomplete")
    require(
        all(len(row) == 2 and row[1] == "0" for row in rows), "a harness case failed"
    )
    for name in expected:
        require(
            not (root / f"{name}.stderr.txt").read_text(encoding="utf-8"),
            f"{name} wrote unexpected stderr",
        )


def verify_all(root: Path) -> dict:
    verify_exits(root)
    verify_outputs(root)
    verify_traces(root)
    verify_o_excl(root)
    verify_symbols(root)
    verify_report(root)
    verify_provenance(root)
    return {
        "schema": "generativeqc.rhf-provider-fake-verification.v1",
        "outcome": "PASS",
        "scope": "fake ELF forwarding and observer protocol only; no CUDA device or science",
        "versioned_targets": TARGET_VERSIONS,
        "cases": [
            "positive baseline parity",
            "provider failure parity",
            "nested provider memory origin",
            "resolution failure",
            "O_EXCL and 0600 mode",
            "512-event bound and loss",
            "single write failure and terminal counters",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    arguments = parser.parse_args()
    output = arguments.result_directory / "verification.json"
    try:
        result = verify_all(arguments.result_directory)
    except (OSError, ValueError, KeyError, VerificationError) as error:
        result = {
            "schema": "generativeqc.rhf-provider-fake-verification.v1",
            "outcome": "FAIL",
            "error": str(error),
            "scope": "fake ELF forwarding and observer protocol only; no CUDA device or science",
        }
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 1
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
