"""Offline contracts for the bounded no-upload same-runner diagnostic."""

from __future__ import annotations

import argparse
import copy
import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[2]
BASE_SPEC = importlib.util.spec_from_file_location(
    "codspeed_baseline", ROOT / "tools/codspeed_baseline.py"
)
assert BASE_SPEC is not None and BASE_SPEC.loader is not None
baseline = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(baseline)
sys.modules["codspeed_baseline"] = baseline
SPEC = importlib.util.spec_from_file_location(
    "codspeed_pair", ROOT / "tools/codspeed_pair.py"
)
assert SPEC is not None and SPEC.loader is not None
pair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pair)
EVENTS = "Ir Dr Dw I1mr D1mr D1mw ILmr DLmr DLmw sysCount sysTime sysCpuTime Ct Cl"
VALUES = "100 20 10 1 2 3 4 5 6 7 8 9 1100 1200"


def _part(uri: str, index: int = 1, values: str = VALUES) -> str:
    return (
        f"part: {index}\ndesc: Trigger: Client Request: {uri}\n"
        f"events: {EVENTS}\nsummary: {values}\nfn=(1) benchmark\n"
        f"1 {values}\ntotals: {values}\n"
    )


def _profile(folder: Path, uris: tuple[str, ...] = pair.URIS) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "123.out"
    path.write_text(
        _part("Metadata: pytest-codspeed 5.0.3", 0)
        + "".join(_part(uri, index + 1) for index, uri in enumerate(uris))
        + "part: 6\ndesc: Trigger: Program termination\nevents: Ir\ntotals: 99999\n"
    )
    return path


def test_parse_measured_parts_not_metadata_or_cumulative_costs(tmp_path: Path) -> None:
    _profile(tmp_path)
    records = pair.parse_arm(tmp_path)
    assert [x["uri"] for x in records] == list(pair.URIS)
    assert all(x["totals"]["Ir"] == 100 for x in records)
    assert all(x["totals"]["Ct"] == 1100 for x in records)
    assert all(x["summary"] == x["totals"] for x in records)


def test_trailing_zero_counters_are_padded_without_losing_unknown_events() -> None:
    assert pair.counters(["Ir", "Ct", "Cl", "future"], "100 200 300") == {
        "Ir": 100,
        "Ct": 200,
        "Cl": 300,
        "future": 0,
    }


@pytest.mark.parametrize("text", ["1 2 3", "-1", "1.2", "NaN", "inf", "a"])
def test_malformed_counter_rows_fail_closed(text: str) -> None:
    with pytest.raises(ValueError):
        pair.counters(["Ir", "Ct"], text)


@pytest.mark.parametrize(
    "uris", [pair.URIS[:-1], pair.URIS[::-1], (*pair.URIS, pair.URIS[0]), ()]
)
def test_missing_reordered_duplicate_and_empty_suites_rejected(
    tmp_path: Path, uris: tuple[str, ...]
) -> None:
    _profile(tmp_path, uris)
    with pytest.raises(ValueError):
        pair.parse_arm(tmp_path)


def test_split_process_history_rejected(tmp_path: Path) -> None:
    (tmp_path / "1.out").write_text(_part(pair.URIS[0]))
    (tmp_path / "2.out").write_text("".join(_part(uri) for uri in pair.URIS[1:]))
    with pytest.raises(ValueError):
        pair.parse_arm(tmp_path)


@pytest.mark.parametrize(
    "edit",
    [
        lambda s: s.replace("totals:", "not-totals:"),
        lambda s: s.replace("events: Ir", "events: Ir Ir"),
        lambda s: s.replace("Ct Cl", "Ct Other"),
        lambda s: s.replace(VALUES, "0 20 10 1 2 3 4 5 6 7 8 9 1100 1200"),
        lambda s: s.replace(pair.URIS[0], "benchmarks/unexpected.py::test_other"),
        lambda s: s + "totals: " + VALUES + "\n",
        lambda s: s.replace(pair.URIS[0], "Metadata: unknown-plugin 9.0"),
        lambda s: s.replace("Client Request: " + pair.URIS[0], "unknown trigger"),
        lambda s: s.replace("desc: Trigger:", "unknown:"),
    ],
)
def test_unproven_profile_measurement_rejected(
    tmp_path: Path, edit: Callable[[str], str]
) -> None:
    path = tmp_path / "1.out"
    path.write_text(edit(_part(pair.URIS[0])))
    with pytest.raises(ValueError):
        pair.parse_profile(path)


def test_two_pairs_report_separately_without_backend_score(tmp_path: Path) -> None:
    records = pair.parse_profile(_profile(tmp_path))
    arms = {name: copy.deepcopy(records) for name in pair.ARMS}
    arms["head-1"][0]["totals"]["Ir"] = 110
    arms["head-2"][0]["totals"]["Ir"] = 120
    result = pair.comparison(arms)[pair.URIS[0]]["Ir"]
    assert [x["cost_change_percent"] for x in result["pairs"]] == pytest.approx(
        [10, 20]
    )
    assert result["pairs"][0]["inverse_cost_change_percent"] == pytest.approx(-100 / 11)
    assert result["repeat_spread_percent"]["head"] == pytest.approx(100 / 11)
    assert result["repeat_spread_percent"]["base"] == 0


def test_zero_counter_ratio_is_undefined_not_infinite(tmp_path: Path) -> None:
    records = pair.parse_profile(_profile(tmp_path))
    arms = {name: copy.deepcopy(records) for name in pair.ARMS}
    for records in arms.values():
        records[0]["totals"]["D1mr"] = 0
    result = pair.comparison(arms)[pair.URIS[0]]["D1mr"]
    assert result["pairs"][0]["cost_change_percent"] is None
    json.dumps(result, allow_nan=False)


def test_arm_command_keeps_allocation_cost_and_disables_upload(tmp_path: Path) -> None:
    command = pair.arm_command(
        tmp_path / ".venv/bin/python", tmp_path, tmp_path / "profile"
    )
    assert "--skip-upload" in command and "--skip-setup" in command
    assert "--exclude-allocations=false" in command
    assert "--cycle-estimation=true" in command
    assert "--simulation-tool=callgrind" in command
    assert command[-1].endswith(
        "-m pytest benchmarks/test_cpu_codspeed.py --codspeed -q"
    )
    assert not any(x in command for x in ("--base", "--token", "--allow-empty"))


def _build(root: Path) -> None:
    build = root / "build-benchmark"
    (build / "CMakeFiles").mkdir(parents=True, exist_ok=True)
    (build / "CMakeCache.txt").write_text(
        "".join(f"{k}:STRING={v}\n" for k, v in pair.BUILD_OPTIONS.items())
        + f"CMAKE_HOME_DIRECTORY:INTERNAL={root}\n"
    )
    (build / "CMakeFiles/rules.ninja").write_text(
        "rule CXX_COMPILER__generativeqc_Release\n  command = /usr/bin/ccache /usr/bin/c++\n"
    )
    (build / "libgenerativeqc.so.0.1").write_bytes(b"fixture library")
    if (build / "libgenerativeqc.so").is_symlink():
        (build / "libgenerativeqc.so").unlink()
    (build / "libgenerativeqc.so").symlink_to("libgenerativeqc.so.0.1")


def test_verified_build_accepts_only_in_build_soname_link(tmp_path: Path) -> None:
    _build(tmp_path)
    verified = pair.verify_build(tmp_path)
    assert verified["resolved_library"].endswith("libgenerativeqc.so.0.1")
    (tmp_path / "build-benchmark/libgenerativeqc.so").unlink()
    (tmp_path / "outside.so").write_bytes(b"wrong")
    (tmp_path / "build-benchmark/libgenerativeqc.so").symlink_to(
        tmp_path / "outside.so"
    )
    with pytest.raises(ValueError):
        pair.verify_build(tmp_path)


@pytest.mark.parametrize("field", list(pair.BUILD_OPTIONS))
def test_changed_build_options_rejected(tmp_path: Path, field: str) -> None:
    _build(tmp_path)
    p = tmp_path / "build-benchmark/CMakeCache.txt"
    p.write_text(
        p.read_text().replace(
            f"{field}:STRING={pair.BUILD_OPTIONS[field]}", f"{field}:STRING=wrong"
        )
    )
    with pytest.raises(ValueError):
        pair.verify_build(tmp_path)


def test_raw_retention_is_lossless_bounded_and_allowlisted(tmp_path: Path) -> None:
    source, dest = tmp_path / "profiles", tmp_path / "artifact"
    source.mkdir()
    (source / "123.out").write_bytes(b"retained raw bytes\n")
    (source / "valgrind.123.log").write_bytes(b"log")
    (source / "runner.log").write_bytes(b"do not publish arbitrary config")
    (source / "environment-123.json").write_bytes(b"not allowlisted")
    rows, remaining = pair.preserve_profiles(source, dest, 100)
    assert {x["file"] for x in rows} == {"123.out.gz", "valgrind.123.log.gz"}
    assert (
        gzip.decompress((dest / "123.out.gz").read_bytes()) == b"retained raw bytes\n"
    )
    assert remaining == 78
    with pytest.raises(ValueError):
        pair.preserve_profiles(source, tmp_path / "small", 1)


@pytest.mark.parametrize(
    "bad",
    [
        "skip-upload",
        "threads",
        "selector",
        "ancestry",
        "fingerprint",
        "failed-arm",
        "pycache",
    ],
)
def test_diagnostic_execution_is_bounded_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    result = _fake_execute(tmp_path, monkeypatch, bad)
    assert result[0] == 1
    assert result[1]["performance_clearance"] is False
    assert result[1]["backend_score_available"] is False
    assert "comparisons" not in result[1]
    assert len(result[2]) <= 1


def _fake_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str = ""
) -> tuple[int, dict[str, object], list[str]]:
    root = tmp_path / "head"
    root.mkdir()
    (root / "benchmarks").mkdir()
    (root / pair.BENCHMARK).write_text("unchanged benchmark fixture")
    _build(root)
    monkeypatch.chdir(root)
    monkeypatch.setenv(
        "CODSPEED_SKIP_UPLOAD", "false" if bad == "skip-upload" else "true"
    )
    for key in pair.THREAD_VARIABLES:
        monkeypatch.setenv(key, "2" if bad == "threads" else "1")
    selected = {
        "tier": "full" if bad == "selector" else "pr",
        "extra_cases": ["wb97mv"],
        "source_sha256": pair.sha256(root / pair.BENCHMARK),
    }
    monkeypatch.setattr(baseline, "benchmark_selection", lambda: selected)

    def authenticate(*args: str) -> str:
        if bad == "ancestry":
            raise ValueError("wrong merge")
        return "a" * 40

    monkeypatch.setattr(baseline, "tested_master_base", authenticate)
    calls = []
    fingerprints = 0

    def fingerprint() -> dict[str, object]:
        nonlocal fingerprints
        fingerprints += 1
        return {
            "cpu": {
                "vendor": "test",
                "model": "changed"
                if bad == "fingerprint" and fingerprints > 1
                else "cpu",
                "flags": "test",
            }
        }

    monkeypatch.setattr(baseline, "environment_fingerprint", fingerprint)
    current_head = "b" * 40

    def run(command: list[str], **kwargs: object) -> None:
        nonlocal current_head
        assert 0 < kwargs["timeout"] <= pair.ARM_SECONDS
        handle = kwargs["stdout"]
        if command[:2] == ["ccache", "--version"]:
            handle.write("ccache version 4.14\n")
        elif command[:2] == ["codspeed", "--version"]:
            handle.write("codspeed 5.0.1\n")
        elif command[:2] == ["git", "checkout"]:
            if bad == "restore" and len(calls) == 4 and command[-1] == "b" * 40:
                raise subprocess.CalledProcessError(1, command)
            current_head = command[-1]
        elif command[:2] == ["git", "rev-parse"]:
            handle.write(current_head + "\n")
        elif command[:2] == ["cmake", "--build"]:
            _build(kwargs["cwd"])
            (root / "build-benchmark/libgenerativeqc.so.0.1").write_bytes(
                b"baseline fixture library"
            )
        elif command[:2] == ["codspeed", "run"]:
            profile = Path(command[command.index("--profile-folder") + 1])
            calls.append(profile.name)
            assert kwargs["env"]["CODSPEED_SKIP_UPLOAD"] == "true"
            assert kwargs["env"]["PYTHONPATH"] == str(kwargs["cwd"] / "python")
            assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
            pycache = Path(kwargs["env"]["PYTHONPYCACHEPREFIX"])
            assert pycache == tmp_path / "diagnostic/empty-pycache"
            assert not list(pycache.iterdir())
            if bad == "pycache":
                (pycache / "unexpected.pyc").write_bytes(b"unexpected cache write")
            assert kwargs["env"]["GENERATIVEQC_LIBRARY"] == str(
                tmp_path / "diagnostic/load/libgenerativeqc.so"
            )
            if bad == "failed-arm":
                raise subprocess.CalledProcessError(1, command)
            _profile(profile)

    monkeypatch.setattr(pair, "run_bounded", run)
    args = argparse.Namespace(
        merge_sha="b" * 40,
        head_sha="c" * 40,
        base_ref="master",
        output=tmp_path / "diagnostic",
    )
    code = pair.execute(args)
    report = json.loads((args.output / "artifact/report.json").read_text())
    if report.get("restored_tested_checkout"):
        assert current_head == "b" * 40
    return code, report, calls


def test_complete_stubbed_execution_runs_exact_abba_without_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, report, calls = _fake_execute(tmp_path, monkeypatch)
    assert code == 0 and calls == list(pair.ARMS)
    assert (
        report["performance_clearance"] is False
        and report["backend_score_available"] is False
    )
    assert len(report["comparisons"]) == 5
    assert all(len(x["measurements"]) == 5 for x in report["arms"].values())
    assert report["provenance"]["actual_base"] == "a" * 40
    assert report["restored_tested_checkout"] is True
    assert (
        report["builds"]["base"]["library_sha256"]
        != report["builds"]["head"]["library_sha256"]
    )


def test_failed_checkout_restoration_is_a_diagnostic_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, report, calls = _fake_execute(tmp_path, monkeypatch, "restore")
    assert code == 1 and calls == list(pair.ARMS)
    assert "restore_error" in report
    assert report["performance_clearance"] is False


def test_no_benchmark_definition_or_five_case_order_change() -> None:
    assert (
        pair.sha256(ROOT / pair.BENCHMARK)
        == "4b128805eb00b5189479196ddd8ba9a952e3465104e01e98ad94242d968946ea"
    )
    assert len(pair.URIS) == 5


def test_timeout_only_terminates_the_owned_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits = 0
    signals = []

    class Process:
        pid = 1234

        def wait(self, timeout: float) -> int:
            nonlocal waits
            waits += 1
            if waits < 3:
                raise subprocess.TimeoutExpired(["owned-child"], timeout)
            return -9

    def start(command: list[str], **kwargs: object) -> Process:
        assert kwargs["start_new_session"] is True
        return Process()

    monkeypatch.setattr(pair.subprocess, "Popen", start)
    monkeypatch.setattr(pair.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with (tmp_path / "log").open("w") as log, pytest.raises(subprocess.TimeoutExpired):
        pair.run_bounded(["owned-child"], cwd=tmp_path, stdout=log, timeout=1)
    assert signals == [(1234, pair.signal.SIGTERM), (1234, pair.signal.SIGKILL)]
