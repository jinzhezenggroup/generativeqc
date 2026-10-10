"""Validate exact-base, environment-matched CodSpeed comparison policy."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "codspeed_baseline", ROOT / "tools/codspeed_baseline.py"
)
assert SPEC is not None and SPEC.loader is not None
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)
SHA = "a" * 40


def _environment() -> dict[str, object]:
    return {
        "cpu": {
            "vendor": "GenuineIntel",
            "model": "Xeon Test",
            "flags": "avx2 sse4_2",
        },
        "machine": "x86_64",
        "libc": ["glibc", "2.39"],
        "python": "3.11.14",
        "compiler": "g++ (Ubuntu) 13",
        "ccache": "4.14",
        "packages": {"numpy": "2", "pytest": "9", "pytest-codspeed": "5"},
        "simulation": "CodSpeedHQ/action-v5.0.1/pytest-codspeed-5.0.3",
    }


def _selection(*extras: str) -> dict[str, object]:
    return {"source_sha256": "a" * 64, "tier": "pr", "extra_cases": list(extras)}


def _receipt() -> dict[str, object]:
    return {
        "schema": baseline._SCHEMA,
        "sha": SHA,
        "environment": _environment(),
        "benchmark_selection": _selection("wb97mv"),
    }


def test_exact_environment_qualifies() -> None:
    assert baseline.qualify(_receipt(), SHA, _environment(), _selection("wb97mv"))[0]


@pytest.mark.parametrize(
    "key",
    ["cpu", "libc", "python", "packages", "compiler", "simulation"],
)
def test_environment_mismatch_is_not_comparable(key: str) -> None:
    altered = _environment()
    altered[key] = "different"
    valid, reason = baseline.qualify(_receipt(), SHA, altered, _selection("wb97mv"))
    assert not valid
    assert key in reason


def test_stale_base_missing_environment_or_unknown_schema_rejected() -> None:
    assert not baseline.qualify(_receipt(), "b" * 40, _environment(), _selection())[0]
    assert not baseline.qualify({"sha": SHA}, SHA, _environment(), _selection())[0]
    bad = _receipt()
    bad["schema"] = "legacy"
    assert not baseline.qualify(bad, SHA, _environment(), _selection())[0]


def test_cpu_identity_ignores_processor_number_and_sorts_features() -> None:
    cpu = baseline._cpu_identity(
        "processor: 0\nvendor_id: GenuineIntel\n"
        "model name: Xeon Test\nflags: sse4_2 avx2\n\n"
        "processor: 1\nvendor_id: GenuineIntel\n"
        "model name: Xeon Test\nflags: avx2 sse4_2\n"
    )
    assert cpu == {
        "vendor": "GenuineIntel",
        "model": "Xeon Test",
        "flags": "avx2 sse4_2",
    }


def test_unknown_cpu_and_bad_sha_rejected() -> None:
    current = _environment()
    current["cpu"] = {"vendor": "", "model": "", "flags": ""}
    assert not baseline.qualify(
        {**_receipt(), "environment": current}, SHA, current, _selection("wb97mv")
    )[0]
    with pytest.raises(ValueError):
        baseline._baseline_name("bad-hash")


def test_lookup_selects_exact_master_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = baseline._baseline_name(SHA)
    exact = {
        "name": name,
        "expired": False,
        "created_at": "2026-10-08T12:00:00Z",
        "workflow_run": {
            "head_sha": SHA,
            "head_branch": "master",
            "id": 42,
        },
    }
    unrelated = {
        "name": name,
        "expired": False,
        "created_at": "2026-10-08T13:00:00Z",
        "workflow_run": {
            "head_sha": "b" * 40,
            "head_branch": "master",
            "id": 99,
        },
    }

    def fake_command(*args: str) -> str:
        if args[1] == "api":
            assert "-X" in args and "GET" in args
            return json.dumps({"artifacts": [unrelated, exact]})
        assert args[:4] == ("gh", "run", "download", "42")
        location = Path(args[args.index("-D") + 1]) / "baseline.json"
        location.write_text(json.dumps(_receipt()), encoding="utf-8")
        return ""

    monkeypatch.setattr(baseline, "_command", fake_command)
    assert baseline._lookup_baseline("jinzhezenggroup/generativeqc", SHA) == (
        _receipt()
    )


def test_missing_or_wrong_branch_artifact_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        baseline,
        "_command",
        lambda *args: json.dumps(
            {
                "artifacts": [
                    {
                        "name": baseline._baseline_name(SHA),
                        "expired": False,
                        "workflow_run": {
                            "head_sha": SHA,
                            "head_branch": "feature",
                            "id": 3,
                        },
                    }
                ]
            }
        ),
    )
    with pytest.raises(LookupError):
        baseline._lookup_baseline("jinzhezenggroup/generativeqc", SHA)


def test_unqualified_writes_neutral_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "step-output"
    summary = tmp_path / "step-summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    baseline._output(False, "different benchmark environments: cpu")
    assert output.read_text(encoding="utf-8") == "qualified=false\n"
    assert "unqualified" in summary.read_text(encoding="utf-8")


def test_existing_v2_master_selection_still_qualifies() -> None:
    assert baseline.qualify(_receipt(), SHA, _environment(), _selection("wb97mv"))[0]
    receipt = _receipt()
    receipt["benchmark_selection"] = _selection()
    valid, reason = baseline.qualify(receipt, SHA, _environment(), _selection("wb97mv"))
    assert not valid
    assert "selection" in reason


@pytest.mark.parametrize(
    ("recorded", "requested"),
    [((), ("wb97mv",)), (("wb97mv",), ())],
)
def test_selection_requires_identical_execution_history(
    recorded: tuple[str, ...], requested: tuple[str, ...]
) -> None:
    receipt = {**_receipt(), "benchmark_selection": _selection(*recorded)}
    valid, reason = baseline.qualify(
        receipt, SHA, _environment(), _selection(*requested)
    )
    assert not valid
    assert "selection" in reason


@pytest.mark.parametrize(
    "extras",
    [
        ["future"],
        ["wb97mv", "future"],
        [""],
        [" wb97mv "],
        ["wb97mv", "wb97mv"],
        [None],
        "wb97mv",
    ],
)
def test_matching_unknown_or_malformed_extras_are_rejected(extras: object) -> None:
    selection = {**_selection(), "extra_cases": extras}
    receipt = {**_receipt(), "benchmark_selection": selection}
    valid, reason = baseline.qualify(receipt, SHA, _environment(), selection)
    assert not valid
    assert "selection" in reason


@pytest.mark.parametrize(
    "recorded",
    [
        None,
        {},
        {"source_sha256": "a" * 64, "tier": "pr"},
        {"source_sha256": "b" * 64, "tier": "pr", "extra_cases": ["wb97mv"]},
        {"source_sha256": "a" * 64, "tier": "full", "extra_cases": ["wb97mv"]},
        {"source_sha256": "a" * 64, "tier": "pr", "extra_cases": [None]},
    ],
)
def test_unproven_benchmark_selection_rejected(recorded: object) -> None:
    receipt = _receipt()
    receipt["benchmark_selection"] = recorded
    assert not baseline.qualify(receipt, SHA, _environment(), _selection())[0]


def test_selection_binds_real_source_and_normalizes_extras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    monkeypatch.setenv("GENERATIVEQC_CODSPEED_TIER", "pr")
    monkeypatch.setenv("GENERATIVEQC_CODSPEED_EXTRA_CASES", " wb97mv,wb97mv, ")
    selected = baseline.benchmark_selection()
    assert selected == {
        "source_sha256": hashlib.sha256(
            (ROOT / "benchmarks/test_cpu_codspeed.py").read_bytes()
        ).hexdigest(),
        "tier": "pr",
        "extra_cases": ["wb97mv"],
    }


@pytest.mark.parametrize(
    "failure", [LookupError(), OSError(), ValueError(), TypeError()]
)
def test_lookup_failure_remains_advisory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: Exception
) -> None:
    import sys

    def unavailable(*args: str) -> dict[str, object]:
        raise failure

    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(
        sys, "argv", ["qualifier", "qualify", "--base-sha", SHA, "--repo", "owner/repo"]
    )
    monkeypatch.setattr(baseline, "environment_fingerprint", _environment)
    monkeypatch.setattr(baseline, "_lookup_baseline", unavailable)
    baseline.main()
    assert output.read_text() == "qualified=false\n"


def test_record_includes_selection_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    output = tmp_path / "baseline.json"
    monkeypatch.setattr(
        sys, "argv", ["qualifier", "record", "--sha", SHA, "--output", str(output)]
    )
    monkeypatch.setattr(baseline, "environment_fingerprint", _environment)
    monkeypatch.setattr(baseline, "benchmark_selection", lambda: _selection("wb97mv"))
    baseline.main()
    assert json.loads(output.read_text()) == _receipt()
