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


def _receipt() -> dict[str, object]:
    return {
        "schema": baseline._SCHEMA,
        "sha": SHA,
        "environment": _environment(),
    }


def test_exact_environment_qualifies() -> None:
    assert baseline.qualify(_receipt(), SHA, _environment())[0]


@pytest.mark.parametrize(
    "key",
    ["cpu", "libc", "python", "packages", "compiler", "simulation"],
)
def test_environment_mismatch_is_not_comparable(key: str) -> None:
    altered = _environment()
    altered[key] = "different"
    valid, reason = baseline.qualify(_receipt(), SHA, altered)
    assert not valid
    assert key in reason


def test_stale_base_missing_environment_or_unknown_schema_rejected() -> None:
    assert not baseline.qualify(_receipt(), "b" * 40, _environment())[0]
    assert not baseline.qualify({"sha": SHA}, SHA, _environment())[0]
    bad = _receipt()
    bad["schema"] = "legacy"
    assert not baseline.qualify(bad, SHA, _environment())[0]


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
    assert not baseline.qualify({**_receipt(), "environment": current}, SHA, current)[0]
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
