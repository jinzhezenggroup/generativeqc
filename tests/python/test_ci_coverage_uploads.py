"""Guard coverage completeness and the upload-only OIDC trust boundary."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ci.yml"


def _job() -> str:
    return (
        WORKFLOW.read_text()
        .split("\n  upload-coverage:\n", 1)[1]
        .split("\n  native-cli-artifact:\n", 1)[0]
    )


def _steps() -> list[str]:
    return re.split(
        r"^      - ", _job().split("    steps:\n", 1)[1], flags=re.MULTILINE
    )[1:]


def _field(block: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}: (.+?)(?: #.*)?$", block, re.MULTILINE)
    assert match, f"Missing {key}"
    return match[1]


def test_coverage_batches_match_every_producer_without_merging_reports() -> None:
    workflow = WORKFLOW.read_text()
    python = workflow.split("\n  python:\n", 1)[1].split("    steps:\n", 1)[0]
    shards = re.search(r"shard: \[([^\]]+)\]", python)
    assert shards
    reports = ["cpp", *(f"python-{shard.strip()}" for shard in shards[1].split(","))]
    batches = re.split(r"^          - batch: ", _job(), flags=re.MULTILINE)[1:]
    assert len(batches) == 3
    actual = []
    for batch in batches:
        entries = re.split(r"^              - artifact: ", batch, flags=re.MULTILINE)[
            1:
        ]
        assert len(entries) == 3
        for entry in entries:
            artifact, rest = entry.split("\n", 1)
            report = artifact.removeprefix("coverage-report-")
            suffix = "info" if report == "cpp" else "xml"
            assert artifact == f"coverage-report-{report}"
            assert (
                _field(rest, "file") == f"coverage/{report}/coverage-{report}.{suffix}"
            )
            assert _field(rest, "directory") == f"coverage/{report}"
            assert _field(rest, "flag") == ("cpp" if report == "cpp" else "python")
            assert _field(rest, "name") == f"code-coverage/{report}"
            actual.append(report)
    assert sorted(actual) == sorted(reports)


def test_coverage_failures_are_local_and_every_report_has_a_finite_budget() -> None:
    job = _job()
    assert _field(job, "continue-on-error") == "true"
    assert _field(job, "fail-fast") == "false"
    steps = _steps()
    assert len(steps) == 7
    # Reserve cleanup time even if every action consumes its full allowance.
    assert sum(int(_field(step, "timeout-minutes")) for step in steps) + 1 <= int(
        _field(job, "timeout-minutes")
    )
    assert int(_field(job, "timeout-minutes")) == 8
    for index in range(3):
        download, upload = steps[1 + 2 * index : 3 + 2 * index]
        assert _field(download, "continue-on-error") == "true"
        assert not re.search(r"^        if:", download, re.MULTILINE)
        step_id = _field(download, "id")
        assert _field(upload, "if") == (
            "${{ !cancelled() && steps." + step_id + ".outcome == 'success' }}"
        )
        assert _field(upload, "continue-on-error") == "true"
        for key, field in (("name", "artifact"), ("path", "directory")):
            # Restrict matching to action inputs, not the human-readable name.
            inputs = download.split("        with:\n", 1)[1]
            assert _field(inputs, key) == f"${{{{ matrix.reports[{index}].{field} }}}}"
        inputs = upload.split("        with:\n", 1)[1]
        for key, field in (("files", "file"), ("flags", "flag"), ("name", "name")):
            assert _field(inputs, key) == f"${{{{ matrix.reports[{index}].{field} }}}}"
        for key, value in (
            ("disable_search", "true"),
            ("fail_ci_if_error", "false"),
            ("use_oidc", "true"),
        ):
            assert _field(inputs, key) == value


def test_coverage_oidc_job_only_consumes_artifacts_with_pinned_actions() -> None:
    job = _job()
    assert _field(job, "needs") == "[cpu, python]"
    assert _field(job, "if") == (
        "${{ github.event_name != 'merge_group' && github.event_name != 'push' "
        "&& !cancelled() && needs.cpu.result == 'success' "
        "&& needs.python.result == 'success' }}"
    )
    permissions = job.split("    permissions:\n", 1)[1].split("    steps:\n", 1)[0]
    assert dict(re.findall(r"^      ([\w-]+): (\w+)$", permissions, re.MULTILINE)) == {
        "contents": "read",
        "id-token": "write",
    }
    allowed = {
        "actions/checkout",
        "actions/download-artifact",
        "codecov/codecov-action",
    }
    for step in _steps():
        action, version = _field(step, "uses").split("@")
        assert action in allowed
        assert re.fullmatch(r"[a-f0-9]{40}", version)
        assert not re.search(
            r"^        (?:run|env|working-directory):", step, re.MULTILINE
        )
    checkout = _steps()[0]
    assert _field(checkout, "uses").startswith("actions/checkout@")
    assert _field(checkout, "persist-credentials") == "false"
    assert _field(checkout, "fetch-depth") == "0"
    pass_job = WORKFLOW.read_text().split("\n  pass:\n", 1)[1]
    assert "upload-coverage" not in _field(pass_job, "needs")
