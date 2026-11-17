"""Cold3 protocol, allocation metadata and archived-v1 compatibility."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.readme_cold_repeats import SCHEMA, aggregate, population_order
from benchmarks.readme_defaults import allocation_metadata
from tests.python.test_readme_defaults import receipt


def matrix() -> list[dict]:
    records = []
    for repeat in range(3):
        for method, mode, atoms, arm in population_order(repeat):
            record = receipt(arm)
            record.update(
                method=method,
                mode=mode,
                atoms=atoms,
                process_repeat=repeat,
                accelerator={"nvidia_smi": {"uuid": "same-gpu"}},
                native_build={"library_sha256": "same-library"},
                runner_sha256="same-runner",
            )
            record["samples"][0]["seconds"] = [5, 90, 4][repeat]
            for sample in record["samples"]:
                sample["forces"] = [[0.0, 0.0, 0.0]] * atoms
            records.append(record)
    return records


def test_all_cold_outliers_are_retained_and_warm_grouping_is_explicit() -> None:
    rows = aggregate(matrix())
    assert len(rows) == 128
    assert rows[0]["cold"] == {"seconds": [5, 90, 4], "median": 5, "min": 4, "max": 90}
    assert rows[0]["warm"]["process_medians"] == [2, 2, 2]
    assert len(rows[0]["warm"]["all_seconds"]) == 9


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "failed", "gpu", "source", "warm", "nonfinite", "shape"],
)
def test_no_complete_summary_for_a_partial_or_mixed_campaign(change: str) -> None:
    records = matrix()
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records.append(deepcopy(records[0]))
    elif change == "failed":
        records[0]["status"] = "failed"
    elif change == "gpu":
        records[0]["accelerator"]["nvidia_smi"]["uuid"] = "other-gpu"
    elif change == "source":
        records[0]["source_revision"] = "b" * 40
    elif change == "warm":
        records[0]["samples"][2]["repeat"] = 0
    elif change == "nonfinite":
        records[0]["samples"][0]["seconds"] = float("nan")
    else:
        records[0]["samples"][0]["forces"] = [[0, 0, 0]]
    with pytest.raises(ValueError):
        aggregate(records)


def test_qz_never_fabricates_slurm_and_rejects_unbound_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("SLURM_JOB_ID", "LSB_JOBID", "GQC_QZ_JOB_NAME", "GQC_QZ_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="real GPU work"):
        allocation_metadata("test-job")
    monkeypatch.setenv("GQC_QZ_JOB_NAME", "test-job")
    monkeypatch.setenv("GQC_QZ_PROJECT", "project")
    monkeypatch.setattr("socket.gethostname", lambda: "other-job--pod")
    with pytest.raises(ValueError):
        allocation_metadata("test-job")
    monkeypatch.setattr(
        "socket.gethostname",
        lambda: "job-12345678-1234-1234-1234-123456789abc-worker-0",
    )
    assert allocation_metadata("test-job")["kind"] == "qz"
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    assert allocation_metadata() == {"kind": "slurm"}
    with pytest.raises(ValueError, match="combine"):
        allocation_metadata("test-job")


def test_each_round_balances_first_engine_order() -> None:
    for repeat in range(3):
        order = population_order(repeat)
        first = [order[i][3] for i in range(0, len(order), 2)]
        assert first.count("standard") == first.count("gpu4pyscf") == 32
        assert len(set(order)) == 128


def test_both_recorded_qz_host_conventions_are_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("LSB_JOBID", raising=False)
    monkeypatch.setenv("GQC_QZ_JOB_NAME", "bound-job")
    monkeypatch.setenv("GQC_QZ_PROJECT", "project")
    for hostname in (
        "job-12345678-1234-1234-1234-123456789abc-worker-0",
        "bound-job--notebook",
    ):
        monkeypatch.setattr("socket.gethostname", lambda hostname=hostname: hostname)
        assert allocation_metadata("bound-job")["platform_receipt_required"] is True


@pytest.mark.parametrize("device", ["h100", "rtx5090"])
def test_original_v1_cohorts_keep_every_aggregate_value(device: str) -> None:
    from benchmarks.paper_hfdft_replay import DEFAULT, populations, read_xz

    payload = read_xz((DEFAULT / device / "samples.json.xz").read_bytes())
    assert SCHEMA == "generativeqc.readme-cold-repeats.v1"
    actual = {tuple(row["key"]): row for row in aggregate(payload["records"])}
    expected = {tuple(row["key"]): row for row in populations(payload["records"])}
    assert actual == expected


@pytest.mark.parametrize("fail_first", [False, True])
def test_acquisition_keeps_fixed_process_census_and_never_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_first: bool
) -> None:
    from benchmarks import readme_cold_repeats as collector

    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-test-only")
    monkeypatch.delenv("LSB_JOBID", raising=False)
    templates = {
        (r["arm"], r["method"], r["mode"], r["atoms"], r["process_repeat"]): r
        for r in matrix()
    }
    invoked = []

    class Child:
        def __init__(self, argv: list[str], **kwargs: object) -> None:
            invoked.append(argv)
            arm = argv[3]
            method, mode = (
                argv[argv.index("--method") + 1],
                argv[argv.index("--mode") + 1],
            )
            atoms = int(argv[argv.index("--atoms") + 1])
            target = Path(argv[argv.index("--output") + 1])
            repeat = int(target.stem.rsplit("-r", 1)[1])
            record = deepcopy(templates[(arm, method, mode, atoms, max(0, repeat))])
            record.pop("process_repeat")
            record["samples"] = record["samples"][
                : int(argv[argv.index("--warm-repeats") + 1]) + 1
            ]
            target.write_text(json.dumps(record), encoding="utf-8")

        def wait(self, timeout: float) -> int:
            assert timeout > 0
            return 1 if fail_first else 0

    monkeypatch.setattr(collector.subprocess, "Popen", Child)
    config = {
        "source": str(tmp_path),
        "inputs": str(tmp_path),
        "source_revision": "a" * 40,
        "deadline_seconds": 1000,
        "process_timeout_seconds": 30,
        "interpreters": {a: "fake-python" for a in ("standard", "gpu4pyscf")},
        "environment": {},
        "arm_environment": {a: {} for a in ("standard", "gpu4pyscf")},
    }
    output = tmp_path / "acquisition"
    if fail_first:
        with pytest.raises(RuntimeError, match="no retry"):
            collector.acquire(config, output)
    else:
        collector.acquire(config, output)
    campaign = json.loads((output / "campaign.json").read_bytes())
    if fail_first:
        assert len(invoked) == 1
        assert campaign["status"] == "failed"
    else:
        assert len(invoked) == 512
        assert campaign["formal_processes"] == 384
        assert campaign["formal_endpoints"] == 1536
        assert len([r for r in campaign["commands"] if r["stage"] == "prewarm"]) == 128
