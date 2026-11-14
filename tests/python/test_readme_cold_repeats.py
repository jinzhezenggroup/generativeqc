"""Independent cold-population completeness and allocation behavior."""

from copy import deepcopy

import pytest

from benchmarks.readme_cold_repeats import aggregate, population_order
from benchmarks.readme_cold_repeats_report import diagnostics, verify_build
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


def test_single_round_peak_is_reported_without_selecting_or_repairing_it() -> None:
    rows = aggregate(matrix())
    spike = next(row for row in rows if row["key"] == ["gpu4pyscf", "hf", "direct", 12])
    spike["cold"]["seconds"][1] = 120
    result = diagnostics(rows)
    assert result["cold_decreases"] == []
    assert result["cold_decreases_by_process"]["0"] == []
    assert result["cold_decreases_by_process"]["2"] == []
    assert result["cold_decreases_by_process"]["1"] == [
        {
            "arm": "gpu4pyscf",
            "method": "hf",
            "mode": "direct",
            "atoms": [12, 48],
            "seconds": [120, 90],
            "relative_drop": 0.25,
        }
    ]
    assert spike["cold"]["seconds"] == [5, 120, 4]
    assert spike["cold"]["median"] == 5


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


def test_publication_requires_the_actual_built_image_and_recorder() -> None:
    record = matrix()[0]
    record["native_build"]["source_identity"] = "frozen-source-identity"
    record["accelerator"]["pci_bus_id"] = "0000:00:00.0"
    record["environment"] = {"packages": {"numpy": "fixture-version"}}
    build = {
        "status": "BUILD_PASS_SCIENCE_NOT_RUN",
        "stages": [
            {"name": name, "status": "PASS", "returncode": 0}
            for name in ("configure", "build", "aot-audit")
        ],
        "source": {
            "revision": record["source_revision"],
            "overlay_sha256": {
                "benchmarks/readme_defaults.py": record["runner_sha256"]
            },
        },
        "library_sha256": record["native_build"]["library_sha256"],
        "native_source_identity": record["native_build"]["source_identity"],
        "device": "NVIDIA test GPU, same-gpu, 12.0, fixture-driver, fixture-power",
        "pci_bus_id": "0000:00:00.0",
    }
    verify_build([record], build)
    mutations = [
        {**build, "status": "RUNNING"},
        {**build, "stages": build["stages"][:-1]},
        {**build, "library_sha256": "another-image"},
        {**build, "native_source_identity": "another-source"},
        {**build, "pci_bus_id": "0000:01:00.0"},
        {
            **build,
            "device": "NVIDIA test GPU, other-gpu, 12.0, fixture-driver, fixture-power",
        },
        {**build, "source": {**build["source"], "revision": "b" * 40}},
        {
            **build,
            "source": {
                **build["source"],
                "overlay_sha256": {"benchmarks/readme_defaults.py": "another-recorder"},
            },
        },
    ]
    for changed in mutations:
        with pytest.raises(ValueError):
            verify_build([record], changed)
    second = deepcopy(record)
    second["environment"]["packages"]["numpy"] = "changed-version"
    with pytest.raises(ValueError, match="software environment"):
        verify_build([record, second], build)
    moved = deepcopy(record)
    moved.update(slurm_job_id="new-allocation")
    moved["accelerator"].update(
        pci_bus_id="0000:01:00.0", nvidia_smi={"uuid": "new-gpu"}
    )
    execution = {
        "schema": "generativeqc.paper-cold3-execution.v1",
        "status": "READY_SOURCE_MATCHED_AUTHENTICATED_BUILD_REUSE",
        "source_revision": build["source"]["revision"],
        "native_source_identity": build["native_source_identity"],
        "library_sha256": build["library_sha256"],
        "device": "NVIDIA test GPU, new-gpu, 12.0, fixture-driver, fixture-power",
        "pci_bus_id": "0000:01:00.0",
        "scheduler": {"SLURM_JOB_ID": "new-allocation"},
    }
    verify_build([moved], build, execution)
    with pytest.raises(ValueError, match="execution binding"):
        verify_build(
            [moved],
            build,
            {**execution, "scheduler": {"SLURM_JOB_ID": "different-allocation"}},
        )
    with pytest.raises(ValueError, match="authenticated build"):
        verify_build([moved], build, {**execution, "library_sha256": "another-library"})
