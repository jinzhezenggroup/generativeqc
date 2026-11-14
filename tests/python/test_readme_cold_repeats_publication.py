"""Replay both actual frozen GPU cohorts and reject altered science/work."""

import json
import lzma
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.readme_cold_repeats import population_order
from benchmarks.readme_cold_repeats_replay import replay
from tools.generativeqc_validation.publication import validate_publication

DIRECTORY = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/paper-cold3-20261114"
)


@pytest.fixture(scope="module")
def bundle() -> tuple[dict, dict[str, bytes], list[dict]]:
    manifest = json.loads((DIRECTORY / "publication.json").read_bytes())
    files = {
        entry["path"]: (DIRECTORY / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    oracle = DIRECTORY.parent / "readme-refresh-20261026/oracles.json.xz"
    return manifest, files, json.loads(lzma.decompress(oracle.read_bytes()))


@pytest.mark.parametrize("device", ("h100", "rtx5090"))
def test_complete_real_cohort_replays_all_values_and_original_common_gates(
    bundle: tuple[dict, dict[str, bytes], list[dict]], device: str
) -> None:
    manifest, files, oracles = bundle
    payload = json.loads(lzma.decompress(files[device + ".json.xz"]))
    result = replay(payload, oracles)
    assert len(payload["records"]) == 384
    assert sum(len(r["samples"]) for r in payload["records"]) == 1536
    assert len(result["populations"]) == 128
    for repeat in range(3):
        records = [r for r in payload["records"] if r["process_repeat"] == repeat]
        assert [
            tuple(r[k] for k in ("method", "mode", "atoms", "arm")) for r in records
        ] == population_order(repeat)
        checks = [r for r in result["accuracy"] if r["process_repeat"] == repeat]
        assert Counter(r["status"] for r in checks) == {"qualified": 116, "failed": 12}
    expected_failed = {
        (arm, "hf", "df", n)
        for arm in ("standard", "gpu4pyscf")
        for n in (3, 12, 48, 96)
    } | {
        (arm, "m06-2x", mode, 3)
        for arm in ("standard", "gpu4pyscf")
        for mode in ("direct", "df")
    }
    assert {
        (r["arm"], r["method"], r["mode"], r["atoms"])
        for r in result["accuracy"]
        if r["status"] == "failed"
    } == expected_failed
    assert manifest["decision"]["status"] == "inconclusive"
    envelope = json.loads(files["evidence.json"])
    assert envelope["stages"]["numerical"]["status"] == "fail"
    assert envelope["performance"]["status"] == "not-run"
    assert payload["projection"]["original_reconstruction"] is False
    platform = json.loads(files["platform-receipts.json"])
    if device == "h100":
        status = platform["qz_status"]["data"]
        assert status["status"] == "SUCCEEDED"
        assert all(
            r["allocation"]["job_name"] == status["name"]
            and r["allocation"]["project"] == status["project"]
            for r in payload["records"]
        )
    else:
        status = platform["ustc_status"]
        assert status["job_state"] == "COMPLETED"
        assert status["exit_code"] == "0:0"
        assert {str(r["slurm_job_id"]) for r in payload["records"]} == {
            status["job_id"]
        }


@pytest.mark.parametrize("change", ("missing", "time", "full_force", "warm_work"))
def test_actual_projection_does_not_accept_silent_sample_or_work_edits(
    bundle: tuple[dict, dict[str, bytes], list[dict]], change: str
) -> None:
    _, files, oracles = bundle
    payload = deepcopy(json.loads(lzma.decompress(files["h100.json.xz"])))
    if change == "missing":
        payload["records"].pop()
    elif change == "time":
        payload["records"][0]["samples"][0]["seconds"] += 1
    elif change == "full_force":
        payload["records"][0]["samples"][0]["forces"][0][0] += 1e-3
    else:
        payload["records"][0]["samples"][1]["fock_builds"] += 1
    with pytest.raises(ValueError):
        replay(payload, oracles)
