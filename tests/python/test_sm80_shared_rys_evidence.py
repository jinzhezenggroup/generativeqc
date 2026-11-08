"""Historical E/F failures must survive replay and cannot authorize promotion."""

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
READER = ROOT / "benchmarks/results/a100-shared-rys-xy-20261108/replay.py"


@pytest.fixture(scope="module")
def reader() -> dict[str, Any]:
    return runpy.run_path(str(READER))


def test_default_force_failure_is_not_hidden_by_native_parity(
    reader: dict[str, Any],
) -> None:
    summary = reader["replay"]()
    assert not summary["independent_default_promotion_qualified"]
    for card in summary["cards"].values():
        case = card["default_cases"]["48"]
        assert case["all_120_native_pairs_pass"]
        assert len(case["independent_default_observations"]) == 16
        assert all(
            not item["passed"] for item in case["independent_default_observations"]
        )
        assert all(
            item["force"] > 1e-7 for item in case["independent_default_observations"]
        )
        assert all(
            item["passed"]
            for item in card["default_cases"]["96"]["independent_default_observations"]
        )
        assert all(item["passed"] for item in card["moved_and_restored_hf96"])


def test_tight_request_is_reported_separately(reader: dict[str, Any]) -> None:
    summary = reader["replay"]()
    for card in summary["cards"].values():
        for samples in card["strict_request_hf48"].values():
            assert len(samples) == 4
            assert all(item["passed"] for item in samples)
    assert summary["paired_warm_ratio_ceiling"] == 1.03
    assert summary["independent_gates"]["maximum_force"] == 1e-7


def test_require_promotion_fails_closed(
    reader: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.argv", [str(READER), "--require-promotion"])
    with pytest.raises(SystemExit, match="promotion remains blocked"):
        reader["main"]()


@pytest.mark.parametrize("forces", [[[float("nan"), 0, 0]], [[0, 0]]])
def test_invalid_independent_forces_cannot_pass(
    reader: dict[str, Any], forces: list[list[float]]
) -> None:
    actual = {"energy": 0.0, "forces": forces, "converged": True}
    target = {"energy": 0.0, "forces": [[0.0, 0.0, 0.0]], "converged": True}
    with pytest.raises(ValueError):
        reader["observable_error"](actual, target)


@pytest.mark.parametrize("stream", ("receipts.json.xz", "fp64.xz"))
def test_changed_capsule_is_rejected(
    reader: dict[str, Any], tmp_path: Path, stream: str
) -> None:
    directory = READER.parent
    target = tmp_path / directory.name
    shutil.copytree(directory, target)
    shared = directory.parent / "hf-screened-handoff-rejections-20261108"
    storage = tmp_path / shared.name
    shutil.copytree(shared, storage)
    payload = storage / stream
    payload.write_bytes(payload.read_bytes() + b"altered")
    with pytest.raises(ValueError, match=stream):
        reader["load"](target)


@pytest.mark.parametrize("converged", (False, None, 1))
def test_reference_convergence_is_required(
    reader: dict[str, Any], converged: object
) -> None:
    """Matching observables do not qualify an unknown or failed reference."""
    actual = {"energy": 0.0, "forces": [[0.0, 0.0, 0.0]], "converged": True}
    target = {**actual, "converged": converged}
    assert not reader["observable_error"](actual, target)["passed"]
    target.pop("converged")
    assert not reader["observable_error"](actual, target)["passed"]


@pytest.mark.parametrize("size", ("48", "96"))
def test_replay_rejects_failed_selected_reference(
    reader: dict[str, Any], monkeypatch: pytest.MonkeyPatch, size: str
) -> None:
    """Explicit nonconvergence cannot be overridden by historical provenance."""
    cohort, capsule = reader["load"]()
    capsule = dict(capsule)
    name = cohort["reference_files"][size]
    capsule[name] = json.dumps({**json.loads(capsule[name]), "converged": False})
    monkeypatch.setitem(
        reader["replay"].__globals__, "load", lambda _: (cohort, capsule)
    )
    with pytest.raises(ValueError, match="Unconverged independent reference"):
        reader["replay"]()


@pytest.mark.parametrize("corruption", ("reference", "producer", "recipe"))
def test_historical_reference_requires_bound_producer(
    reader: dict[str, Any], monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    """The legacy missing convergence field is allowed only for its provenance."""
    cohort, capsule = reader["load"]()
    capsule = dict(capsule)
    if corruption == "reference":
        name = cohort["reference_files"]["96"]
        record = json.loads(capsule[name])
        record["energy"] += 1e-12
    elif corruption == "producer":
        name = "oracle-hf96-producer-228320.json"
        record = json.loads(capsule[name])
        record[0]["energy"] += 1e-12
    else:
        name = "receipts/native-a100-frozen-hf96-gates.json"
        record = json.loads(capsule[name])
        record["scientific_recipe_sha256"][
            "tests/python/test_hf_cold_screening_cuda.py"
        ] = "0" * 64
    capsule[name] = json.dumps(record)
    monkeypatch.setitem(
        reader["replay"].__globals__, "load", lambda _: (cohort, capsule)
    )
    with pytest.raises(ValueError, match="historical HF96 reference provenance"):
        reader["replay"]()


@pytest.mark.parametrize("card", ("a100", "5090"))
@pytest.mark.parametrize(
    "corruption", ("source_identity", "library_sha256", "geometry", "arm", "job")
)
def test_strict_auxiliary_requires_its_image_and_scientific_inputs(
    reader: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    card: str,
    corruption: str,
) -> None:
    cohort, capsule = reader["load"]()
    capsule = dict(capsule)
    name = f"final-{card}/hf48-tight-control.json"
    record = json.loads(capsule[name])
    if corruption in ("source_identity", "library_sha256"):
        record["native_build"][corruption] = "0" * 64
    elif corruption == "geometry":
        record["geometry_bohr"][1][1][0] += 0.01
    elif corruption == "arm":
        record["xy_control"] = None
    else:
        record["scheduler"]["LSB_JOBID" if card == "a100" else "SLURM_JOB_ID"] = "0"
    capsule[name] = json.dumps(record)
    monkeypatch.setitem(
        reader["replay"].__globals__, "load", lambda _: (cohort, capsule)
    )
    with pytest.raises(ValueError, match="auxiliary"):
        reader["replay"]()


@pytest.mark.parametrize("card", ("a100", "5090"))
@pytest.mark.parametrize(
    "corruption", ("geometry", "settings", "engine", "identity", "job", "exit")
)
def test_legacy_extra_requires_bound_same_job_provenance(
    reader: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    card: str,
    corruption: str,
) -> None:
    cohort, capsule = reader["load"]()
    capsule = dict(capsule)
    name = f"final-{card}/extra-native.json"
    if corruption == "exit":
        capsule[f"final-{card}/exit-code.txt"] = "1\n"
    else:
        if corruption in ("identity", "job"):
            name = f"final-{card}/identity-48-4-control.json"
        record = json.loads(capsule[name])
        if corruption == "geometry":
            record["observations"][0]["geometry_bohr"][1][1][0] += 0.01
        elif corruption == "settings":
            record["observations"][1]["settings"]["screening_tolerance"] = 1e-6
        elif corruption == "engine":
            record["engine"] = "reference"
        elif corruption == "identity":
            record["library_sha256"] = "0" * 64
        else:
            record["scheduler"]["LSB_JOBID" if card == "a100" else "SLURM_JOB_ID"] = "0"
        capsule[name] = json.dumps(record)
    monkeypatch.setitem(
        reader["replay"].__globals__, "load", lambda _: (cohort, capsule)
    )
    with pytest.raises(ValueError, match="auxiliary"):
        reader["replay"]()


@pytest.mark.parametrize(
    "name",
    (
        "drivers/strict-native.py",
        "drivers/qualify-extra.py",
        "drivers/worker-final2-a100.sh",
        "drivers/worker-final2-5090.sh",
        "final2/image-identity.json",
    ),
)
def test_historical_auxiliary_recipe_cannot_be_substituted(
    reader: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """Bind missing legacy fields to retained inputs, never current source hashes."""
    cohort, capsule = reader["load"]()
    capsule = {**capsule, name: capsule[name] + "\n"}
    monkeypatch.setitem(
        reader["replay"].__globals__, "load", lambda _: (cohort, capsule)
    )
    with pytest.raises(ValueError, match="auxiliary provenance"):
        reader["replay"]()


def test_legacy_extra_image_scope_is_explicit(reader: dict[str, Any]) -> None:
    for card in reader["replay"]()["cards"].values():
        assert (
            "no per-record native image measurement"
            in card["auxiliary_provenance"]["moved_and_restored_hf96"]
        )
