"""Protect independent gates, retained force bits and honest population labels."""

import base64
import hashlib
import json
import lzma
from pathlib import Path

import numpy as np
import pytest

from tools.generativeqc_validation import occupied_xc_evidence as evidence


@pytest.fixture
def packet() -> dict:
    return json.loads(
        lzma.decompress((evidence.ROOT / "evidence.json.xz").read_bytes())
    )


def save(root: Path, packet: dict) -> None:
    (root / "evidence.json.xz").write_bytes(lzma.compress(json.dumps(packet).encode()))


def test_retained_force_archive_integrity_manifest() -> None:
    digest, name = (evidence.ROOT / "SHA256SUMS").read_text().split()
    assert name == "evidence.json.xz"
    assert hashlib.sha256((evidence.ROOT / name).read_bytes()).hexdigest() == digest


def test_independent_replay_and_observational_receipts_are_separate() -> None:
    result = evidence.summarize()
    assert result["qualified_profile_calls"] > 0
    assert result["observational_calls"] > 0
    for population in result["populations"].values():
        assert [row["phase"] for row in population["calls"]] == evidence.REFERENCE[
            "PHASES"
        ]
        assert population["maximum_energy_error"] <= 1e-8
        assert population["maximum_force_error"] <= 1e-7
    assert not set(result["observational_populations"]) & set(result["populations"])


def test_force_bit_corruption_is_rejected(tmp_path: Path, packet: dict) -> None:
    raw = bytearray(base64.b64decode(packet["forces"]["data"]))
    raw[0] ^= 1
    packet["forces"]["data"] = base64.b64encode(raw).decode()
    packet["forces"]["byteplanes_sha256"] = hashlib.sha256(raw).hexdigest()
    save(tmp_path, packet)
    with pytest.raises(AssertionError):
        evidence.summarize(tmp_path)


@pytest.mark.parametrize("field", ["source", "protocol", "energy", "work", "wall"])
def test_inconsistent_scientific_or_timing_receipts_are_rejected(
    tmp_path: Path, packet: dict, field: str
) -> None:
    profile = next(iter(packet["profiles"].values()))
    native = profile["scientific_records"]
    if field == "source":
        native["native_build"]["probe"]["source_identity"] = "0" * 64
    elif field == "protocol":
        native["protocol_sha256"] = "0" * 64
    elif field == "energy":
        native["records"][0]["energy"] += 1e-5
    elif field == "work":
        profile["windows"]["cold"]["record"]["fock_builds"] += 1
    else:
        profile["windows"]["cold"]["complete_endpoint_wall_ledger"][
            "complete_exclusive_wall_seconds"
        ]["endpoint_preparation"] += 1
    save(tmp_path, packet)
    with pytest.raises(AssertionError):
        evidence.summarize(tmp_path)


def test_profile_convergence_trajectories_are_not_normalized() -> None:
    result = evidence.summarize()
    for profile in result["profiles"].values():
        for window in profile["windows"].values():
            assert window["record"]["iterations"] > 0
            assert window["record"]["fock_builds"] > 0
            assert all(
                seconds >= 0 for seconds in window["kernel_device_seconds"].values()
            )


def test_kernel_device_times_do_not_become_wall_time(
    tmp_path: Path, packet: dict
) -> None:
    profile = next(iter(packet["profiles"].values()))
    window = profile["windows"]["cold"]
    window["kernel_device_seconds"]["concurrent_kernel_activity"] = (
        2 * window["public_execute_seconds"]
    )
    wall = dict(window["complete_endpoint_wall_ledger"])
    save(tmp_path, packet)
    result = evidence.summarize(tmp_path)
    actual = next(iter(result["profiles"].values()))["windows"]["cold"]
    assert actual["complete_endpoint_wall_ledger"] == wall


@pytest.mark.parametrize("receipt", ["library.sha256", "independent_validation"])
def test_observational_receipts_must_belong_to_the_loaded_library(
    tmp_path: Path, packet: dict, receipt: str
) -> None:
    stored = next(iter(packet["observational_populations"].values()))
    if receipt == "library.sha256":
        stored["receipts"][receipt] = "0" * 64 + "  foreign-library.so"
    else:
        stored["independent_validation_receipt"]["library_sha256"] = "0" * 64
    save(tmp_path, packet)
    with pytest.raises(AssertionError):
        evidence.summarize(tmp_path)


def test_negative_wall_partitions_cannot_hide_behind_a_reconciled_sum(
    tmp_path: Path, packet: dict
) -> None:
    profile = next(iter(packet["profiles"].values()))
    ledger = profile["windows"]["cold"]["complete_endpoint_wall_ledger"][
        "complete_exclusive_wall_seconds"
    ]
    ledger["fabricated_negative"] = -1.0
    ledger["fabricated_positive"] = 1.0
    save(tmp_path, packet)
    with pytest.raises(AssertionError):
        evidence.summarize(tmp_path)


def test_observational_partial_population_is_not_a_completed_comparison(
    tmp_path: Path, packet: dict
) -> None:
    key, stored = next(iter(packet["observational_populations"].items()))
    stored["stage"], stored["status"] = "warm/3", "running"
    stored["records"] = stored["records"][:4]
    save(tmp_path, packet)
    result = evidence.summarize(tmp_path)
    assert key not in result["observational_populations"]
    assert result["incomplete_populations"][key]["completed_calls"] == 4


def population_packet(
    *, count: int = 12, incomplete: bool = False, bad_energy: bool = False
) -> dict:
    """Make a codec fixture from public retained samples, not a new measurement."""
    case = "r2scan-3c-48"
    native = evidence.REFERENCE["load"](f"raw/candidate-{case}.json.xz")
    oracle = evidence.REFERENCE["load"](f"raw/reference-{case}.json.xz")
    receipt = evidence.REFERENCE["qualify"](native, oracle)
    build = native["native_build"]
    stored = {
        "protocol_key": case,
        "source_key": "mainline/baseline",
        "protocol_sha256": evidence._digest(native["protocol"]),
        "native_build": build,
        "scheduler": native["scheduler"],
        "records": native["records"][:count],
        "stage": "incomplete" if incomplete else "complete",
        "status": "timed_out" if incomplete else "measured",
        "force_retention": "lossless",
        "original_json_sha256": evidence._digest(native),
        "independent_validation_receipt": receipt,
    }
    if bad_energy:
        stored["records"][0]["energy"] += 1e-5
        stored["records"][0]["energy_error"] = 0.0
    stored["projection_sha256"] = evidence._digest(stored)
    force_digest = hashlib.sha256()
    anchors = {
        row["geometry"]: row["forces"]
        for row in native["records"]
        if row["phase"] in ("cold", "moved")
    }
    encoded = np.zeros((count, 288), dtype="<u8")
    for index, row in enumerate(stored["records"]):
        forces = np.asarray(row.pop("forces"), dtype="<f8").reshape(-1)
        anchor = np.asarray(anchors[row["geometry"]], dtype="<f8").reshape(-1)
        force_digest.update(forces.tobytes())
        encoded[index, : forces.size] = forces.view("<u8") ^ anchor.view("<u8")
        row["forces_column"] = index
    stored["receipts"] = {
        "library.sha256": build["library_sha256"],
        "exit-status.txt": "0",
    }
    raw = encoded.view("u1").reshape(count, 288, 8).transpose(2, 0, 1).copy().tobytes()
    return {
        "schema": "occupied-xc-cold-v2",
        "reference_sha256": {
            name: hashlib.sha256(
                (evidence.REFERENCE["ROOT"] / name).read_bytes()
            ).hexdigest()
            for name in ("forces.npz", "evidence.json.xz")
        },
        "populations": {f"test/baseline/{case}": stored},
        "profiles": {},
        "observational_populations": {},
        "sources": {
            "mainline/baseline": {
                "library_sha256": build["library_sha256"],
                "source_identity": build["probe"]["source_identity"],
            }
        },
        "forces": {
            "encoding": "xor-shared-native-fp64-byteplanes",
            "shape": [8, count, 288],
            "data": base64.b64encode(raw).decode(),
            "byteplanes_sha256": hashlib.sha256(raw).hexdigest(),
            "reconstructed_sha256": force_digest.hexdigest(),
        },
    }


def test_complete_population_is_independently_replayed(tmp_path: Path) -> None:
    save(tmp_path, population_packet())
    result = evidence.summarize(tmp_path)
    assert result["qualified_calls"] == len(evidence.REFERENCE["PHASES"])
    assert not result["incomplete_populations"]


def test_timeout_is_reported_without_complete_endpoint_comparison(
    tmp_path: Path,
) -> None:
    save(tmp_path, population_packet(count=11, incomplete=True))
    result = evidence.summarize(tmp_path)
    assert result["qualified_calls"] == 0 and not result["comparisons"]
    assert (
        next(iter(result["incomplete_populations"].values()))["completed_calls"] == 11
    )


def test_missing_call_cannot_be_relabelled_as_complete(tmp_path: Path) -> None:
    save(tmp_path, population_packet(count=11))
    with pytest.raises(AssertionError):
        evidence.summarize(tmp_path)


def test_false_gate_receipt_cannot_replace_independent_oracle(tmp_path: Path) -> None:
    save(tmp_path, population_packet(bad_energy=True))
    with pytest.raises(AssertionError):
        evidence.summarize(tmp_path)
