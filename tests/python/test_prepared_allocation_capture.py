"""Coordinator protocol gates; real Slurm captures remain separate evidence."""

from __future__ import annotations

import copy
import hashlib
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tools.audit_replay_allocations import SCHEMA_V2, InvalidReceipt, verify_receipt
from tools.capture_prepared_allocations import (
    DEVICE,
    ENDPOINTS,
    HOST,
    ROOT,
    assemble_windows,
    freeze_source,
    load,
    numerical_evidence,
    schedule,
    source_tree,
    verify_source,
    workload,
    write_new,
)


@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_pinned_moved_geometry_changes_internal_distances(method: str) -> None:
    """A stale geometry must not pass the audit solely by rigid translation."""
    case = workload(method)
    for atoms, moved in zip(case["systems"], case["moved_coordinates"], strict=True):
        original = np.array([position for _, position in atoms])
        displaced = np.asarray(moved)
        np.testing.assert_array_equal(displaced[:-1], original[:-1])
        np.testing.assert_array_equal(displaced[-1, :2], original[-1, :2])
        assert displaced[-1, 2] == original[-1, 2] + case["moved_dz"]
        assert np.linalg.norm(displaced[-1] - displaced[0]) != pytest.approx(
            np.linalg.norm(original[-1] - original[0]), abs=1e-12, rel=0.0
        )


def entries_for(root: Path) -> list[dict[str, str]]:
    """Use Git's independent staging representation, including symlink blobs."""
    entries = []
    rows = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root)
    for row in rows.split(b"\0"):
        if row:
            metadata, name = row.split(b"\t", 1)
            mode, blob, _ = metadata.decode().split()
            path = root / os.fsdecode(name)
            content = (
                os.fsencode(os.readlink(path))
                if path.is_symlink()
                else path.read_bytes()
            )
            entries.append(
                {
                    "path": os.fsdecode(name),
                    "mode": mode,
                    "blob_sha1": blob,
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    return entries


def test_exported_tree_matches_independent_git_order_modes_and_symlinks(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "name").mkdir()
    (tmp_path / "name/child.py").write_text("payload")
    (tmp_path / "name.extra").write_text("sibling")
    (tmp_path / "executable").write_text("tool")
    (tmp_path / "executable").chmod(0o755)
    (tmp_path / "link").symlink_to("name/child.py")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    independent = subprocess.check_output(
        ["git", "write-tree"], cwd=tmp_path, text=True
    ).strip()
    entries = entries_for(tmp_path)
    assert source_tree(tmp_path, entries) == independent
    assert source_tree(tmp_path, list(reversed(entries))) == independent
    (tmp_path / "name/child.py").write_text("changed")
    with pytest.raises(InvalidReceipt, match="bytes mismatch"):
        source_tree(tmp_path, entries)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate", "duplicate"),
        ("unsafe", "unsafe"),
        ("mode", "mode mismatch"),
        ("digest", "bytes mismatch"),
    ],
)
def test_source_manifest_rejects_unsafe_or_mismatched_entries(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"content")
    entry = {
        "path": "file",
        "mode": "100644",
        "blob_sha1": hashlib.sha1(
            b"blob 7\0content", usedforsecurity=False
        ).hexdigest(),
        "sha256": hashlib.sha256(b"content").hexdigest(),
    }
    entries = [entry]
    if mutation == "duplicate":
        entries.append(dict(entry))
    elif mutation == "unsafe":
        entry["path"] = "../file"
    elif mutation == "mode":
        entry["mode"] = "100755"
    else:
        entry["sha256"] = "0" * 64
    with pytest.raises(InvalidReceipt, match=match):
        source_tree(tmp_path, entries)


def test_freeze_verifies_full_working_tree_without_modifying_real_index(
    tmp_path: Path,
) -> None:
    index = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--git-path", "index"],
            cwd=ROOT,
            text=True,
        ).strip()
    )
    before = index.read_bytes()
    output = tmp_path / "manifest.json"
    freeze_source(ROOT, output)
    manifest = load(output)
    verify_source(ROOT, manifest)
    assert source_tree(ROOT, manifest["entries"]) == manifest["source_tree"]
    assert index.read_bytes() == before
    assert (
        manifest["source_commit"]
        == subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    )
    with pytest.raises(FileExistsError):
        write_new(output, {})
    output.write_text('{"key": 1, "key": 2}')
    with pytest.raises(InvalidReceipt, match="duplicate"):
        load(output)


def test_source_export_rejects_extra_or_omitted_production_inputs(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "python").mkdir()
    (tmp_path / "python/known.py").write_text("known")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    entries = entries_for(tmp_path)
    manifest = {
        "source_commit": "a" * 40,
        "source_tree": source_tree(tmp_path, entries),
        "entries": entries,
    }
    verify_source(tmp_path, manifest)
    (tmp_path / "python/extra.py").write_text("unexpected import source")
    with pytest.raises(InvalidReceipt, match="unmanifested"):
        verify_source(tmp_path, manifest)
    (tmp_path / "python/extra.py").unlink()
    manifest["entries"] = []
    with pytest.raises(InvalidReceipt, match="source mismatch"):
        verify_source(tmp_path, manifest)


def coordinated_fixture() -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    """Synthetic ordering only: every marker remains owned until tail."""
    device = {
        "domain": DEVICE,
        "initial_live": {},
        "events": [],
        "event_end": 0,
        "dropped_events": 0,
    }
    heap = {
        "domain": HOST,
        "initial_live": {},
        "events": [],
        "event_end": 0,
        "dropped_events": 0,
        "marker_positions": {},
    }
    boundaries = []
    for specification in schedule()[:-1]:
        name = specification["id"]
        heap["marker_positions"][name] = len(heap["events"])
        heap["events"].append(
            {
                "sequence": len(heap["events"]),
                "kind": "allocate",
                "allocation_id": name,
                "requested_bytes": 31,
            }
        )
        boundaries.append(
            (
                name,
                specification["phase"] in ("setup", "publication"),
                copy.deepcopy(device),
            )
        )
    for boundary in boundaries:
        heap["events"].append(
            {
                "sequence": len(heap["events"]),
                "kind": "release",
                "allocation_id": boundary[0],
                "requested_bytes": 31,
            }
        )
    heap["event_end"] = len(heap["events"])
    return heap, boundaries, device


def test_joint_coordinator_consumes_every_host_event_and_scopes_zero_assertion() -> (
    None
):
    heap, boundaries, device = coordinated_fixture()
    windows = assemble_windows(heap, boundaries, device)
    assert windows[0]["observations"][0]["event_end"] == 1
    assert windows[1]["observations"][0]["event_end"] == 1
    assert windows[2]["observations"][0]["metrics"]["allocation_count"] == 2
    assert windows[-1]["observations"][0]["event_end"] == len(heap["events"])
    assert windows[-1]["observations"][0]["metrics"]["live_bytes"] == 0
    identity = {
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "library_sha256": "c" * 64,
        "artifact_sha256": "d" * 64,
        "workload_sha256": "e" * 64,
        "toolchain": "protocol fixture",
        "device": "fixture",
        "endpoint": "fixture",
    }
    expected = {
        "schema": SCHEMA_V2,
        "identity": identity,
        "domains": [HOST, DEVICE],
        "windows": schedule(),
    }
    receipt = {
        "schema": SCHEMA_V2,
        "identity": identity,
        "windows": windows,
        "execution": {"kind": "runtime", "completed": True, "source_matched": True},
    }
    assert verify_receipt(receipt, expected)["status"] == "PASS"


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("omitted", "boundary"),
        ("side", "marker side"),
        ("dropped", "dropped"),
        ("leak", "leaked"),
    ],
)
def test_joint_coordinator_fails_closed(mutation: str, match: str) -> None:
    heap, boundaries, device = coordinated_fixture()
    if mutation == "omitted":
        del boundaries[1]
    elif mutation == "side":
        name, _, snapshot = boundaries[1]
        boundaries[1] = (name, True, snapshot)
    elif mutation == "dropped":
        heap["dropped_events"] = 1
    else:
        device["events"] = [
            {
                "sequence": 0,
                "kind": "allocate",
                "allocation_id": "leaked",
                "requested_bytes": 16,
            }
        ]
        device["event_end"] = 1
    with pytest.raises(InvalidReceipt, match=match):
        assemble_windows(heap, boundaries, device)


def matched_results() -> list[Any]:
    return [
        SimpleNamespace(
            items=(
                SimpleNamespace(
                    succeeded=True,
                    converged=True,
                    energy=-1.0,
                    forces=np.zeros((2, 3)),
                    iterations=2,
                ),
            )
        )
        for _ in ENDPOINTS
    ]


def test_numerical_gate_covers_changed_geometry_and_real_work_counts() -> None:
    evidence = numerical_evidence(matched_results(), matched_results())
    assert [row["id"] for row in evidence] == [row[0] for row in ENDPOINTS]
    assert evidence[-1]["force_error"] == 0
    assert all(
        row["iterations"] == row["reference_iterations"] == [2] for row in evidence
    )


def test_matched_ordinary_iteration_changes_are_not_silently_accepted() -> None:
    actual = matched_results()
    actual[1].items[0].iterations = 3
    with pytest.raises(InvalidReceipt, match="iteration counts differ"):
        numerical_evidence(actual, matched_results())


def test_independent_backend_counts_are_retained_without_equating_histories() -> None:
    actual = matched_results()
    actual[1].items[0].iterations = 3
    evidence = numerical_evidence(actual, matched_results(), match_iterations=False)
    assert evidence[1]["iterations"] == [3]
    assert evidence[1]["reference_iterations"] == [2]
    actual[-1].items[0].energy += 1e-5
    with pytest.raises(InvalidReceipt, match="energy acceptance"):
        numerical_evidence(actual, matched_results(), match_iterations=False)


@pytest.mark.parametrize("invalid", [True, -1, 1 << 64, 1.5])
def test_reference_iteration_counts_remain_uint64(invalid: Any) -> None:
    actual = matched_results()
    actual[1].items[0].iterations = invalid
    with pytest.raises(InvalidReceipt, match="actual iterations"):
        numerical_evidence(actual, matched_results(), match_iterations=False)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("convergence", "unconverged"),
        ("nan-energy", "energy"),
        ("nan-force", "force"),
        ("changed-geometry", "energy"),
    ],
)
def test_numerical_gate_rejects_failures(mutation: str, match: str) -> None:
    actual = matched_results()
    if mutation == "convergence":
        actual[0].items[0].converged = False
    elif mutation == "nan-energy":
        actual[0].items[0].energy = float("nan")
    elif mutation == "nan-force":
        actual[2].items[0].forces[0, 0] = float("nan")
    else:
        actual[-1].items[0].energy += 1e-5
    with pytest.raises(InvalidReceipt, match=match):
        numerical_evidence(actual, matched_results())
