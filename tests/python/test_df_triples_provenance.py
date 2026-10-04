"""Host-only cross-file frame admission for the supplied-state triples adapter."""

import json
import sys
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

from benchmarks import df_triples_native_probe as probe
from benchmarks.df_triples_provenance import (
    frame_identity,
    sha256,
    validate_manifest,
    write_manifest,
)


@pytest.fixture
def inputs(tmp_path: Path) -> dict:
    o, v, q = 1, 2, 3
    state, reference = tmp_path / "state.npz", tmp_path / "oracle.json"
    replay, manifest = tmp_path / "replay.bin", tmp_path / "frame.json"
    t1, t2 = (
        np.arange(o * v).reshape(o, v) / 100,
        np.arange(o * o * v * v).reshape(o, o, v, v) / 100,
    )
    arrays = {
        "coefficients": np.eye(o + v),
        "orbital_energies": np.array([-0.5, 0.2, 0.7]),
        "t1": t1,
        "t2": t2,
        "metric_whitening": np.eye(q),
    }
    np.savez(state, **arrays)
    record = {
        "atoms_angstrom": [["He", [0.0, 0.0, 0.0]]],
        "basis": "test",
        "auxiliary_basis": "test-aux",
        "basis_definition": {"He": [[0, [1.0, 1.0]]]},
        "auxiliary_basis_definition": {"He": [[0, [0.5, 1.0]]]},
        "basis_representation": "spherical",
        "orbital_order": "occupied_then_virtual",
        "reference_mode": "conventional-unscreened-rhf",
        "metric_policy": "symmetric inverse square root",
        "metric_relative_threshold": 1e-10,
        "metric_rank": q,
        "nocc": o,
        "nvir": v,
        "naux": q,
        "triples_energy": -0.01,
        "state_sha256": sha256(state),
    }
    reference.write_text(json.dumps(record))
    frame = frame_identity(state, reference)
    sizes = [
        o * o,
        o * v,
        v * v,
        o * o * v * v,
        o * o * v * v,
        o * o * v * v,
        o**3 * v,
        o**4,
        o * v,
        o * o * v * v,
    ]
    with replay.open("wb") as stream:
        np.array([o, v, q, 1 << 30, 100, 6, 1], dtype="<u8").tofile(stream)
        for size in sizes:
            np.zeros(size, dtype="<f8").tofile(stream)
        t1.astype("<f8").tofile(stream)
        t2.astype("<f8").tofile(stream)
        np.zeros(q * o * v + q * v * v, dtype="<f8").tofile(stream)
    write_manifest(manifest, replay, state, reference, source_frame=frame)
    return {
        "state": state,
        "reference": reference,
        "input": replay,
        "manifest": manifest,
        "arrays": arrays,
        "record": record,
        "frame": frame,
        "output": tmp_path / "result.json",
    }


def _invoke(
    inputs: dict, monkeypatch: pytest.MonkeyPatch, *, reaches_probe: bool
) -> None:
    calls = []

    def load_library(*args: object, **kwargs: object) -> NoReturn:
        calls.append(args)
        raise RuntimeError("reached CUDA load boundary")

    monkeypatch.setattr(probe.ct, "CDLL", load_library)
    argv = ["df_triples_native_probe"]
    for key in ("input", "state", "reference", "manifest", "output"):
        argv.extend(["--" + key, str(inputs[key])])
    argv.extend(["--library", "unused-library", "--probe", "unused-probe"])
    monkeypatch.setattr(sys, "argv", argv)
    inputs["output"].write_text("sentinel")
    if reaches_probe:
        with pytest.raises(RuntimeError, match="reached CUDA load boundary"):
            probe.main()
        assert len(calls) == 1
    else:
        with pytest.raises(ValueError):
            probe.main()
        assert calls == []
    assert inputs["output"].read_text() == "sentinel"


def test_matching_frame_reaches_probe(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = validate_manifest(
        *(inputs[key] for key in ("manifest", "input", "state", "reference"))
    )
    assert manifest["frame"] == inputs["frame"]
    assert manifest["upstream_factor_acceptance"] == "unqualified"
    _invoke(inputs, monkeypatch, reaches_probe=True)


@pytest.mark.parametrize(
    "field", ["orbital_energies", "coefficients", "t1", "t2", "metric_whitening"]
)
def test_changed_state_rejected_before_cuda(
    inputs: dict, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    arrays = inputs["arrays"]
    if field == "orbital_energies":
        # Finite and still canonical, with the same dimensions and occupied gap.
        arrays[field] = np.array([-0.6, 0.25, 0.8])
    else:
        arrays[field] = arrays[field].copy()
        arrays[field].flat[0] += 0.01
    np.savez(inputs["state"], **arrays)
    _invoke(inputs, monkeypatch, reaches_probe=False)


@pytest.mark.parametrize(
    "field",
    [
        "atoms_angstrom",
        "basis_definition",
        "basis_representation",
        "metric_relative_threshold",
        "metric_rank",
    ],
)
def test_changed_reference_frame_rejected(
    inputs: dict, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    record = inputs["record"]
    record[field] = {
        "atoms_angstrom": [["He", [0.1, 0.0, 0.0]]],
        "basis_definition": {"He": [[0, [2.0, 1.0]]]},
        "basis_representation": "cartesian",
        "metric_relative_threshold": 1e-9,
        "metric_rank": 2,
    }[field]
    inputs["reference"].write_text(json.dumps(record))
    _invoke(inputs, monkeypatch, reaches_probe=False)


def test_changed_replay_rejected_before_cuda(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    with inputs["input"].open("r+b") as stream:
        stream.seek(-8, 2)
        np.array([0.25], dtype="<f8").tofile(stream)
    _invoke(inputs, monkeypatch, reaches_probe=False)


def test_source_producer_rejects_changed_frame(inputs: dict) -> None:
    arrays = inputs["arrays"]
    arrays["orbital_energies"] = np.array([-0.6, 0.25, 0.8])
    np.savez(inputs["state"], **arrays)
    inputs["record"]["state_sha256"] = sha256(inputs["state"])
    inputs["reference"].write_text(json.dumps(inputs["record"]))
    with pytest.raises(ValueError, match="source-construction frame"):
        write_manifest(
            *(inputs[key] for key in ("manifest", "input", "state", "reference")),
            source_frame=inputs["frame"],
        )


def test_replay_amplitudes_must_match_frame(inputs: dict) -> None:
    o, v = 1, 2
    with inputs["input"].open("r+b") as stream:
        prefix = o * o + o * v + v * v + 4 * o * o * v * v + o**3 * v + o**4 + o * v
        stream.seek(56 + 8 * prefix)
        np.array([0.25], dtype="<f8").tofile(stream)
    with pytest.raises(ValueError, match="replay t1"):
        write_manifest(
            *(inputs[key] for key in ("manifest", "input", "state", "reference")),
            source_frame=inputs["frame"],
        )


def test_legacy_reference_not_retroactively_certified(inputs: dict) -> None:
    del inputs["record"]["basis_definition"]
    inputs["reference"].write_text(json.dumps(inputs["record"]))
    with pytest.raises(ValueError, match="explicit source-frame metadata"):
        frame_identity(inputs["state"], inputs["reference"])


def test_manifest_cannot_promote_factors(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = json.loads(inputs["manifest"].read_text())
    manifest["upstream_factor_acceptance"] = "passed"
    inputs["manifest"].write_text(json.dumps(manifest))
    _invoke(inputs, monkeypatch, reaches_probe=False)
