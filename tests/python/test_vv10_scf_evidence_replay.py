"""Retained VV10 qualification must fail closed on incomplete evidence."""

import copy
import hashlib
import importlib.util
import json
import lzma
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/vv10-scf-features-20261106"


@pytest.fixture
def evidence() -> dict[str, Any]:
    return json.loads(lzma.decompress((BUNDLE / "evidence.json.xz").read_bytes()))


@pytest.fixture
def reference(evidence: dict[str, Any]) -> dict[str, Any]:
    return json.loads(
        lzma.decompress((ROOT / evidence["reference"]["path"]).read_bytes())
    )


def replay(
    evidence: dict[str, Any],
    reference: dict[str, Any],
    tmp_path: Path,
) -> dict[str, Any]:
    """Run the real replay on independently mutated, checksummed input files."""
    directory = tmp_path / BUNDLE.relative_to(ROOT)
    directory.mkdir(parents=True)
    reference_path = tmp_path / evidence["reference"]["path"]
    reference_path.parent.mkdir(parents=True)
    reference_bytes = lzma.compress(json.dumps(reference).encode(), preset=0)
    reference_path.write_bytes(reference_bytes)
    evidence["reference"]["sha256"] = hashlib.sha256(reference_bytes).hexdigest()
    (directory / "evidence.json.xz").write_bytes(
        lzma.compress(json.dumps(evidence).encode(), preset=0)
    )
    spec = importlib.util.spec_from_file_location(
        "vv10_scf_replay", BUNDLE / "replay.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = str(directory / "replay.py")
    return module.replay()


def test_complete_evidence_reproduces_retained_summary(
    evidence: dict[str, Any], reference: dict[str, Any], tmp_path: Path
) -> None:
    assert replay(evidence, reference, tmp_path) == json.loads(
        (BUNDLE / "summary.json").read_text()
    )


@pytest.mark.parametrize("count", [0, 1, 3])
def test_reference_sample_population_is_required(
    evidence: dict[str, Any], reference: dict[str, Any], tmp_path: Path, count: int
) -> None:
    record = reference["records"]["A100/reference"]
    record["samples"] = (record["samples"] * 2)[:count]
    with pytest.raises(AssertionError, match="independent reference samples"):
        replay(evidence, reference, tmp_path)


@pytest.mark.parametrize("field,value", [("status", "failed"), ("arm", "native")])
def test_complete_independent_reference_record_is_required(
    evidence: dict[str, Any],
    reference: dict[str, Any],
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    reference["records"]["A100/reference"][field] = value
    with pytest.raises(AssertionError):
        replay(evidence, reference, tmp_path)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize(
    "change",
    [
        {"converged": False},
        {"phase": "missing"},
        {"energy": float("nan")},
        {"forces": []},
        {"forces": [[0.0, 0.0]] * 96},
        {"forces": [[0.0, 0.0, float("inf")]] * 96},
    ],
)
def test_reference_samples_must_be_complete_and_finite(
    evidence: dict[str, Any],
    reference: dict[str, Any],
    tmp_path: Path,
    index: int,
    change: dict[str, Any],
) -> None:
    reference["records"]["A100/reference"]["samples"][index].update(change)
    with pytest.raises(AssertionError):
        replay(evidence, reference, tmp_path)


@pytest.mark.parametrize(
    "devices", [[], ["A100"], ["RTX5090"], ["A100", "RTX5090", "other"]]
)
def test_integration_requires_exactly_both_devices(
    evidence: dict[str, Any],
    reference: dict[str, Any],
    tmp_path: Path,
    devices: list[str],
) -> None:
    endpoints = evidence["latest_integration"]["endpoints"]
    evidence["latest_integration"]["endpoints"] = {
        device: copy.deepcopy(endpoints.get(device, endpoints["A100"]))
        for device in devices
    }
    with pytest.raises(AssertionError):
        replay(evidence, reference, tmp_path)


@pytest.mark.parametrize("missing", [False, True])
def test_integration_section_cannot_be_omitted(
    evidence: dict[str, Any], reference: dict[str, Any], tmp_path: Path, missing: bool
) -> None:
    if missing:
        del evidence["latest_integration"]
    else:
        evidence["latest_integration"] = None
    with pytest.raises((AssertionError, KeyError)):
        replay(evidence, reference, tmp_path)


@pytest.mark.parametrize("device", ["A100", "RTX5090"])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_integration_cold_and_warm_samples_are_required(
    evidence: dict[str, Any],
    reference: dict[str, Any],
    tmp_path: Path,
    device: str,
    count: int,
) -> None:
    record = evidence["latest_integration"]["endpoints"][device]["record"]
    record["samples"] = (record["samples"] * 2)[:count]
    with pytest.raises(AssertionError):
        replay(evidence, reference, tmp_path)


@pytest.mark.parametrize("observable", ["energy", "forces"])
def test_independent_reference_errors_remain_gated(
    evidence: dict[str, Any],
    reference: dict[str, Any],
    tmp_path: Path,
    observable: str,
) -> None:
    target = reference["records"]["A100/reference"]["samples"][0]
    if observable == "energy":
        target["energy"] += 1.0
    else:
        target["forces"][0][0] += 1e-3
    with pytest.raises(AssertionError):
        replay(evidence, reference, tmp_path)
