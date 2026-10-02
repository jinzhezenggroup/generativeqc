"""Host regressions for frozen orbital/auxiliary inputs used by DF GPU entries."""

import importlib
import json
from pathlib import Path

import pytest
from generativeqc import load_basis
from generativeqc.profiles import canonical_hash
from test_df_practical_response_cuda import _practical_model

from benchmarks._cases import benchmark_cases
from benchmarks._retained_basis import load_retained_basis, retained_basis_record

IDENTITY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/issue206-practical-auxiliary/identity"
)


@pytest.mark.parametrize("name", ["cc-pvdz", "cc-pvdz-jkfit"])
def test_retained_basis_loads_without_changing_data(name: str) -> None:
    source = IDENTITY / f"{name}.json"
    original_bytes = source.read_bytes()
    original = json.loads(original_bytes)
    with pytest.raises(ValueError, match="unsupported canonical basis schema"):
        load_basis(source)

    with retained_basis_record(source) as record:
        adapted = json.loads(record.read_text())
        basis = load_basis(record)
    assert not record.exists()
    assert load_retained_basis(source) == basis
    assert adapted["schema"] == "generativeqc.basis"
    assert adapted["checksum"] != original["checksum"]
    assert json.loads(json.dumps(basis.to_payload())) == adapted
    for metadata in ("schema", "checksum"):
        original.pop(metadata)
        adapted.pop(metadata)
    assert adapted == original
    assert source.read_bytes() == original_bytes


@pytest.mark.parametrize("name", ["cc-pvdz", "cc-pvdz-jkfit"])
@pytest.mark.parametrize("tamper", ["decimal", "checksum"])
def test_retained_basis_rejects_tampering_before_translation(
    tmp_path: Path, name: str, tamper: str
) -> None:
    payload = json.loads((IDENTITY / f"{name}.json").read_text())
    if tamper == "decimal":
        payload["elements"][0]["shells"][0]["exponents"][0] = "1.2345"
    else:
        payload.pop("checksum")
    source = tmp_path / "tampered.json"
    source.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="retained basis fixture checksum mismatch"):
        load_retained_basis(source)


@pytest.mark.parametrize("schema", ["generativeqc.basis", "unknown.basis"])
def test_retained_basis_only_translates_expected_legacy_schema(
    tmp_path: Path, schema: str
) -> None:
    payload = json.loads((IDENTITY / "cc-pvdz-jkfit.json").read_text())
    payload.pop("checksum")
    payload["schema"] = schema
    payload["checksum"] = canonical_hash(payload)
    source = tmp_path / "other-schema.json"
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported retained basis fixture schema"):
        load_retained_basis(source)


@pytest.mark.parametrize(
    "case_name", ["oh-def2-svp-spherical-uhf", "water-tetramer-def2-svp-spherical"]
)
def test_practical_response_model_loads_retained_inputs(case_name: str) -> None:
    case, orbital, auxiliary, cpu_orbital, cpu_auxiliary = _practical_model(case_name)
    assert case == benchmark_cases()[case_name]
    assert orbital.name == "cc-pvdz"
    assert auxiliary.name == "cc-pvdz-jkfit"
    assert set(cpu_orbital) == set(cpu_auxiliary) == {"H", "N", "O"}


@pytest.mark.parametrize(
    "module_name,compute_forces",
    [("readme_method_endpoints", False), ("compare_df_direct_endpoint", True)],
)
def test_benchmark_retained_auxiliary_input(
    module_name: str, compute_forces: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = importlib.import_module(f"benchmarks.{module_name}")
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    case = benchmark_cases()["water-tetramer-def2-svp-spherical"]
    auxiliary, reference = benchmark.load_retained_comparison_basis(
        benchmark.AUXILIARY, case, role="auxiliary", compute_forces=compute_forces
    )
    assert auxiliary.name == "cc-pvdz-jkfit"
    assert set(reference) == {"H", "N", "O"}
