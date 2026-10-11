"""Fail-closed scientific/shape gates without loading CUDA or PySCF."""

import copy
import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import structured_scf_endpoint_qualification as qualification


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
def test_binary_provenance_includes_driver_or_force_companions(
    tmp_path: Path, method: str
) -> None:
    library, driver = tmp_path / "libgenerativeqc.so", tmp_path / "driver"
    library.write_bytes(b"main library")
    driver.write_bytes(b"native HF driver")
    for suffix in ("", "_spd"):
        stem = f"generativeqc_stationary_pbe_{method}{suffix}"
        (tmp_path / f"lib{stem}.so").write_bytes(b"force companion")
        (tmp_path / f"{stem}.json").write_text("{}")
    before = qualification.artifact_hashes(library, driver, method)
    assert len(before) == (2 if method.endswith("hf") else 5)
    target = (
        driver
        if method.endswith("hf")
        else tmp_path / f"libgenerativeqc_stationary_pbe_{method}_spd.so"
    )
    target.write_bytes(b"replaced binary")
    assert qualification.artifact_hashes(library, driver, method) != before


def sample() -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
    reference = {
        "energy": -1.0,
        "density": [[[1.0]]],
        "fock": [[[-0.5]]],
        "forces": [[0.0, 0.0, 0.0]],
        "oracle_accepted": True,
    }
    actual = {
        **copy.deepcopy(reference),
        "converged": True,
        "status": 0,
        "backend": "cuda",
        "final_state_gates": {"accepted": True},
    }
    return actual, reference


def test_matched_complete_state_passes() -> None:
    actual, reference = sample()
    assert qualification.assess(actual, reference)["accepted"]


@pytest.mark.parametrize("field", qualification.GATES)
def test_each_independent_gate_is_required(field: str) -> None:
    actual, reference = sample()
    if field == "energy":
        actual[field] += 2 * qualification.GATES[field]
    else:
        actual[field][0][0] = 99.0
    assert not qualification.assess(actual, reference)["accepted"]


@pytest.mark.parametrize("field", qualification.GATES)
@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), []])
def test_invalid_or_missing_values_cannot_qualify(
    field: str, invalid: typing.Any
) -> None:
    actual, reference = sample()
    actual[field] = invalid
    assert not qualification.assess(actual, reference)["accepted"]


@pytest.mark.parametrize("field,value", [("status", 1), ("converged", False)])
def test_solver_failure_cannot_supply_performance_evidence(
    field: str, value: int | bool
) -> None:
    actual, reference = sample()
    actual[field] = value
    assert not qualification.assess(actual, reference)["accepted"]


def test_failed_reference_is_not_an_oracle() -> None:
    actual, reference = sample()
    reference["oracle_accepted"] = False
    assert not qualification.assess(actual, reference)["accepted"]


def test_final_state_proof_is_mandatory() -> None:
    actual, reference = sample()
    actual.pop("final_state_gates")
    assert not qualification.assess(actual, reference)["accepted"]


def test_native_orbitals_are_column_major_per_system_and_spin() -> None:
    coefficients = np.arange(24).reshape(2, 3, 2, 2)
    native = coefficients.swapaxes(-1, -2).reshape(-1).tolist()
    np.testing.assert_array_equal(
        qualification.coefficient_matrices(native, 2, 3, 2), coefficients
    )


@pytest.mark.parametrize("field", qualification.GATES)
def test_nonfinite_reference_is_rejected_without_nan_errors(field: str) -> None:
    actual, reference = sample()
    reference[field] = float("nan") if field == "energy" else [[[float("nan")]]]
    result = qualification.assess(actual, reference)
    assert result["accepted"] is False
    assert result["errors"][field] is None


@pytest.mark.parametrize("case", ["through-d-cartesian", "through-d-spherical"])
def test_admitted_spd_fixture_preserves_rectangular_transform_domain(case: str) -> None:
    spec = qualification.model(case, "rks")
    _, _, inputs, _ = qualification.basis_and_inputs(spec)
    assert max(shell["angular_momentum"] for shell in inputs["shells"]) == 2
    assert len(inputs["atomic_numbers"]) == 2
    assert inputs["charge"] == 1


def test_large_spd_fixture_does_not_silently_change_the_f_fixture() -> None:
    spd = qualification.model("large-spd-cartesian", "rks")
    original = qualification.model("large-cartesian", "rks")
    assert spd["basis"] == "def2-svp"
    assert original["basis"] == "def2-tzvp"
    assert spd["coordinates"] == original["coordinates"]


def test_transport_is_detached_from_the_prepared_owner_not_batch_item(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import generativeqc

    def hook(mask: int) -> bool:
        return -1 <= mask <= 127

    def diagnostic() -> bytes:
        return b"{}"

    item = SimpleNamespace(
        status=3,
        status_message="failed solve",
        converged=False,
        executed_backend="cuda",
        energy=-1.0,
        forces=None,
        iterations=1,
        ks_diagnostic=None,
    )
    transport = SimpleNamespace(to_payload=lambda: {"iterations": 1})

    class Prepared:
        resource_diagnostics = None
        ks_transport_diagnostics = (transport,)

        def __enter__(self) -> typing.Any:
            return self

        def __exit__(self, *_arguments: object) -> None:
            pass

        def execute(
            self, *_arguments: typing.Any, **_keywords: typing.Any
        ) -> SimpleNamespace:
            return SimpleNamespace(items=(item,))

    calculator = SimpleNamespace(
        _library=SimpleNamespace(
            tensor_structured_scf_qualification_for_test_v1=hook,
            tensor_structured_scf_preparation_diagnostic_for_test_v1=diagnostic,
        ),
        prepare_batch=lambda *_arguments, **_keywords: Prepared(),
    )
    monkeypatch.setattr(generativeqc, "Calculator", lambda **_keywords: calculator)
    rows = qualification.ks_probe(
        qualification.model("small", "rks"), tmp_path, tmp_path, 127, 1
    )
    assert len(rows) == 5
    assert all(row["transport_diagnostic"] == {"iterations": 1} for row in rows)
