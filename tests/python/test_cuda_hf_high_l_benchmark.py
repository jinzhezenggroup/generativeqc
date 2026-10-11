"""Host-only contracts for complete CUDA HF endpoint qualification receipts."""

import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def driver() -> ModuleType:
    """Import metadata and validation without loading a runtime or GPU oracle."""
    path = ROOT / "benchmarks/cuda_hf_high_l.py"
    spec = importlib.util.spec_from_file_location("cuda_hf_high_l_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "case", ("water-tzvp", "water-dimer-tzvp", "formaldehyde-tzvp")
)
def test_high_l_cases_preserve_basis_and_atom_count(
    driver: ModuleType, case: str
) -> None:
    """The larger workload must not silently become the small water fixture."""
    atoms, basis = driver.fixture(case)
    assert basis == "def2-tzvp"
    assert (
        len(atoms)
        == {"water-tzvp": 3, "water-dimer-tzvp": 6, "formaldehyde-tzvp": 4}[case]
    )


def test_source_domain_distinguishes_logical_bounds_from_executed_work(
    driver: ModuleType,
) -> None:
    """Count all Cartesian component orbits but never fabricate device counts."""
    shells = [SimpleNamespace(atom_index=0, angular_momentum=3, primitives=(1, 2))]
    domain = driver.source_domain(shells, "spherical")
    assert domain["public_aos"] == 7
    assert domain["cartesian_aos"] == 10
    assert domain["f_shells"] == 1
    assert domain["unscreened_cartesian_canonical_quartets"] == 55 * 56 // 2
    assert domain["unscreened_primitive_component_products"] == 55 * 56 // 2 * 16
    assert domain["executed_primitive_component_evaluations"] is None


@pytest.mark.parametrize(
    "field,value",
    (
        ("executed_backend", "cpu_reference"),
        ("converged", False),
        ("energy_error", 1.01e-8),
        ("energy_error", math.nan),
        ("force_error", 1.01e-7),
        ("force_error", math.nan),
    ),
)
def test_numerical_gates_reject_substitution_and_nonfinite_values(
    driver: ModuleType, field: str, value: object
) -> None:
    """Missing CUDA execution and nonfinite independent errors cannot pass."""
    sample = {
        "executed_backend": "cuda",
        "converged": True,
        "energy_error": 1e-10,
        "force_error": 1e-10,
    }
    sample[field] = value
    with pytest.raises(RuntimeError, match="acceptance failed"):
        driver.validate_sample(sample, True)


def test_energy_only_gate_does_not_claim_force_qualification(
    driver: ModuleType,
) -> None:
    """Energy receipts permit absent forces and retain an explicit null error."""
    sample = {
        "executed_backend": "cuda",
        "converged": True,
        "energy_error": 1e-10,
        "force_error": None,
    }
    driver.validate_sample(sample, False)


def test_nonfinite_failure_receipt_stays_valid_json_without_fabricating_errors(
    driver: ModuleType,
) -> None:
    """Null failure fields are not silently replaced with successful zero errors."""
    report = {
        "status": "fail",
        "samples": [{"energy_error": math.nan, "force_error": math.inf}],
        "failure": "nonfinite measurement",
    }
    safe = driver.json_safe(report)
    assert safe["status"] == "fail"
    assert safe["samples"] == [{"energy_error": None, "force_error": None}]
    assert safe["failure"] == "nonfinite measurement"
