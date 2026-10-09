"""Protect independent, all-repeat gates for the benchmark-only DF density handoff."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from benchmarks.df_hf_preconvergence import (
    correlation_audit,
    maximum_difference,
    oracle,
    prepare_case,
    prepare_jk,
    summarize,
)


def sample(mode: str = "direct", endpoint: str = "forces") -> dict:
    """Minimal converged exact state, including physical rather than DIIS residuals."""
    return {
        "mode": mode,
        "endpoint": endpoint,
        "pre_tolerance": 1e-6,
        "reference_energy": -1.0,
        "energy": -1.1,
        "density": [2.0, 0.0],
        "orbital_energies": [-0.5, 0.2],
        "forces": [0.01, -0.01],
        "energy_change": 1e-14,
        "density_rms": 1e-12,
        "commutator_residual": 1e-12,
        "canonical_density_drift": 1e-12,
        "eigen_residual": 1e-12,
        "orthogonality_max": 1e-14,
        "converged": True,
        "pre_fallback": False,
        "seed_fallback": False,
        "rhf_seconds": 2.0,
        "endpoint_seconds": 3.0,
        "direct_iterations": 4,
        "direct_fock_builds": 6,
        "pre_seconds": 0.1,
        "pre_iterations": 3,
    }


def test_changed_geometry_and_budget_fixture_metadata(tmp_path: Path) -> None:
    """Changed geometry and resource controls must not silently change orbital shells."""
    original = (
        (
            Path(__file__).resolve().parents[2]
            / "benchmarks/results/df-lambda-gemm-20261004/ethane230.input"
        )
        .read_text()
        .splitlines()
    )
    for case in ("moved", "budget8"):
        output = tmp_path / f"{case}.input"
        prepare_case(case, output)
        actual = output.read_text().splitlines()
        assert actual[9:] == original[9:]
        if case == "moved":
            assert float(actual[2].split()[1]) - float(
                original[2].split()[1]
            ) == pytest.approx(0.02)
            assert actual[0] == original[0]
        else:
            assert actual[1:9] == original[1:9]
            assert actual[0].split()[3] == str(8 << 30)


@pytest.mark.parametrize(
    "case,functions,electrons",
    [
        ("propane322", 322, 26),
        ("butane414", 414, 34),
        ("methane34", 34, 10),
        ("ethane58", 58, 18),
        ("ethane144", 144, 18),
    ],
)
def test_larger_fixture_raw_basis_roundtrip(
    tmp_path: Path, case: str, functions: int, electrons: int
) -> None:
    """Qualification inputs round-trip raw basis metadata without an oracle computation."""
    pytest.importorskip("pyscf")
    from benchmarks.df_hf_preconvergence import read_molecule

    output = tmp_path / f"{case}.input"
    prepare_case(case, output)
    molecule = read_molecule(output)
    assert molecule.nao_nr() == functions
    assert molecule.nelectron == electrons


def records(tmp_path: Path, *rows: dict) -> list[Path]:
    """Write private raw records, without publishing benchmark evidence."""
    paths = []
    for index, row in enumerate(rows):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(row))
        paths.append(path)
    return paths


def test_all_repeat_pairs_are_gated_even_with_different_iteration_counts(
    tmp_path: Path,
) -> None:
    inaccurate = sample("df-direct")
    inaccurate["forces"] = [0.01001, -0.01]
    inaccurate["direct_iterations"] = 100
    result = summarize(
        records(tmp_path, sample(), sample("df-direct"), inaccurate), None
    )
    assert not result["all_gates_pass"]
    assert result["groups"]["forces/pre-1e-06"]["maximum_errors"]["forces"] > 3e-9


@pytest.mark.parametrize(
    "field,value",
    [
        ("commutator_residual", 1e-5),
        ("canonical_density_drift", 1e-5),
        ("eigen_residual", 1e-5),
        ("orthogonality_max", 1e-5),
        ("density_rms", 1e-8),
        ("energy_change", 1e-8),
        ("converged", False),
    ],
)
def test_physical_reference_audit_cannot_be_replaced_by_energy_agreement(
    tmp_path: Path, field: str, value: float | bool
) -> None:
    candidate = sample("df-direct")
    candidate[field] = value
    assert not summarize(records(tmp_path, sample(), candidate), None)["all_gates_pass"]


def test_independent_oracle_failure_rejects_both_native_paths(tmp_path: Path) -> None:
    paths = records(tmp_path, sample(), sample("df-direct"))
    independent = tmp_path / "oracle.json"
    independent.write_text(
        json.dumps(
            {
                "reference_energy": -1.01,
                "density": [2.0, 0.0],
                "orbital_energies": [-0.5, 0.2],
            }
        )
    )
    assert not summarize(paths, independent)["all_gates_pass"]


def test_uninstrumented_retry_does_not_fabricate_work_count(tmp_path: Path) -> None:
    candidate = sample("df-direct")
    candidate["seed_fallback"] = True
    candidate["direct_fock_builds"] = None
    result = summarize(records(tmp_path, sample(), candidate), None)
    group = result["groups"]["forces/pre-1e-06"]
    assert group["candidate_median"]["direct_fock_builds"] is None
    assert group["fallbacks"] == 1


def test_energy_only_force_error_remains_absent(tmp_path: Path) -> None:
    result = summarize(
        records(tmp_path, sample(endpoint="energy"), sample("df-direct", "energy")),
        None,
    )
    assert result["all_gates_pass"]
    assert result["groups"]["energy/pre-1e-06"]["maximum_errors"]["forces"] is None


@pytest.mark.parametrize("other", [[], [1.0], [float("nan"), 0.0]])
def test_truncated_or_nonfinite_states_cannot_pass(other: list[float]) -> None:
    with pytest.raises(ValueError):
        maximum_difference([2.0, 0.0], other)


def test_nonfinite_energy_cannot_hide_behind_an_accurate_repeat(tmp_path: Path) -> None:
    invalid = sample("df-direct")
    invalid["energy"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite final"):
        summarize(records(tmp_path, sample(), sample("df-direct"), invalid), None)


def test_correlation_oracle_angstrom_schema_and_force_sign(tmp_path: Path) -> None:
    """The independent derivative sign is opposite the returned molecular force."""
    independent = tmp_path / "cc-oracle.json"
    independent.write_text(
        json.dumps(
            {
                "pyscf_version": "2.14.0",
                "nbf": 2,
                "naux": 2,
                "total_energy": -1.1,
                "atoms_angstrom": [
                    ["H", [0.0, 0.0, 0.0]],
                    ["C", [0.52917721092, 0.0, 0.0]],
                ],
            }
        )
    )
    finite = tmp_path / "finite.json"
    rows = []
    for step in (1e-4, 3e-5):
        rows.append(
            {
                "step_bohr": step,
                "minus": {
                    "pyscf_version": "2.14.0",
                    "atoms_bohr": [[1, [-step, 0.0, 0.0]], [6, [1.0, 0.0, 0.0]]],
                    "total_energy": -1.1 + 0.01 * step,
                },
                "plus": {
                    "pyscf_version": "2.14.0",
                    "atoms_bohr": [[1, [step, 0.0, 0.0]], [6, [1.0, 0.0, 0.0]]],
                    "total_energy": -1.1 - 0.01 * step,
                },
            }
        )
    finite.write_text(json.dumps({"rows": rows}))
    row = sample()
    row.update(
        {
            "nbf": 2,
            "naux_correlation": 2,
            "forces": [0.01, 0.0, 0.0, -0.01, 0.0, 0.0],
            "lambda_residual": 1e-12,
            "z_residual": 1e-12,
            "stationarity": 1e-12,
        }
    )
    assert correlation_audit([row], independent, finite)["gates_pass"]
    row["forces"] = [-0.01, 0.0, 0.0, 0.01, 0.0, 0.0]
    assert not correlation_audit([row], independent, finite)["gates_pass"]
    row["naux_correlation"] = 3
    with pytest.raises(ValueError, match="dimensions differ"):
        correlation_audit([row], independent, finite)


@pytest.mark.parametrize(
    "mode,endpoint,tolerance,limit,message",
    [
        ("df", "hf", "1e-6", "50", "invalid mode"),
        ("df-direct", "gradient", "1e-6", "50", "invalid endpoint"),
        ("df-direct", "hf", "nan", "50", "invalid preconvergence tolerance"),
        ("df-direct", "hf", "1e-6x", "50", "invalid preconvergence tolerance"),
        ("df-direct", "hf", "1e-6", "0", "invalid preconvergence iteration limit"),
        ("df-direct", "hf", "1e-6", "151", "invalid preconvergence iteration limit"),
    ],
)
def test_invalid_native_controls_fail_before_gpu_setup(
    mode: str, endpoint: str, tolerance: str, limit: str, message: str
) -> None:
    binary = os.environ.get("GENERATIVEQC_DF_PRECONVERGENCE_BINARY")
    if not binary:
        pytest.skip("requires the compiled preconvergence experiment")
    result = subprocess.run(
        [
            binary,
            "/nonexistent/input",
            "/nonexistent/jk",
            "/nonexistent/output",
            mode,
            endpoint,
            tolerance,
            limit,
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert message in result.stderr


def test_native_density_handoff_and_failed_preconvergence(tmp_path: Path) -> None:
    """Exercise actual native HF and E+F, including the bounded cold fallback."""
    binary = os.environ.get("GENERATIVEQC_DF_PRECONVERGENCE_BINARY")
    if not binary or os.environ.get("GENERATIVEQC_DF_PRECONVERGENCE_CUDA_TEST") != "1":
        pytest.skip("requires the explicit Slurm CUDA experiment gate")
    source = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/df-lambda-transpose-actions/methane34.input"
    )
    jk = tmp_path / "jk.shells"
    prepare_jk(source, jk, "def2-universal-jkfit")
    reference = tmp_path / "oracle.json"
    oracle(source, reference, 1)
    paths = []
    for endpoint in ("hf", "forces"):
        for mode in ("direct", "df-direct"):
            path = tmp_path / f"{mode}-{endpoint}.json"
            subprocess.run(
                [binary, str(source), str(jk), str(path), mode, endpoint],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            paths.append(path)
    assert summarize(paths, reference)["all_gates_pass"]
    automatic = tmp_path / "auto-forces.json"
    subprocess.run(
        [binary, str(source), "-", str(automatic), "auto-direct", "forces"],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    automatic_result = json.loads(automatic.read_text())
    assert automatic_result["pre_iterations"] == 0
    assert automatic_result["pre_seconds"] == 0
    assert not automatic_result["pre_fallback"]
    assert summarize([*paths, automatic], reference)["all_gates_pass"]
    refused = tmp_path / "refused.json"
    subprocess.run(
        [binary, str(source), str(jk), str(refused), "df-direct", "hf", "1e-6", "1"],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    result = json.loads(refused.read_text())
    assert result["pre_fallback"]
    assert not result["pre_converged"]
    direct = json.loads(paths[0].read_text())
    assert abs(result["reference_energy"] - direct["reference_energy"]) < 1e-10
    assert result["direct_fock_builds"] == direct["direct_fock_builds"]
