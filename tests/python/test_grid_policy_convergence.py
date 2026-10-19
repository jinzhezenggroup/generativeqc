"""Production grid accuracy/cost gate against an independently denser quadrature."""

import pytest

from benchmarks.grid_policy_convergence import qualify


def test_production_grid_profiles_converge_against_dense_pyscf() -> None:
    pytest.importorskip(
        "pyscf", reason="independent grid-convergence oracle requires PySCF"
    )
    result = qualify()
    assert result["version"] == 2
    assert result["passed"]
    assert result["profiles"]["pbe-standard"]["passed"]
    assert result["profiles"]["pbe-tight"]["passed"]
    assert result["accuracy_cost_checks"]["pbe_standard_improves_radial48_force"]
    for family in ("lda", "pbe"):
        legacy = result["profiles"][f"{family}-legacy-v1"]["spec"]
        assert legacy["version"] == 1
        assert legacy["element_radii"] == ()
    radial = result["profiles"]["pbe-radial48"]["spec"]
    assert radial["version"] == 2
    assert radial["element_radii"]


def test_opt_in_pruning_meets_the_same_independent_energy_and_force_gates() -> None:
    """Point reduction is accepted only alongside grid-converged SCF gradients."""
    pytest.importorskip("pyscf", reason="independent pruning oracle requires PySCF")
    result = qualify(include_pruning=True)
    assert result["version"] == 3 and result["passed"]
    for name in ("lda-standard", "lda-tight", "pbe-standard", "pbe-tight"):
        candidate = result["profiles"][f"{name}-pruned"]
        assert candidate["passed"]
        assert candidate["spec"]["pruning"] == "radial-v1"
        assert candidate["gate"] == result["profiles"][name]["gate"]
        assert candidate["point_fraction_of_unpruned"] < 1
