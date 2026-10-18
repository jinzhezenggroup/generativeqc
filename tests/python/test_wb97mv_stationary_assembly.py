"""Independent physical source sums for native stationary WB97M-V composition."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.method.wb97mv_stationary_assembly import assembly_maps
from generativeqc_compiler.tensor import execute

from tools.generate_wb97mv_stationary_assembly import generate


@pytest.mark.parametrize("spins", [1, 2])
def test_physical_spin_weights_and_frozen_rhs(spins: int) -> None:
    random = np.random.default_rng(1527 + spins)
    programs = assembly_maps(4, spins, 9, 3)
    density = random.normal(size=(spins, 4, 4))
    weights = random.normal(size=(spins, 4, 4))
    actual = execute(
        programs["weights"], {"spin_density": density, "spin_weights": weights}
    ).outputs
    np.testing.assert_allclose(actual["core"], density.sum(axis=0))
    np.testing.assert_allclose(actual["overlap"], -weights.sum(axis=0))
    shared = {
        name: random.normal(size=(4, 4))
        for name in (
            "core_direction",
            "ecp_local_direction",
            "ecp_nonlocal_direction",
        )
    }
    coupled = {
        name: random.normal(size=(spins, 4, 4))
        for name in ("exact_direction", "fitted_direction", "xc_direction")
    }
    actual = execute(programs["rhs"], shared | coupled).outputs["rhs"]
    np.testing.assert_allclose(actual, sum(shared.values()) + sum(coupled.values()))


def test_nuclear_pair_incidence_and_stationary_sum() -> None:
    random = np.random.default_rng(423)
    programs = assembly_maps(2, 1, 9, 3)
    inputs = {
        name: random.normal(size=shape)
        for name, shape in (
            ("first_charge", (3,)),
            ("second_charge", (3,)),
            ("first_position", (3, 3)),
            ("second_position", (3, 3)),
            ("first_motion", (3, 3)),
            ("second_motion", (3, 3)),
        )
    }
    actual = execute(programs["nuclear_inputs"], inputs).outputs
    np.testing.assert_allclose(
        actual["charge_product"], inputs["first_charge"] * inputs["second_charge"]
    )
    np.testing.assert_allclose(
        actual["displacement"], inputs["first_position"] - inputs["second_position"]
    )
    np.testing.assert_allclose(
        actual["motion"], inputs["first_motion"] - inputs["second_motion"]
    )
    pair = random.normal(size=(3, 3))
    second = execute(programs["nuclear_second"], {"pair_value": pair}).outputs["second"]
    np.testing.assert_array_equal(second, -pair)
    incidences = random.normal(size=(3, 2))
    atom = execute(programs["nuclear_atom"], {"incidences": incidences}).outputs["atom"]
    np.testing.assert_allclose(atom, incidences.sum(axis=1))
    vectors = {
        name: random.normal(size=(9,))
        for name in (
            "core",
            "exact",
            "fitted",
            "ecp_local",
            "ecp_nonlocal",
            "grid",
            "nuclear",
        )
    }
    np.testing.assert_allclose(
        execute(programs["hvp"], vectors).outputs["hvp"], sum(vectors.values())
    )


def test_generator_is_deterministic_and_rejects_invalid_shapes() -> None:
    assert generate() == generate()
    for shape in ((0, 1, 3, 0), (2, 3, 6, 1), (2, 1, 4, 1), (2, 1, 6, -1)):
        with pytest.raises(ValueError):
            assembly_maps(*shape)


def test_generation_needs_no_numerical_runtime(tmp_path: Path) -> None:
    script = """
import runpy, sys
class Deny:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'generativeqc', 'cupy', 'torch', 'pyscf'}:
            raise RuntimeError('forbidden runtime import: ' + fullname)
sys.meta_path.insert(0, Deny())
sys.argv = ['generate_wb97mv_stationary_assembly.py', '--output', sys.argv[1]]
runpy.run_path('tools/generate_wb97mv_stationary_assembly.py', run_name='__main__')
"""
    output = tmp_path / "maps.cuh"
    subprocess.run(
        [sys.executable, "-c", script, str(output)],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert output.read_text() == generate()
