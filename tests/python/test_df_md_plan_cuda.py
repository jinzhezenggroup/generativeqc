"""Independent metric/J gates for the optional prepared Hermite value owner.

Private algebra coverage lives in test_df_md_j_cuda. These tests exercise the
actual source, metric and J handoff, budget fallback, and graph replay together.
With GENERATIVEQC_TEST_DF_MD_FORCE=1 they also check the complete Coulomb
gradient against displaced libcint energy derivatives. They do not qualify
SCF or default performance admission.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TEST_DF_MD_PLAN") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires explicit DF MD plan qualification in a finite Slurm job",
)


@pytest.fixture(scope="module")
def df_md_plan_probe(tmp_path_factory: Any, required_native_cxx: NativeCxx) -> Path:
    """Link the actual compiled core; do not rebuild a substitute J implementation."""
    folder = tmp_path_factory.mktemp("df-md-plan")
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).absolute()
    cuda = Path(os.environ["CUDA_PATH"])
    return required_native_cxx.build_executable(
        [ROOT / "tests/native/df_md_plan_probe.cpp"],
        folder / "probe",
        compile_args=(
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-I" + str(cuda / "include"),
        ),
        link_args=(
            str(library),
            "-Wl,-rpath," + str(library.parent),
            "-L" + str(cuda / "lib64"),
            "-lcudart",
        ),
    )


@pytest.mark.parametrize(
    ("angular", "spherical", "duplicate_auxiliary"),
    [
        (angular, spherical, False)
        for angular, spherical in product(
            (
                (0, 0, 0),
                (0, 1, 2),
                (1, 2, 3),
                (2, 1, 4),
                (2, 2, 4),
                (3, 1, 3),
                (3, 3, 4),
            ),
            (False, True),
        )
    ]
    + [((1, 2, 2), spherical, True) for spherical in (False, True)],
)
def test_metric_applied_j_and_changed_input_graph(
    df_md_plan_probe: Path,
    tmp_path: Path,
    angular: tuple[int, int, int],
    spherical: bool,
    duplicate_auxiliary: bool,
    record_property: Any,
) -> None:
    """Libcint raw integrals and NumPy's independent metric solve gate every replay."""
    from pyscf import lib

    from tools.generate_validation_references import pyscf_molecule
    from tools.generativeqc_validation.df_values import make_df_value_fixture

    lib.num_threads(1)
    fixture = make_df_value_fixture(angular, primitive_lengths=(2, 3, 2))
    if duplicate_auxiliary:
        fixture.inputs["shells"].append(copy.deepcopy(fixture.inputs["shells"][-1]))
    mol, scales, _ = pyscf_molecule(fixture.inputs)
    loc = mol.ao_loc_nr(cart=not spherical)
    n, a = int(loc[2]), int(loc[-1] - loc[2])
    suffix = "sph" if spherical else "cart"

    def independent_integrals(molecule: Any) -> tuple[np.ndarray, np.ndarray]:
        """Libcint values at the requested geometry, independently normalized."""
        raw = np.empty((n, n, a))
        for i, j, k in product(range(2), range(2), range(2, molecule.nbas)):
            block = molecule.intor_by_shell("int3c2e_" + suffix, (i, j, k))
            if not spherical:
                block *= (
                    scales[loc[i] : loc[i + 1], None, None]
                    * scales[None, loc[j] : loc[j + 1], None]
                    * scales[None, None, loc[k] : loc[k + 1]]
                )
            raw[
                loc[i] : loc[i + 1],
                loc[j] : loc[j + 1],
                loc[k] - loc[2] : loc[k + 1] - loc[2],
            ] = block
        metric = np.empty((a, a))
        for i, j in product(range(2, molecule.nbas), repeat=2):
            block = molecule.intor_by_shell("int2c2e_" + suffix, (i, j))
            if not spherical:
                block *= (
                    scales[loc[i] : loc[i + 1], None]
                    * scales[None, loc[j] : loc[j + 1]]
                )
            metric[
                loc[i] - loc[2] : loc[i + 1] - loc[2],
                loc[j] - loc[2] : loc[j + 1] - loc[2],
            ] = block
        return raw, metric

    raw, metric = independent_integrals(mol)
    density = np.random.default_rng(8173).normal(size=(n, n))
    charge = np.einsum("ijp,ij->p", raw, density)
    expected = np.einsum(
        "ijp,p->ij", raw, np.linalg.pinv(metric, rcond=1e-12, hermitian=True) @ charge
    )
    lines = []
    for shells in (fixture.inputs["shells"][:2], fixture.inputs["shells"][2:]):
        lines.append(f"3 {len(shells)} 1 {int(spherical)}")
        lines.extend(
            "1 " + " ".join(map(str, r)) for r in fixture.inputs["coordinates"]
        )
        for shell in shells:
            lines.append(
                f"{shell['atom_index']} {shell['angular_momentum']} {len(shell['primitives'])}"
            )
            lines.extend(" ".join(map(str, p)) for p in shell["primitives"])
    lines.append(" ".join(map(str, density.ravel())))
    text = "\n".join(lines) + "\n"
    (tmp_path / "basis-and-density.txt").write_text(text)
    output = tmp_path / "metric-and-j.bin"
    gradients_requested = os.environ.get("GENERATIVEQC_TEST_DF_MD_FORCE") == "1"
    run = subprocess.run(
        [str(df_md_plan_probe), str(output)]
        + (["--gradient"] if gradients_requested else []),
        input=text,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    (tmp_path / "probe.stdout").write_text(run.stdout)
    (tmp_path / "probe.stderr").write_text(run.stderr)
    assert run.returncode == 0, run.stderr
    result = np.fromfile(output, dtype=np.float64)
    np.testing.assert_allclose(
        result[5 * n * n : 5 * n * n + a * a].reshape(a, a),
        metric,
        atol=2e-10,
        rtol=2e-11,
    )
    values = result[: 5 * n * n].reshape(5, n, n)
    assert np.isfinite(values).all()
    for sample, scale in zip(values, (1, 1, 1, 0, -0.37), strict=True):
        np.testing.assert_allclose(sample, scale * expected, atol=3e-10, rtol=3e-11)
    np.testing.assert_array_equal(values[3], 0)
    record_property("max_abs_error", float(np.max(abs(values[:3] - expected))))
    inventory = json.loads(run.stdout)
    assert inventory["admitted"] is not duplicate_auxiliary
    record_property("native_inventory", inventory)

    if gradients_requested:
        observed = result[5 * n * n + a * a :].reshape(4, mol.natm, 3)
        symmetric_density = (density + density.T) * 0.5
        coordinates = mol.atom_coords()

        def energy(atom: int, axis: int, displacement: float) -> float:
            """Differentiate the independent complete DF Coulomb energy."""
            geometry = coordinates.copy()
            geometry[atom, axis] += displacement
            shifted = mol.copy()
            shifted.set_geom_(geometry, unit="Bohr")
            integrals, metric_at_geometry = independent_integrals(shifted)
            rho = np.einsum("ijp,ij->p", integrals, symmetric_density)
            inverse = np.linalg.pinv(metric_at_geometry, rcond=1e-12, hermitian=True)
            return float(0.5 * rho @ inverse @ rho)

        step = 2e-4
        derivative = np.empty((mol.natm, 3))
        for atom, axis in product(range(mol.natm), range(3)):
            derivative[atom, axis] = (
                energy(atom, axis, -2 * step)
                - 8 * energy(atom, axis, -step)
                + 8 * energy(atom, axis, step)
                - energy(atom, axis, 2 * step)
            ) / (12 * step)
        for sample, scale in zip(observed, (1.0, 1.0, 0.0, 0.37**2), strict=True):
            np.testing.assert_allclose(sample, scale * derivative, atol=2e-8, rtol=2e-8)
            np.testing.assert_allclose(sample.sum(axis=0), 0.0, atol=2e-9)
        np.testing.assert_array_equal(observed[2], 0.0)
        record_property(
            "gradient_max_abs_error", float(np.max(abs(observed[:2] - derivative)))
        )
