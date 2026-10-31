"""Independent device gates for the proposed tensor-free DF J contractions.

This private seam qualifies public-AO charge/projection and replay reset only.
It also exercises bounded owner setup and changed-density graph replay.
It is not a production SCF, metric-policy, force, or endpoint performance gate.
Every real-device execution requires an explicitly enabled finite Slurm/LSF job.
"""

from __future__ import annotations

import math
import os
import subprocess
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc_compiler.integral.df_md_j_cuda import emit_df_md_j_header

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TEST_DF_MD_J") != "1"
    or not (os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID")),
    reason="requires explicit DF MD qualification in a finite Slurm/LSF GPU allocation",
)


@pytest.fixture(scope="module")
def df_md_probe(tmp_path_factory: Any, required_native_cxx: NativeCxx) -> Path:
    """Compile the device seam with the verified compiler cache, without JIT."""
    folder = tmp_path_factory.mktemp("df-md-device")
    (folder / "generated_df_md_j.cuh").write_text(emit_df_md_j_header())
    # Keep the caller's library directory: a diagnostic SONAME link may live
    # beside this symlink rather than beside its immutable target.
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).absolute()
    cuda = Path(os.environ["CUDA_PATH"])
    compiler = replace(required_native_cxx, compiler=str(cuda / "bin/nvcc"))
    return compiler.build_executable(
        [
            ROOT / "tests/native/df_md_j_probe.cu",
            ROOT / "src/scf/cuda/df_md_owner.cu",
        ],
        folder / "probe",
        compile_args=(
            "-std=c++20",
            "-O2",
            "-arch=" + os.environ.get("GENERATIVEQC_TEST_CUDA_ARCH", "sm_120"),
            "-I" + str(folder),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ),
        link_args=(str(library), "-Xlinker=-rpath," + str(library.parent)),
        compile_timeout=600,
        link_timeout=60,
    )


def _evaluate(
    probe: Path,
    folder: Path,
    angular: tuple[int, int, int],
    spherical: bool,
    variant: str = "asymmetric",
    exponent_scale: float = 1.0,
    translation: float = 0.0,
    primitive_lengths: tuple[int, int, int] = (2, 3, 2),
    separation_scale: float = 1.0,
    density_scale: float = 1.0,
) -> dict[str, Any]:
    from pyscf import lib

    from tools.generate_validation_references import pyscf_molecule
    from tools.generativeqc_validation.df_values import make_df_value_fixture

    lib.num_threads(1)
    fixture = make_df_value_fixture(
        angular,
        variant=variant,
        primitive_lengths=primitive_lengths,
        exponent_scale=exponent_scale,
        separation_scale=separation_scale,
    )
    for r in fixture.inputs["coordinates"]:
        for axis in range(3):
            r[axis] += translation
    mol, scales, _ = pyscf_molecule(fixture.inputs)
    loc = mol.ao_loc_nr(cart=not spherical)
    n, a = int(loc[2]), int(loc[3] - loc[2])
    raw = np.empty((n, n, a))
    for i, j in product(range(2), repeat=2):
        block = mol.intor_by_shell(
            "int3c2e_sph" if spherical else "int3c2e_cart", (i, j, 2)
        )
        if not spherical:
            block = (
                block
                * scales[loc[i] : loc[i + 1], None, None]
                * scales[None, loc[j] : loc[j + 1], None]
                * scales[None, None, loc[2] : loc[3]]
            )
        raw[loc[i] : loc[i + 1], loc[j] : loc[j + 1], :] = block
    rng = np.random.default_rng(2917)
    density = density_scale * rng.normal(
        size=(n, n)
    )  # Deliberately nonsymmetric and indefinite.
    potential = rng.normal(size=a)
    lines = []
    for shells in (fixture.inputs["shells"][:2], fixture.inputs["shells"][2:]):
        lines.append(f"3 {len(shells)} 1 {int(spherical)}")
        lines.extend(
            "1 " + " ".join(map(str, r)) for r in fixture.inputs["coordinates"]
        )
        for shell in shells:
            lines.append(
                f"{shell['atom_index']} {shell['angular_momentum']} "
                f"{len(shell['primitives'])}"
            )
            lines.extend(" ".join(map(str, p)) for p in shell["primitives"])
    lines.append(" ".join(map(str, density.ravel())))
    lines.append(" ".join(map(str, potential)))
    text = "\n".join(lines) + "\n"
    (folder / "basis-and-vectors.txt").write_text(text)
    output = folder / "products.bin"
    derivatives = os.environ.get("GENERATIVEQC_TEST_DF_MD_DERIVATIVE") == "1"
    run = subprocess.run(
        [str(probe), str(output)] + (["--gradient"] if derivatives else []),
        input=text,
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )
    (folder / "probe.stdout").write_text(run.stdout)
    (folder / "probe.stderr").write_text(run.stderr)
    assert run.returncode == 0, run.stderr
    all_values = np.fromfile(output, dtype=np.float64).reshape(
        3, a + n * n + (3 * mol.natm if derivatives else 0)
    )
    result = all_values[:, : a + n * n]
    assert np.isfinite(result).all()
    charge = np.einsum("ijp,ij->p", raw, density)
    jmatrix = np.einsum("ijp,p->ij", raw, potential)
    expected = np.concatenate((charge, jmatrix.ravel()))
    np.testing.assert_allclose(result[0], expected, atol=2e-10, rtol=2e-11)
    np.testing.assert_array_equal(result[1], 0.0)
    refreshed = np.concatenate((-0.37 * charge, 2.0 * jmatrix.ravel()))
    np.testing.assert_allclose(result[2], refreshed, atol=2e-10, rtol=2e-11)
    np.testing.assert_allclose(
        result[0, :a] @ potential,
        np.sum(result[0, a:].reshape(n, n) * density),
        atol=3e-10,
        rtol=3e-11,
    )
    derivative_error = None
    if derivatives:
        coordinates = mol.atom_coords()

        def energy(atom: int, axis: int, displacement: float) -> float:
            """Independent raw three-center scalar; no native density/metric oracle."""
            geometry = coordinates.copy()
            geometry[atom, axis] += displacement
            shifted = mol.copy()
            shifted.set_geom_(geometry, unit="Bohr")
            value = 0.0
            for i, j in product(range(2), repeat=2):
                block = shifted.intor_by_shell(
                    "int3c2e_sph" if spherical else "int3c2e_cart", (i, j, 2)
                )
                if not spherical:
                    block *= (
                        scales[loc[i] : loc[i + 1], None, None]
                        * scales[None, loc[j] : loc[j + 1], None]
                        * scales[None, None, loc[2] : loc[3]]
                    )
                value += np.einsum(
                    "ijp,ij,p->",
                    block,
                    density[loc[i] : loc[i + 1], loc[j] : loc[j + 1]],
                    potential,
                )
            return float(value)

        step = 2e-4 / math.sqrt(exponent_scale)
        derivative = np.empty((mol.natm, 3))
        for atom, axis in product(range(mol.natm), range(3)):
            derivative[atom, axis] = (
                energy(atom, axis, -2 * step)
                - 8 * energy(atom, axis, -step)
                + 8 * energy(atom, axis, step)
                - energy(atom, axis, 2 * step)
            ) / (12 * step)
        gradients = all_values[:, a + n * n :].reshape(3, mol.natm, 3)
        for observed, scale in zip(gradients, (1.25, 0.0, -0.925), strict=True):
            np.testing.assert_allclose(
                observed - 0.25, scale * derivative, atol=2e-8, rtol=2e-8
            )
            np.testing.assert_allclose((observed - 0.25).sum(axis=0), 0.0, atol=2e-9)
        np.testing.assert_array_equal(gradients[1], 0.25)
        derivative_error = float(np.max(abs(gradients[0] - 0.25 - 1.25 * derivative)))
    return {
        "derivative_max_abs_error": derivative_error,
        "max_abs_error": float(np.max(abs(result[0] - expected))),
        "replay_max_abs_error": float(np.max(abs(result[2] - refreshed))),
        "native_inventory": run.stdout.strip(),
    }


@pytest.mark.parametrize("splits", (1, 3, 8, 64))
@pytest.mark.parametrize("angular", ((0, 0, 0), (2, 1, 4)))
def test_partitioned_charge_with_multiple_source_groups(
    df_md_probe: Path,
    tmp_path: Path,
    splits: int,
    angular: tuple[int, int, int],
    monkeypatch: Any,
    record_property: Any,
) -> None:
    """Long contractions exercise nonzero splits, tails and capacity fallback.

    The same independent oracle checks normal, zero and changed-density graph
    replay. The existing derivative opt-in also checks the raised charge pass.
    """
    monkeypatch.setenv("GENERATIVEQC_DF_MD_CHARGE_SPLITS", str(splits))
    record_property(
        "qualification",
        _evaluate(df_md_probe, tmp_path, angular, True, primitive_lengths=(8, 9, 5)),
    )


@pytest.mark.parametrize("angular", ((0, 0, 0), (0, 0, 4)))
def test_automatic_long_charge_with_independent_oracle(
    df_md_probe: Path,
    tmp_path: Path,
    angular: tuple[int, int, int],
    monkeypatch: Any,
    record_property: Any,
) -> None:
    """Qualify default partitioning on long signed primitive contractions.

    Same-shell primitive products exceed the automatic work boundary. The
    independent libcint oracle covers charge, projection, changed/zero graph
    replay and, with the existing opt-in, displaced three-center derivatives.
    """
    monkeypatch.delenv("GENERATIVEQC_DF_MD_CHARGE_SPLITS", raising=False)
    record_property(
        "qualification",
        _evaluate(df_md_probe, tmp_path, angular, True, primitive_lengths=(96, 97, 5)),
    )


@pytest.mark.parametrize("angular", tuple(product(range(4), range(4), range(5))))
@pytest.mark.parametrize("spherical", (False, True), ids=("cartesian", "spherical"))
def test_charge_and_projection_all_orbital_f_auxiliary_g(
    df_md_probe: Path,
    tmp_path: Path,
    angular: tuple[int, int, int],
    spherical: bool,
    record_property: Any,
) -> None:
    """Signed contractions, both shell orientations, self pairs and ragged lanes."""
    record_property(
        "qualification", _evaluate(df_md_probe, tmp_path, angular, spherical)
    )


@pytest.mark.parametrize("angular", ((0, 0, 0), (2, 1, 3), (3, 3, 4)))
@pytest.mark.parametrize("spherical", (False, True), ids=("cartesian", "spherical"))
@pytest.mark.parametrize(
    ("variant", "exponent_scale", "translation"),
    (("coincident", 1.0, 0.0), ("asymmetric", 0.01, 0.0), ("asymmetric", 100.0, 10.0)),
)
def test_geometry_and_exponent_ranges(
    df_md_probe: Path,
    tmp_path: Path,
    angular: tuple[int, int, int],
    spherical: bool,
    variant: str,
    exponent_scale: float,
    translation: float,
    record_property: Any,
) -> None:
    record_property(
        "qualification",
        _evaluate(
            df_md_probe,
            tmp_path,
            angular,
            spherical,
            variant,
            exponent_scale,
            translation,
        ),
    )


@pytest.mark.parametrize("angular", ((0, 0, 0), (2, 1, 3), (3, 3, 4)))
@pytest.mark.parametrize("metric_scale", (1.0, 1e-12, 0.0))
@pytest.mark.parametrize("density_scale", (1.0, 1e100))
def test_metric_bounded_sources_preserve_independent_products_and_replay(
    df_md_probe: Path,
    tmp_path: Path,
    monkeypatch: Any,
    angular: tuple[int, int, int],
    metric_scale: float,
    density_scale: float,
) -> None:
    """Current signed inputs, diffuse tails, graph resets and singular fail-open.

    The probe also sends NaN/Inf after graph replay and validates resource
    refusal/cleanup. A zero eigenvalue must disable charge omission; it does
    not affect projection's independent current-source bound.
    """
    if density_scale == 1.0:
        monkeypatch.setenv("GENERATIVEQC_TEST_DF_MD_DERIVATIVE", "1")
    monkeypatch.setenv("GENERATIVEQC_TEST_DF_MD_SCREEN_BUDGET", "1e-15")
    monkeypatch.setenv("GENERATIVEQC_TEST_DF_MD_METRIC_SCALE", str(metric_scale))
    result = _evaluate(
        df_md_probe,
        tmp_path,
        angular,
        True,
        separation_scale=10.0,
        density_scale=density_scale,
    )
    fields = dict(field.split("=", 1) for field in result["native_inventory"].split())
    assert float(fields["max_j_omission"]) == 1e-15
    assert float(fields["projection_error_per_primitive"]) > 0
    assert (float(fields["source_norm_limit"]) > 0) == (metric_scale > 0)


@pytest.mark.parametrize("spherical", (False, True), ids=("cartesian", "spherical"))
@pytest.mark.parametrize("angular", ((0, 1), (2, 3), (3, 4), (4, 4)))
def test_factored_metric_gradient_replay(
    df_md_probe: Path, tmp_path: Path, spherical: bool, angular: tuple[int, int]
) -> None:
    """Signed metric factors match libcint finite differences through g/g.

    Two different auxiliary centers make the force nonzero. The native seam
    covers fresh/zero/changed captured factors, addition to an existing sink,
    exact-capacity allocation refusal and no allocations during replay.
    """
    import copy

    from pyscf import lib

    from tools.generate_validation_references import pyscf_molecule
    from tools.generativeqc_validation.df_values import make_df_value_fixture

    lib.num_threads(1)
    inputs = copy.deepcopy(
        make_df_value_fixture((0, 0, angular[0]), primitive_lengths=(2, 3, 2)).inputs
    )
    inputs["coordinates"].append([-0.24, -0.54, 0.73])
    inputs["atomic_numbers"].append(1)
    inputs["charge"] = 0
    inputs["multiplicity"] = 1
    extra = copy.deepcopy(inputs["shells"][2])
    extra["atom_index"] = 3
    extra["angular_momentum"] = angular[1]
    extra["primitives"] = [
        [1.3 * exponent, coefficient] for exponent, coefficient in extra["primitives"]
    ]
    inputs["shells"].append(extra)
    mol, scales, _ = pyscf_molecule(inputs)
    loc = mol.ao_loc_nr(cart=not spherical)
    n, a = int(loc[2]), int(loc[4] - loc[2])
    rng = np.random.default_rng(9041)
    density, potential = rng.normal(size=(n, n)), rng.normal(size=a)
    lines = []
    for shells in (inputs["shells"][:2], inputs["shells"][2:]):
        lines.append(f"4 {len(shells)} 0 {int(spherical)}")
        lines.extend("1 " + " ".join(map(str, r)) for r in inputs["coordinates"])
        for shell in shells:
            lines.append(
                f"{shell['atom_index']} {shell['angular_momentum']} {len(shell['primitives'])}"
            )
            lines.extend(" ".join(map(str, p)) for p in shell["primitives"])
    lines.extend((" ".join(map(str, density.ravel())), " ".join(map(str, potential))))
    output = tmp_path / "metric.bin"
    run = subprocess.run(
        [str(df_md_probe), str(output), "--metric-gradient"],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    observed = (
        np.fromfile(output, dtype=np.float64)
        .reshape(3, a + n * n + 12)[:, -12:]
        .reshape(3, 4, 3)
    )

    def energy(atom: int, axis: int, displacement: float) -> float:
        geometry = mol.atom_coords()
        geometry[atom, axis] += displacement
        shifted = mol.copy()
        shifted.set_geom_(geometry, unit="Bohr")
        metric = np.empty((a, a))
        for i, j in product((2, 3), repeat=2):
            block = shifted.intor_by_shell(
                "int2c2e_sph" if spherical else "int2c2e_cart", (i, j)
            )
            if not spherical:
                block = (
                    block
                    * scales[loc[i] : loc[i + 1], None]
                    * scales[None, loc[j] : loc[j + 1]]
                )
            metric[
                loc[i] - loc[2] : loc[i + 1] - loc[2],
                loc[j] - loc[2] : loc[j + 1] - loc[2],
            ] = block
        return 0.5 * float(potential @ metric @ potential)

    step = 2e-4
    gradient = np.empty((4, 3))
    for atom, axis in product(range(4), range(3)):
        gradient[atom, axis] = (
            energy(atom, axis, -2 * step)
            - 8 * energy(atom, axis, -step)
            + 8 * energy(atom, axis, step)
            - energy(atom, axis, 2 * step)
        ) / (12 * step)
    assert np.max(abs(gradient)) > 1e-4
    for value, scale in zip(observed, (1.25, 0.0, 5.0), strict=True):
        np.testing.assert_allclose(value - 0.25, scale * gradient, atol=3e-8, rtol=3e-8)
        np.testing.assert_allclose((value - 0.25).sum(axis=0), 0.0, atol=3e-9)
    np.testing.assert_array_equal(observed[1], 0.25)
