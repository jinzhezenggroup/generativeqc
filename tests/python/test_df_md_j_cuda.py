"""Independent device gates for the proposed tensor-free DF J contractions.

This private seam qualifies public-AO charge/projection and replay reset only.
It is not a production SCF, metric-policy, force, or endpoint performance gate.
Every real-device execution requires an explicitly enabled finite Slurm job.
"""

from __future__ import annotations

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
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires explicit DF MD qualification in a finite Slurm GPU allocation",
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
        [ROOT / "tests/native/df_md_j_probe.cu"],
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
) -> dict[str, Any]:
    from pyscf import lib

    from tools.generate_validation_references import pyscf_molecule
    from tools.generativeqc_validation.df_values import make_df_value_fixture

    lib.num_threads(1)
    fixture = make_df_value_fixture(
        angular,
        variant=variant,
        primitive_lengths=(2, 3, 2),
        exponent_scale=exponent_scale,
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
    density = rng.normal(size=(n, n))  # Deliberately nonsymmetric and indefinite.
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
    run = subprocess.run(
        [str(probe), str(output)],
        input=text,
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )
    (folder / "probe.stdout").write_text(run.stdout)
    (folder / "probe.stderr").write_text(run.stderr)
    assert run.returncode == 0, run.stderr
    result = np.fromfile(output, dtype=np.float64).reshape(3, a + n * n)
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
    return {
        "max_abs_error": float(np.max(abs(result[0] - expected))),
        "replay_max_abs_error": float(np.max(abs(result[2] - refreshed))),
        "native_inventory": run.stdout.strip(),
    }


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
