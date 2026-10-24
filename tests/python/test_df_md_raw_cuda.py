"""Independent raw-integral, resource and replay gates for prepared Hermite E."""

from __future__ import annotations

import copy
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
    os.environ.get("GENERATIVEQC_TEST_DF_MD_RAW") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires explicit raw Hermite qualification in a finite Slurm GPU job",
)


@pytest.fixture(scope="module")
def raw_probe(tmp_path_factory: Any, required_native_cxx: NativeCxx) -> Path:
    """Build the production owner with a verified compiler cache."""
    folder = tmp_path_factory.mktemp("df-md-raw")
    (folder / "generated_df_md_j.cuh").write_text(emit_df_md_j_header())
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).absolute()
    compiler = replace(
        required_native_cxx,
        compiler=str(Path(os.environ["CUDA_PATH"]) / "bin/nvcc"),
    )
    return compiler.build_executable(
        [ROOT / "tests/native/df_md_raw_probe.cu", ROOT / "src/scf/cuda/df_md_raw.cu"],
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


def evaluate(
    probe: Path,
    folder: Path,
    angular: tuple[int, int, int],
    representations: tuple[str, str],
    *,
    variant: str = "asymmetric",
    scale: float = 1.0,
    translation: float = 0.0,
    lengths: tuple[int, int, int] = (2, 3, 2),
) -> float:
    """Use independently normalized libcint, including mixed public bases."""
    from pyscf import lib

    from tools.generativeqc_validation.df_gradient import reference_df_matrices
    from tools.generativeqc_validation.df_values import make_df_value_fixture

    lib.num_threads(1)
    fixture = make_df_value_fixture(
        angular, variant=variant, primitive_lengths=lengths, exponent_scale=scale
    )
    bases = []
    rows = []
    for index, shells in enumerate((slice(0, 2), slice(2, 3))):
        basis = copy.deepcopy(fixture.inputs)
        basis["shells"] = basis["shells"][shells]
        basis["basis_representation"] = representations[index]
        for position in basis["coordinates"]:
            position[:] = [value + translation for value in position]
        bases.append(basis)
        rows.append(
            f"3 {len(basis['shells'])} 1 {int(representations[index] == 'spherical')}"
        )
        rows.extend("1 " + " ".join(map(str, r)) for r in basis["coordinates"])
        for shell in basis["shells"]:
            rows.append(
                f"{shell['atom_index']} {shell['angular_momentum']} {len(shell['primitives'])}"
            )
            rows.extend(" ".join(map(str, p)) for p in shell["primitives"])
    raw = reference_df_matrices(*bases)[0]
    expected = raw[np.tril_indices(raw.shape[0])].ravel()
    text = "\n".join(rows) + "\n"
    (folder / "input.txt").write_text(text)
    output = folder / "raw.bin"
    run = subprocess.run(
        [str(probe), str(output)],
        input=text,
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )
    (folder / "stdout").write_text(run.stdout)
    (folder / "stderr").write_text(run.stderr)
    assert run.returncode == 0, run.stderr
    actual = np.fromfile(output, dtype=np.float64).reshape(4, -1)
    for result in actual:
        np.testing.assert_allclose(result, expected, atol=2e-10, rtol=2e-11)
    # Full and ragged panels use the same deterministic primitive order.
    for result in actual[1:]:
        np.testing.assert_array_equal(result, actual[0])
    return float(np.max(abs(actual - expected)))


@pytest.mark.parametrize("angular", tuple(product(range(4), range(4), range(5))))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
def test_raw_all_angular_classes(
    raw_probe: Path,
    tmp_path: Path,
    angular: tuple[int, int, int],
    representation: str,
    record_property: Any,
) -> None:
    """Signed primitives, self pairs, full/tail panels and dirty graph replays."""
    record_property(
        "max_abs_error", evaluate(raw_probe, tmp_path, angular, (representation,) * 2)
    )


@pytest.mark.parametrize(
    "representations", (("cartesian", "spherical"), ("spherical", "cartesian"))
)
@pytest.mark.parametrize(
    "variant,scale,translation,lengths",
    (
        ("coincident", 1.0, 0.0, (2, 3, 2)),
        ("asymmetric", 0.01, 0.0, (2, 3, 2)),
        ("asymmetric", 100.0, 10.0, (2, 3, 2)),
        ("asymmetric", 1.0, 0.003, (8, 7, 6)),
    ),
)
def test_mixed_representations_and_geometry(
    raw_probe: Path,
    tmp_path: Path,
    representations: tuple[str, str],
    variant: str,
    scale: float,
    translation: float,
    lengths: tuple[int, int, int],
) -> None:
    """Fresh physical owners cover tight/diffuse and long signed contractions."""
    evaluate(
        raw_probe,
        tmp_path,
        (3, 2, 4),
        representations,
        variant=variant,
        scale=scale,
        translation=translation,
        lengths=lengths,
    )
