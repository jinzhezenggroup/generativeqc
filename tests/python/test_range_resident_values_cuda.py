"""Independent radial-source matrices and identity gates on a scheduled GPU."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("displacement", [0.0, 0.001])
@pytest.mark.parametrize(
    "mode,copies",
    [("range-oracle", 1), ("range-mixed-oracle", 1), ("range-mixed-oracle", 4)],
)
def test_resident_range_sources_match_independent_matrices(
    tmp_path: Path, native_cxx: NativeCxx, displacement: float, mode: str, copies: int
) -> None:
    """A new geometry owns new sources; signed spin matrices test raw K weights."""
    if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires a Slurm-allocated GPU")
    assert os.environ.get("SLURM_JOB_ID")
    np = pytest.importorskip("numpy")
    pyscf = pytest.importorskip("pyscf")
    from generativeqc import _native

    atoms = [
        ("O", (0.02, -0.03, 0.04)),
        ("H", (0.1, 1.43, 1.1 + displacement)),
        ("H", (-0.15, -1.45, 1.12)),
    ]
    atoms = [
        (symbol, (xyz[0] + 6.0 * copy, xyz[1], xyz[2]))
        for copy in range(copies)
        for symbol, xyz in atoms
    ]
    basis = {symbol: pyscf.gto.basis.load("def2-svp", symbol) for symbol in ("H", "O")}
    mol = pyscf.gto.M(atom=atoms, basis=basis, unit="Bohr", cart=False, verbose=0)
    n = mol.nao_nr()
    indices = np.arange(n)
    density = 0.02 * np.sin(indices[:, None] + indices[None, :] + 1) + 0.03 * np.eye(n)
    coulomb, exchange = pyscf.scf.hf.get_jk(mol, density, hermi=1)
    radial = []
    for omega in (-0.3, 0.3):
        with mol.with_range_coulomb(omega):
            radial.append(pyscf.scf.hf.get_jk(mol, density, hermi=1, with_j=False)[1])
    shells = []
    for atom_index, (symbol, _) in enumerate(atoms):
        for angular, *primitives in basis[symbol]:
            for contraction in range(1, len(primitives[0])):
                shells.append(
                    (atom_index, angular, [(p[0], p[contraction]) for p in primitives])
                )
    lines = [f"{len(atoms)} {len(shells)} {n}"]
    lines.extend(f"{pyscf.gto.charge(s)} {' '.join(map(str, xyz))}" for s, xyz in atoms)
    for atom_index, angular, primitives in shells:
        lines.append(f"{atom_index} {angular} {len(primitives)}")
        lines.extend(f"{e:.17g} {c:.17g}" for e, c in primitives)
    for matrix in (density, coulomb, exchange, *radial):
        lines.extend(" ".join(f"{v:.17g}" for v in row) for row in matrix)
    fixture = tmp_path / "range.input"
    fixture.write_text("\n".join(lines) + "\n")
    if configured := os.environ.get("GENERATIVEQC_KS_PHASE_VALUES_PROBE"):
        executable = Path(configured)
    else:
        nvcc = shutil.which("nvcc")
        assert nvcc, "CUDA compiler environment required"
        cuda = Path(nvcc).resolve().parents[1]
        library = Path(_native.load_library(device="cuda")._name).resolve()
        executable = tmp_path / "probe"
        native_cxx.build_executable(
            [ROOT / "tests/native/ks_phase_values_probe.cpp"],
            executable,
            compile_args=(
                "-std=c++20",
                "-O2",
                f"-I{ROOT / 'src'}",
                f"-I{ROOT / 'include'}",
                f"-I{cuda / 'include'}",
            ),
            link_args=(
                str(library),
                f"-Wl,-rpath,{library.parent}",
                f"-L{cuda / 'lib64'}",
                "-lcudart",
            ),
        )
    result = subprocess.run(
        [str(executable), str(fixture), mode],
        capture_output=True,
        text=True,
        timeout=360,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "mixed range oracle checks 24"
        if mode == "range-mixed-oracle"
        else "range oracle checks 16"
    ) in result.stdout
    if mode == "range-mixed-oracle":
        assert "mixed resident range oracle checks 8" in result.stdout
    (tmp_path / "oracle-result.txt").write_text(result.stdout)
