"""Independent energy and geometry/lifetime checks of automatic and forced RHF values."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import NativeCxx
from generativeqc import _native

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RCCSD_CUDA_TEST") != "1",
    reason="requires a Slurm-allocated CUDA device",
)


@pytest.mark.parametrize("case", ["water", "methane", "methane-spd"])
def test_phase_values_preserve_independent_rhf_and_retire_geometry_leases(
    tmp_path: Path, native_cxx: NativeCxx, case: str
) -> None:
    """The oracle supplies assertions only, never a native density/Fock/ERI input."""
    pyscf = pytest.importorskip("pyscf")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        pytest.skip("CUDA headers unavailable")
    pyscf.lib.num_threads(2)
    if case == "water":
        atoms = [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 1.4, 1.1)),
            ("H", (0.0, -1.4, 1.1)),
        ]
    else:
        atoms = [
            ("C", (0.0, 0.0, 0.0)),
            ("H", (1.2, 1.2, 1.2)),
            ("H", (1.2, -1.2, -1.2)),
            ("H", (-1.2, 1.2, -1.2)),
            ("H", (-1.2, -1.2, 1.2)),
        ]
    basis = {
        symbol: pyscf.gto.basis.load("aug-cc-pvtz", symbol)
        for symbol in {symbol for symbol, _ in atoms}
    }
    if case == "methane-spd":
        basis = {
            symbol: [shell for shell in shells if shell[0] <= 2]
            for symbol, shells in basis.items()
        }

    def oracle(geometry: list[tuple[str, tuple[float, float, float]]]) -> float:
        molecule = pyscf.gto.M(
            atom=geometry, basis=basis, unit="Bohr", cart=False, verbose=0
        )
        solver = pyscf.scf.RHF(molecule)
        solver.conv_tol = 1e-13
        solver.conv_tol_grad = 1e-10
        energy = solver.kernel()
        assert solver.converged
        return energy

    original = oracle(atoms)
    moved = list(atoms)
    moved[1] = (moved[1][0], (*moved[1][1][:2], moved[1][1][2] + 0.01))
    displaced = oracle(moved)
    shells = []
    for atom_index, (symbol, _) in enumerate(atoms):
        for angular, *primitives in basis[symbol]:
            for contraction in range(1, len(primitives[0])):
                shells.append(
                    (
                        atom_index,
                        angular,
                        [(row[0], row[contraction]) for row in primitives],
                    )
                )
    lines = [f"{len(atoms)} {len(shells)} {original:.17g} {displaced:.17g}"]
    lines.extend(
        f"{pyscf.gto.charge(symbol)} {' '.join(map(str, coordinates))}"
        for symbol, coordinates in atoms
    )
    for atom_index, angular, primitives in shells:
        lines.append(f"{atom_index} {angular} {len(primitives)}")
        lines.extend(
            f"{exponent:.17g} {coefficient:.17g}"
            for exponent, coefficient in primitives
        )
    input_path = tmp_path / f"{case}-aug-cc-pvtz.input"
    input_path.write_text("\n".join(lines) + "\n")
    library = Path(_native.load_library(device="cuda")._name).resolve()
    cuda = Path(nvcc).resolve().parents[1]
    executable = tmp_path / "probe"
    native_cxx.build_executable(
        [ROOT / "tests/native/rhf_resident_values_probe.cpp"],
        executable,
        compile_args=(
            "-std=c++20",
            "-O2",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
            f"-I{cuda / 'include'}",
        ),
        link_args=(str(library), f"-Wl,-rpath,{library.parent}"),
    )
    journal = tmp_path / "progress.jsonl"
    result = subprocess.run(
        [str(executable), str(input_path)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env={**os.environ, "GENERATIVEQC_DF_PROGRESS_TRACE": str(journal)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["passed"]
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    leases = [
        record
        for record in records
        if record.get("key") == "phase_resident_value_bytes"
    ]
    assert len(leases) == 10
    assert all(record["value"] > 0 for record in leases[:4])
    assert all(record["value"] == 0 for record in leases[4:6])
    assert (leases[6]["value"] > 0) == (case == "methane")
    assert all(record["value"] == 0 for record in leases[7:])
    selection = [
        record["value"]
        for record in records
        if record.get("key") == "phase_resident_values_selection"
    ]
    assert selection[-4:] == ["auto"] * 4
