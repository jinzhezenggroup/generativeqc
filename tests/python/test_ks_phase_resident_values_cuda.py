"""Slurm-only Hybrid source reuse, independent J/K/energy and endpoint gates."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires a Slurm-allocated CUDA device",
)


@pytest.mark.parametrize("case", ["water", "methane", "water-spd"])
@pytest.mark.parametrize("chunk_width", ["1", "2"])
def test_global_hybrid_phase_source(
    tmp_path: Path, native_cxx: NativeCxx, case: str, chunk_width: str
) -> None:
    """Independent reference data is never supplied to the production KS solve."""
    numpy = pytest.importorskip("numpy")
    pyscf = pytest.importorskip("pyscf")
    pyscf_dft = pytest.importorskip("pyscf.dft")
    pyscf.lib.num_threads(2)
    atoms = (
        [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 1.4, 1.1)), ("H", (0.0, -1.4, 1.1))]
        if case == "water"
        else [
            ("C", (0.0, 0.0, 0.0)),
            ("H", (1.2, 1.2, 1.2)),
            ("H", (1.2, -1.2, -1.2)),
            ("H", (-1.2, 1.2, -1.2)),
            ("H", (-1.2, -1.2, 1.2)),
        ]
    )
    if case == "water-spd":
        from benchmarks.readme_hf_scaling import scaling_cases

        atoms = scaling_cases()["water-24"].atoms
    basis = {
        symbol: pyscf.gto.basis.load(
            "def2-svp" if case == "water-spd" else "aug-cc-pvtz", symbol
        )
        for symbol in {symbol for symbol, _ in atoms}
    }
    molecule = pyscf.gto.M(atom=atoms, basis=basis, unit="Bohr", cart=False, verbose=0)
    dimension = molecule.nao_nr()
    indices = numpy.arange(dimension)
    density = 0.02 * numpy.sin(indices[:, None] + indices[None, :] + 1)
    density += 0.03 * numpy.eye(dimension)
    coulomb, exchange = pyscf.scf.hf.get_jk(molecule, density, hermi=1)
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
    lines = [f"{len(atoms)} {len(shells)} {dimension}"]
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
    for matrix in (density, coulomb, exchange):
        lines.extend(" ".join(f"{value:.17g}" for value in row) for row in matrix)
    input_path = tmp_path / f"{case}.input"
    input_path.write_text("\n".join(lines) + "\n")

    if configured := os.environ.get("GENERATIVEQC_KS_PHASE_VALUES_PROBE"):
        executable = Path(configured)
    else:
        from generativeqc import _native

        nvcc = shutil.which("nvcc")
        if not nvcc:
            pytest.skip("CUDA toolchain unavailable")
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
    execution = subprocess.run(
        [str(executable), str(input_path)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env={**os.environ, "GENERATIVEQC_CUDA_KS_CHUNK": chunk_width},
    )
    Path(f"{input_path}.probe.log").write_text(execution.stdout + execution.stderr)
    assert execution.returncode == 0, execution.stdout + execution.stderr
    oracle_results = []
    for functional in ("pbe0", "b3lyp"):
        state = json.loads(Path(f"{input_path}.{functional}.json").read_text())
        solver = pyscf_dft.RKS(molecule)
        solver.xc = functional
        solver.grids.coords = numpy.asarray(state["points"]).reshape(-1, 3)
        solver.grids.weights = numpy.asarray(state["weights"])
        returned_density = numpy.asarray(state["density"]).reshape(dimension, dimension)
        oracle = solver.energy_tot(dm=returned_density)
        error = abs(state["energy"] - oracle)
        oracle_results.append(
            {
                "functional": functional,
                "native": state["energy"],
                "oracle": oracle,
                "error": error,
            }
        )
        Path(f"{input_path}.oracle.json").write_text(
            json.dumps(oracle_results, indent=2) + "\n"
        )
        assert error < 2e-8
    if (
        os.environ.get("GENERATIVEQC_KS_PHASE_VALUES_ENDPOINT_TEST") == "1"
        and chunk_width == "1"
    ):
        with_forces = os.environ.get("GENERATIVEQC_KS_PHASE_VALUES_FORCE_TEST") == "1"
        endpoints = []
        for selection in ("0", "1"):
            execution = subprocess.run(
                [
                    str(executable),
                    str(input_path),
                    "endpoint-forces" if with_forces else "endpoint",
                ],
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
                env={
                    **os.environ,
                    "GENERATIVEQC_KS_RESIDENT_VALUES": selection,
                    "GENERATIVEQC_CUDA_KS_CHUNK": chunk_width,
                },
            )
            Path(f"{input_path}.endpoint-{selection}.log").write_text(
                execution.stdout + execution.stderr
            )
            assert execution.returncode == 0, execution.stdout + execution.stderr
            endpoints.append(
                [json.loads(line) for line in execution.stdout.splitlines()]
            )
        Path(f"{input_path}.endpoints.json").write_text(
            json.dumps(endpoints, indent=2) + "\n"
        )
        for baseline, resident in zip(*endpoints, strict=True):
            assert (
                baseline["properties"]
                == resident["properties"]
                == ("E+F" if with_forces else "E")
            )
            assert baseline["source_values"] == 0
            assert resident["source_values"] > 0
            assert baseline["fock_builds"] > 0
            assert resident["fock_builds"] > 0
            assert baseline["warm_energy_rebase_focks"] == 0
            assert resident["warm_energy_rebase_focks"] == 1
            assert resident["fock_builds"] == resident["iterations"] + 1
            assert abs(baseline["energy"] - resident["energy"]) < 2e-9
            if with_forces:
                assert (
                    numpy.max(
                        numpy.abs(
                            numpy.asarray(baseline["forces"]) - resident["forces"]
                        )
                    )
                    < 1e-7
                )
