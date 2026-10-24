"""Public Python and Python-free CLI endpoints share prepared KS execution."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc.batch import PreparedBatch


@pytest.fixture
def cli() -> Path:
    """Use the matching build's CLI, or an explicitly supplied installed binary."""
    binary = Path(
        os.environ.get(
            "GENERATIVEQC_CLI",
            Path(__file__).resolve().parents[2] / "build/generativeqc",
        )
    ).resolve()
    if not binary.is_file():
        pytest.skip("requires a native CLI build or GENERATIVEQC_CLI")
    return binary


@pytest.mark.parametrize(
    "method,fitting",
    [
        ("pbe-rks", "none"),
        ("pbe-uks", "none"),
        ("pbe0-rks", "none"),
        ("pbe0-uks", "none"),
        ("b3lyp-rks", "none"),
        ("pbe-rks", "cpu"),
        ("pbe0-uks", "cpu"),
    ],
)
def test_python_cli_energy_force_and_moved_geometry(
    cli: Path,
    method: str,
    fitting: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise complete endpoints without admitting a Python derivative fallback."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("qualified endpoint invoked Python force orchestration")

    monkeypatch.setattr(PreparedBatch, "_public_dft_cpu_force", forbidden)
    charge, multiplicity = (1, 2) if method.endswith("uks") else (0, 1)
    registered = method in ("pbe-rks", "pbe-uks")
    grid = GridSpec.treutler() if registered else GridSpec()
    calculator = Calculator(
        method=method,
        device="cpu",
        basis="sto-3g",
        density_fitting=fitting,
        ks_options=KsOptions(grid=grid),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
    )
    # Registered PBE carriers use the native default; the existing CLI only
    # exposes explicit grids for generated compositions. Preserve legacy-grid
    # coverage on those compositions alongside the new default tests below.
    grid_flags = (
        []
        if registered
        else [
            "--grid-radial-points",
            str(grid.radial_points),
            "--grid-polar-points",
            str(grid.angular_polar),
            "--grid-azimuth-points",
            str(grid.angular_azimuth),
        ]
    )
    # The native child gets neither an interpreter nor a compiler on PATH.
    # Python remains the test driver and only invokes the already built CLI.
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    environment = {**os.environ, "PATH": str(empty_path), "PYTHONPATH": str(empty_path)}
    xyz = tmp_path / "h2.xyz"
    energies = []
    for displacement in (0.0, 0.08):
        z = 0.7 + displacement
        atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, z))]
        xyz.write_text(f"2\nBohr\nH 0 0 -0.7\nH 0 0 {z}\n")
        for forces in (False, True):
            python = calculator.singlepoint(
                atoms,
                charge=charge,
                multiplicity=multiplicity,
                properties=("energy", "forces") if forces else ("energy",),
            )
            run = subprocess.run(
                [
                    str(cli),
                    "run",
                    str(xyz),
                    "--method",
                    method,
                    "--basis",
                    "sto-3g",
                    "--backend",
                    "cpu",
                    "--units",
                    "bohr",
                    "--density-fitting",
                    fitting,
                    "--charge",
                    str(charge),
                    "--multiplicity",
                    str(multiplicity),
                    "--max-iterations",
                    "200",
                    "--energy-tolerance",
                    "1e-12",
                    "--density-tolerance",
                    "1e-10",
                    *grid_flags,
                    *(["--forces"] if forces else []),
                    "--json",
                ],
                env=environment,
                text=True,
                capture_output=True,
                timeout=90,
                check=False,
            )
            assert run.returncode == 0, run.stderr
            native = json.loads(run.stdout)
            assert native["backend"] == python.executed_backend == "cpu_reference"
            assert native["energy_hartree"] == pytest.approx(python.energy, abs=2e-10)
            # Roundoff can change the DIIS stopping iteration of independent
            # solves; both must report actual converged work and the same E/F.
            assert native["iterations"] > 0 and python.iterations > 0
            if forces:
                np.testing.assert_allclose(
                    native["forces_hartree_per_bohr"], python.forces, atol=2e-9, rtol=0
                )
            else:
                assert "forces_hartree_per_bohr" not in native
                assert python.forces is None
                energies.append(python.energy)
    assert abs(energies[1] - energies[0]) > 1e-6


@pytest.mark.parametrize(
    "method,level",
    [("pbe-rks", None), ("pbe0-rks", None), ("pbe0-rks", 0), ("pbe0-rks", 3)],
)
def test_treutler_cli_default_and_level(
    cli: Path, tmp_path: Path, method: str, level: int | None
) -> None:
    """Omitted SDK options and explicit CLI levels match the Python descriptor."""
    atoms = [("O", (0, 0, 0)), ("H", (0, 1.4, 1.1)), ("H", (0, -1.4, 1.1))]
    options = None if level is None else KsOptions(grid=GridSpec.treutler(level))
    expected = Calculator(
        method=method,
        basis="sto-3g",
        device="cpu",
        ks_options=options,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    ).singlepoint(atoms, properties=("energy", "forces"))
    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\nBohr\nO 0 0 0\nH 0 1.4 1.1\nH 0 -1.4 1.1\n")
    arguments = [
        str(cli),
        "run",
        str(xyz),
        "--method",
        method,
        "--basis",
        "sto-3g",
        "--backend",
        "cpu",
        "--units",
        "bohr",
        "--forces",
        "--json",
        "--energy-tolerance",
        "1e-12",
        "--density-tolerance",
        "1e-10",
    ]
    if level is not None:
        arguments += ["--grid-level", str(level)]
    result = subprocess.run(
        arguments, capture_output=True, text=True, timeout=90, check=False
    )
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual["energy_hartree"] == pytest.approx(expected.energy, abs=2e-10)
    np.testing.assert_allclose(
        actual["forces_hartree_per_bohr"], expected.forces, atol=2e-9, rtol=0
    )


@pytest.mark.parametrize(
    "flags",
    [
        ["--grid-level", "10"],
        ["--grid-level", "3", "--grid-radial-points", "24"],
        ["--grid-polar-points", "8", "--grid-level", "3"],
    ],
)
def test_cli_rejects_ambiguous_grid_families(
    cli: Path, tmp_path: Path, flags: list[str]
) -> None:
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\nBohr\nH 0 0 0\nH 0 0 1.4\n")
    result = subprocess.run(
        [str(cli), "run", str(xyz), "--method", "pbe-rks", *flags],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode != 0
    assert "grid" in result.stderr
