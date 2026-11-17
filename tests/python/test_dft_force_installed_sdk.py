"""Installed SDK/CLI/Python force parity with an independent molecular oracle.

Set GENERATIVEQC_NATIVE_SDK_PREFIX and load its installed library through
GENERATIVEQC_LIBRARY. CUDA cases additionally require the normal finite Slurm
allocation and GENERATIVEQC_DFT_CUDA_TEST=1. This covers the curated electronic
inventory, not unqualified custom graphs, resource promises or molecular HVPs.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, _native
from test_r1_native_forces import H2, METHODS, forbid_python_force, require_backend
from test_treutler_endpoints import reference

if TYPE_CHECKING:
    from conftest import NativeCxx


@pytest.fixture(scope="module")
def installed_force_probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> tuple[Path, Path, Path, Path]:
    """Compile against installed public headers and link that exact installed DSO."""
    configured = os.environ.get("GENERATIVEQC_NATIVE_SDK_PREFIX")
    if not configured:
        pytest.skip("requires an explicitly installed native SDK")
    prefix = Path(configured).resolve()
    library = Path(_native.load_library()._name).resolve()
    assert library.is_relative_to(prefix), "Python must load the installed SDK library"
    assert (prefix / "include/generativeqc/ks.hpp").is_file()
    directory = tmp_path_factory.mktemp("installed-force-sdk")
    source = Path(__file__).resolve().parents[1] / "native/dft_force_sdk_probe.cpp"
    executable = required_native_cxx.build_executable(
        [source],
        directory / "force-sdk",
        compile_args=("-std=c++20", "-O2", "-I", str(prefix / "include")),
        link_args=(str(library), f"-Wl,-rpath,{library.parent}"),
    )
    empty_path = directory / "empty-path"
    empty_path.mkdir()
    cli = prefix / "bin/generativeqc"
    assert cli.is_file(), "the installed SDK must include its native CLI"
    molecule = directory / "h2.xyz"
    molecule.write_text(
        "2\nNon-axis-aligned H2, coordinates in Bohr\n"
        + "".join(f"{symbol} {' '.join(map(str, point))}\n" for symbol, point in H2)
    )
    return executable, cli, molecule, empty_path


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("fitted", (False, True))
@pytest.mark.parametrize("spin", (0, 1))
@pytest.mark.parametrize(
    "method,functional,nonlocal_level",
    [(method, functional, 0) for method, functional in METHODS]
    + [("wb97m-v", "WB97M_V", 1)],
)
def test_installed_products_complete_forces_match_independent_oracle(
    installed_force_probe: tuple[Path, Path, Path, Path],
    device: str,
    representation: str,
    fitted: bool,
    spin: int,
    method: str,
    functional: str,
    nonlocal_level: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No product may replace the native force with a Python diagnostic.

    The open-shell H2 cation also exercises an empty beta population. Its
    non-axis-aligned bond prevents a one-component-only force comparison.
    WB97M-V uses the explicitly mixed DF/exact-LR Hamiltonian in every product.
    """
    require_backend(device)
    if device == "cpu" and method in {"m06-2x", "mn15"}:
        pytest.skip("split-hybrid SCF programs have CUDA providers only")
    if device == "cuda" and method == "pbe50":
        pytest.skip("PBE50 SCF has a CPU provider only")
    pytest.importorskip("pyscf")
    if fitted and method == "wb97m-v":
        monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "direct")
    forbid_python_force(monkeypatch)
    executable, cli, molecule, empty_path = installed_force_probe
    fitting = device if fitted else "none"
    completed = subprocess.run(
        [
            str(executable),
            device,
            method,
            representation,
            str(spin),
            fitting,
            str(nonlocal_level),
        ],
        env={**os.environ, "PATH": str(empty_path)},
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    sdk = json.loads(completed.stdout)
    assert sdk["supported_properties"] & _native.PROPERTY_FORCES
    completed_cli = subprocess.run(
        [
            str(cli),
            "run",
            str(molecule),
            "--method",
            f"{'lda_xc_pw' if method == 'lda' else method}-{'uks' if spin else 'rks'}",
            "--basis",
            "sto-3g",
            "--representation",
            representation,
            "--density-fitting",
            fitting,
            "--backend",
            device,
            "--charge",
            str(spin),
            "--multiplicity",
            str(spin + 1),
            "--units",
            "bohr",
            "--grid-level",
            "0",
            *(
                ["--nonlocal-grid-level", str(nonlocal_level)]
                if method == "wb97m-v"
                else []
            ),
            "--max-iterations",
            "200",
            "--energy-tolerance",
            "1e-12",
            "--density-tolerance",
            "1e-10",
            "--screening-tolerance",
            "1e-13",
            "--forces",
            "--json",
        ],
        env={**os.environ, "PATH": str(empty_path)},
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    cli_result = json.loads(completed_cli.stdout)
    calculator = Calculator(
        method=f"{method}-{'uks' if spin else 'rks'}",
        basis="sto-3g",
        basis_representation=representation,
        device=device,
        density_fitting=fitting,
        ks_options=KsOptions(
            grid=GridSpec.treutler(0),
            nonlocal_grid=(
                GridSpec.treutler(nonlocal_level) if method == "wb97m-v" else None
            ),
        ),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-13,
    )
    with calculator.prepare_batch(
        [H2], charges=[spin], multiplicities=[spin + 1]
    ) as batch:
        assert batch._native_dft_force_eligible(0)
        cold = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        warm = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        energy_only = batch.execute(strict=True, properties=("energy",)).items[0]
    from tools._pyscf_vv10_domain import vv10_density_domain

    with vv10_density_domain(1e-10):
        energy, forces = reference(
            calculator,
            fitted=fitted,
            functional=functional,
            spin=spin,
            grid_level=0,
            nonlocal_level=nonlocal_level,
            atom_values=H2,
        )
    backend = (
        _native.BACKEND_CUDA if device == "cuda" else _native.BACKEND_CPU_REFERENCE
    )
    for phase, actual in (("cold", cold), ("warm", warm)):
        record = sdk[phase]
        assert record["backend"] == backend
        assert record["energy"] == pytest.approx(energy, abs=2e-8)
        assert actual.energy == pytest.approx(energy, abs=2e-8)
        sdk_forces = np.asarray(record["forces"]).reshape(2, 3)
        np.testing.assert_allclose(sdk_forces, forces, atol=3e-7, rtol=0)
        np.testing.assert_allclose(actual.forces, forces, atol=3e-7, rtol=0)
        np.testing.assert_allclose(sdk_forces, actual.forces, atol=3e-8, rtol=0)
        np.testing.assert_allclose(sdk_forces.sum(axis=0), 0, atol=3e-10, rtol=0)
    assert sdk["energy_only"]["forces"] is None and energy_only.forces is None
    assert sdk["energy_only"]["energy"] == pytest.approx(energy, abs=2e-8)
    assert energy_only.energy == pytest.approx(energy, abs=2e-8)
    assert cli_result["energy_hartree"] == pytest.approx(energy, abs=2e-8)
    np.testing.assert_allclose(
        cli_result["forces_hartree_per_bohr"], forces, atol=3e-7, rtol=0
    )


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_AUTOMATIC_CUDA_FORCE_TEST") != "1",
    reason="requires the explicitly selected automatic CUDA force package",
)
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("spin", (0, 1))
@pytest.mark.parametrize(
    "name",
    (
        "LDA_C_VWN_4",
        "LDA_C_CHACHIYO",
        "LDA_C_PZ_MOD",
        "GGA_X_APBE",
        "GGA_X_PBE_SOL",
        "GGA_X_RPBE",
        "GGA_X_B86",
        "GGA_C_APBE",
        "MGGA_X_LTA",
        "MGGA_X_GVT4",
        "MGGA_C_TPSS",
    ),
)
def test_installed_automatic_cuda_products_preserve_complete_forces(
    representation: str,
    spin: int,
    name: str,
    installed_force_probe: tuple[Path, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport qualified automatic graphs, not every energy registration.

    The nonempty-spin H4 fixture retains the existing physical domain. The
    products run without Python/toolchains on PATH; a separate PySCF SCF and
    analytic moving-grid gradient provide the scientific acceptance gate.
    """
    from test_dft_complete_cpu import (
        NativeAO,
        StationaryKsState,
        independent_global_hybrid_gradient,
    )

    require_backend("cuda")
    pytest.importorskip("pyscf")
    forbid_python_force(monkeypatch)
    atoms = [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.1, 0.2, 1.4)),
        ("H", (1.6, 0.4, 0.3)),
        ("H", (1.8, 0.3, 1.8)),
    ]
    executable, cli, _, empty_path = installed_force_probe
    completed = subprocess.run(
        [
            str(executable),
            "cuda",
            f"libxc:{name.lower()}",
            representation,
            str(spin),
            "none",
            "0",
        ],
        env={**os.environ, "PATH": str(empty_path)},
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    sdk = json.loads(completed.stdout)
    molecule = tmp_path / "automatic-h4.xyz"
    molecule.write_text(
        "4\nBohr\n"
        + "".join(
            f"{symbol} {point[0]} {point[1]} {point[2]}\n" for symbol, point in atoms
        )
    )
    selector = f"libxc-{'uks' if spin else 'rks'}:{name}"
    completed_cli = subprocess.run(
        [
            str(cli),
            "run",
            str(molecule),
            "--method",
            selector,
            "--basis",
            "sto-3g",
            "--representation",
            representation,
            "--density-fitting",
            "none",
            "--backend",
            "cuda",
            "--charge",
            str(spin),
            "--multiplicity",
            str(spin + 1),
            "--units",
            "bohr",
            "--grid-radial-points",
            "24",
            "--grid-polar-points",
            "8",
            "--grid-azimuth-points",
            "16",
            "--initial-guess",
            "none",
            "--max-iterations",
            "300",
            "--energy-tolerance",
            "1e-13",
            "--density-tolerance",
            "1e-12",
            "--screening-tolerance",
            "1e-12",
            "--forces",
            "--json",
        ],
        env={**os.environ, "PATH": str(empty_path)},
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    cli_result = json.loads(completed_cli.stdout)
    calculator = Calculator(
        method=selector,
        basis="sto-3g",
        basis_representation=representation,
        device="cuda",
        initial_guess=None,
        precision="fp64",
        max_iterations=300,
        energy_tolerance=1e-13,
        density_tolerance=1e-12,
        screening_tolerance=1e-12,
        ks_options=KsOptions(grid=GridSpec(24, 8, 16), tile_points=37),
    )
    with (
        calculator.prepare_batch(
            [atoms], charges=[spin], multiplicities=[spin + 1]
        ) as batch,
        NativeAO(
            atoms, representation=representation, charge=spin, multiplicity=spin + 1
        ) as basis,
    ):
        assert batch._native_dft_force_eligible(0)
        cold = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        warm = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        state = StationaryKsState.from_native(batch, basis)
        try:
            energy, gradient = independent_global_hybrid_gradient(
                basis,
                state,
                "full-range-uks" if spin else "full-range-rks",
                xc=name,
                initial_guess="hcore",
            )
        finally:
            state._source.close()
        energy_only = batch.execute(strict=True, properties=("energy",)).items[0]
    assert sdk["supported_properties"] & _native.PROPERTY_FORCES
    for phase, actual in (("cold", cold), ("warm", warm)):
        record = sdk[phase]
        assert (
            actual.executed_backend == "cuda"
            and record["backend"] == _native.BACKEND_CUDA
        )
        assert actual.energy == pytest.approx(energy, abs=2e-8)
        assert record["energy"] == pytest.approx(energy, abs=2e-8)
        np.testing.assert_allclose(actual.forces, -gradient, atol=3e-7, rtol=0)
        np.testing.assert_allclose(
            np.asarray(record["forces"]).reshape(4, 3), -gradient, atol=3e-7, rtol=0
        )
    assert energy_only.forces is None and sdk["energy_only"]["forces"] is None
    assert energy_only.energy == pytest.approx(energy, abs=2e-8)
    assert sdk["energy_only"]["energy"] == pytest.approx(energy, abs=2e-8)
    assert cli_result["energy_hartree"] == pytest.approx(energy, abs=2e-8)
    np.testing.assert_allclose(
        cli_result["forces_hartree_per_bohr"], -gradient, atol=3e-7, rtol=0
    )
    np.testing.assert_allclose(
        cli_result["forces_hartree_per_bohr"], cold.forces, atol=3e-8, rtol=0
    )
