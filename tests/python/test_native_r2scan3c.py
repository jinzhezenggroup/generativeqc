"""Bound-basis construction and complete native r2SCAN-3c ownership gates."""

from __future__ import annotations

import ctypes as ct
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, _native

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
ATOMS = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
GRID = GridSpec(8, 4, 8)


def test_compiled_bound_basis_preserves_all_source_shells(
    required_native_cxx: NativeCxx, tmp_path: Path
) -> None:
    """Compare all H-Ar data to the pinned input, not a generated fingerprint."""
    library = Path(_native.load_library()._name).resolve()
    executable = tmp_path / "named-basis"
    required_native_cxx.build_executable(
        [ROOT / "tests/native/test_named_ks_catalog.cpp"],
        executable,
        compile_args=(
            "-std=c++20",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
        link_args=(str(library), f"-Wl,-rpath,{library.parent}"),
    )
    output = subprocess.run(
        [str(executable), "dump"], check=True, capture_output=True, text=True
    ).stdout
    actual = []
    for line in output.splitlines():
        z, angular, *values = line.split()
        actual.append((int(z), int(angular), tuple(map(float, values))))
    source = json.loads(
        (ROOT / "python/generativeqc/data/r2scan3c/def2-mtzvpp-h-ar.json").read_text()
    )
    expected = []
    for element in source["elements"]:
        for shell in element["shells"]:
            for column in shell["coefficients"]:
                values = tuple(
                    value
                    for exponent, coefficient in zip(
                        shell["exponents"], column, strict=True
                    )
                    for value in (float(exponent), float(coefficient))
                )
                expected.append(
                    (element["atomic_number"], shell["angular_momentum"], values)
                )
    assert actual == expected
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("fitting", [False, True])
def test_native_owner_skips_python_composition_and_revokes_receipts(
    monkeypatch: pytest.MonkeyPatch, fitting: bool
) -> None:
    """Both direct and DF energies own corrections; electronic force gates persist."""
    from generativeqc import dispersion
    from generativeqc._named_ks import CompositeCorrectionResult

    options = (
        {"density_fitting": "cpu", "auxiliary_basis": "def2-svp"} if fitting else {}
    )
    calculator = Calculator(
        method="r2scan-3c-rks",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=200,
        **options,
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("native named endpoint delegated scientific composition to Python")

    monkeypatch.setattr(dispersion, "D4CorrectionBatch", forbidden)
    monkeypatch.setattr(dispersion, "R2SCAN3CCorrectionBatch", forbidden)
    monkeypatch.setattr(dispersion, "_evaluate_gcp_result", forbidden)
    properties = ("energy",)
    with calculator.prepare_batch([ATOMS, ATOMS], warm_start=False) as batch:
        first = batch.execute(strict=True, properties=properties).items[0]
        assert first.dispersion.gcp.backend == "cpu"
        assert batch.dispersion_diagnostic.d4.profile == "r2scan3c"
        assert first.dispersion.energy == pytest.approx(
            first.dispersion.d4.energy + first.dispersion.gcp.energy, abs=1e-15
        )
        energy = batch.execute(strict=True, properties=("energy",)).items[0]
        assert (
            energy.dispersion.gradient is None
            and energy.dispersion.gcp.gradient is None
        )
        assert energy.energy == pytest.approx(first.energy, abs=1e-11)
        failed = batch.execute([np.full((2, 3), np.nan), None], properties=("energy",))
        assert not failed.items[0].succeeded and failed.items[0].dispersion is None
        assert failed.items[1].succeeded
        receipt = CompositeCorrectionResult(
            ct.sizeof(CompositeCorrectionResult), _native.ABI_VERSION
        )
        query = batch._library.generativeqc_ks_batch_composite_correction_result_v1
        assert (
            query(batch._batch, 0, ct.byref(receipt)) == _native.STATUS_NOT_IMPLEMENTED
        )
        restored = batch.execute(strict=True, properties=properties).items[0]
        assert restored.energy == pytest.approx(first.energy, abs=1e-11)
    bounded = Calculator(
        method="r2scan-3c-rks",
        ks_options=KsOptions(grid=GRID),
        dispersion_memory_budget_bytes=1,
    )
    with pytest.raises(RuntimeError):
        bounded.singlepoint(ATOMS, properties=("energy",))


@pytest.mark.parametrize("spin", [0, 1])
def test_total_matches_independent_pyscf_d4_and_gcp(spin: int) -> None:
    """Independent SCF and correction engines with exactly the same quadrature."""
    pytest.importorskip("pyscf")
    pytest.importorskip("dftd4")
    from dftd4.interface import DampingParam, DispersionModel
    from pyscf import dft, gto

    from tools.generativeqc_gcp.reference import evaluate_r2scan3c_gcp

    snapshot = json.loads(
        (ROOT / "python/generativeqc/data/r2scan3c/def2-mtzvpp-h-ar.json").read_text()
    )
    hydrogen = next(item for item in snapshot["elements"] if item["atomic_number"] == 1)
    basis = [
        [
            shell["angular_momentum"],
            *[
                [
                    float(exponent),
                    *[float(column[i]) for column in shell["coefficients"]],
                ]
                for i, exponent in enumerate(shell["exponents"])
            ],
        ]
        for shell in hydrogen["shells"]
    ]
    molecule = gto.M(
        atom=ATOMS, basis={"H": basis}, unit="Bohr", spin=spin, charge=spin, verbose=0
    )
    engine = (dft.UKS if spin else dft.RKS)(molecule)
    engine.xc = "R2SCAN"
    nodes, weights = np.polynomial.legendre.leggauss(GRID.radial_points)
    t = (nodes + 1) / 2
    radii, measures = t / (1 - t), weights / 2 * t**2 / (1 - t) ** 4
    heights, polar_weights = np.polynomial.legendre.leggauss(GRID.angular_polar)
    polar, azimuth = np.meshgrid(
        heights,
        np.arange(GRID.angular_azimuth) * (2 * np.pi / GRID.angular_azimuth),
        indexing="ij",
    )
    directions = np.stack(
        (
            np.sqrt(1 - polar**2) * np.cos(azimuth),
            np.sqrt(1 - polar**2) * np.sin(azimuth),
            polar,
        ),
        axis=-1,
    ).reshape(-1, 3)
    angular = np.repeat(polar_weights, GRID.angular_azimuth) * (
        2 * np.pi / GRID.angular_azimuth
    )
    table = {
        "H": (
            (radii[:, None, None] * directions[None, :, :]).reshape(-1, 3),
            (measures[:, None] * angular[None, :]).reshape(-1),
        )
    }
    engine.grids.gen_atomic_grids = lambda *args, **kwargs: table
    engine.grids.radii_adjust = None
    engine.grids.becke_scheme = dft.gen_grid.original_becke
    engine.grids.prune = None
    engine.small_rho_cutoff = 0
    engine.conv_tol, engine.conv_tol_grad = 1e-12, 1e-10
    electronic = engine.kernel()
    assert engine.converged
    xyz = np.array([position for _, position in ATOMS])
    d4_result = DispersionModel(
        np.array([1, 1]), xyz, charge=float(spin), ga=2.0, gc=1.0
    ).get_dispersion(DampingParam(s6=1.0, s8=0.0, s9=2.0, a1=0.42, a2=5.65), grad=False)
    gcp = evaluate_r2scan3c_gcp((1, 1), xyz)
    result = Calculator(
        method="r2scan-3c-uks" if spin else "r2scan-3c-rks",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    ).singlepoint(ATOMS, charge=spin, multiplicity=spin + 1, properties=("energy",))
    print(
        f"independent r2SCAN-3c spin={spin}: {electronic + float(d4_result['energy']) + gcp.energy:.17g}"
    )
    assert result.energy == pytest.approx(
        electronic + float(d4_result["energy"]) + gcp.energy, abs=2e-8
    )
    assert result.dispersion.d4.energy == pytest.approx(
        float(d4_result["energy"]), abs=2e-10
    )
    assert result.dispersion.gcp.energy == pytest.approx(gcp.energy, abs=2e-14)
