"""Pinned iodine-ECP HF/response failures from #3034, with independent oracles."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from generativeqc import (
    Atom,
    Calculator,
    CBSBasis,
    CCSDTCBSCalculator,
    _native,
    load_basis,
)

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "tests/data/conventional_ecp_cbs"
CASES = json.loads((INPUTS / "cases.json").read_text())
BOHR_ANGSTROM = 0.529177210903
ENERGY_ATOL = 3e-8
FORCE_ATOL = 3e-7


def test_reported_input_integrity() -> None:
    """Protect pinned independent inputs, not implementation/source spelling."""
    for line in (INPUTS / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split()
        assert hashlib.sha256((INPUTS / name).read_bytes()).hexdigest() == expected


def case_atoms(case_name: str) -> tuple:
    """Preserve archived atom order and the original Angstrom-to-Bohr constant."""
    case = CASES[case_name]
    return tuple(
        (symbol, tuple(value / BOHR_ANGSTROM for value in xyz))
        for symbol, xyz in zip(
            case["symbols"], case["coordinates_angstrom"], strict=True
        )
    )


def direction_for(atoms: tuple) -> np.ndarray:
    """Move every center, including iodine, without a pure translation."""
    direction = np.sin(0.7 * np.arange(1, 3 * len(atoms) + 1)).reshape(-1, 3)
    direction -= direction.mean(axis=0)
    return direction / np.linalg.norm(direction)


def displaced(atoms: tuple, direction: np.ndarray, step: float) -> tuple:
    """Return an immutable geometry suitable for independent-oracle caching."""
    return tuple(
        (symbol, tuple(np.asarray(position) + step * shift))
        for (symbol, position), shift in zip(atoms, direction, strict=True)
    )


def normalized_reference_diis(owner: object, base: type) -> object:
    """Use PySCF's affine DIIS with a dimensionless error metric.

    A common Gram scale leaves the affine equations unchanged. Without it,
    PySCF's absolute eigenvalue cutoff discards the small-error history needed
    for nonvariational CCSD(T) finite differences. Restore the stored raw metric
    after extrapolation so subsequent histories retain their physical units.
    This test-only oracle uses no native integrals, orbitals or solver code.
    """

    class NormalizedDIIS(base):
        def extrapolate(self, nd: int | None = None) -> np.ndarray:
            count = self.get_num_vec() if nd is None else nd
            original = self._H.copy()
            metric = self._H[1 : count + 1, 1 : count + 1]
            scale = np.max(np.abs(metric)) if count else 0.0
            if scale > 0:
                metric /= scale
            try:
                return super().extrapolate(count)
            finally:
                self._H[:] = original

    return NormalizedDIIS(owner)


@pytest.mark.parametrize("scale", (1e-12, 1.0, 1e12))
def test_reference_diis_error_units_do_not_change_extrapolation(scale: float) -> None:
    """An independent two-vector affine minimum survives changes of error units."""
    pytest.importorskip("pyscf")
    from pyscf import lib

    extrapolator = normalized_reference_diis(None, lib.diis.DIIS)
    extrapolator.update(np.array([1.0, 3.0]), xerr=scale * np.array([1.0, 0.0]))
    result = extrapolator.update(
        np.array([-2.0, 4.0]), xerr=scale * np.array([0.0, 2.0])
    )
    np.testing.assert_allclose(result, [0.4, 3.2], atol=1e-13, rtol=0)


@cache
def reference_components(
    case_name: str, cardinal: str, atoms: tuple
) -> dict[str, float]:
    """Run libcint/PySCF from the archived basis/ECP, never native orbitals."""
    pytest.importorskip("pyscf")
    from pyscf import cc, gto, lib, mp, scf

    recipe = json.loads((INPUTS / "pyscf_reference_basis.json").read_text())[cardinal]
    molecule = gto.M(
        atom=atoms,
        unit="Bohr",
        basis=recipe["pyscf_basis"],
        ecp=recipe["pyscf_ecp"],
        charge=CASES[case_name]["charge"],
        spin=0,
        cart=False,
        verbose=0,
        max_memory=8000,
    )
    assert molecule.nao == CASES[case_name]["ao_functions"][cardinal]
    reference = scf.RHF(molecule)
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-11
    reference.max_cycle = 200
    reference.direct_scf_tol = 1e-16
    reference.diis = normalized_reference_diis(reference, scf.diis.CDIIS)
    reference.kernel()
    assert reference.converged, "independent physical HF reference did not converge"
    assert (
        np.linalg.norm(reference.get_grad(reference.mo_coeff, reference.mo_occ))
        <= 1e-11
    )
    perturbation = mp.MP2(reference).run(verbose=0)
    result = {"rhf": float(reference.e_tot), "mp2": float(perturbation.e_tot)}
    if cardinal == "dz":
        coupled_cluster = cc.CCSD(reference)
        coupled_cluster.conv_tol = 1e-13
        coupled_cluster.conv_tol_normt = 1e-12
        # Oracle-only bounds and precision never change native stopping controls.
        coupled_cluster.max_cycle = 1200
        coupled_cluster.diis = normalized_reference_diis(coupled_cluster, lib.diis.DIIS)
        integrals = coupled_cluster.ao2mo()
        coupled_cluster.kernel(eris=integrals)
        assert coupled_cluster.converged, "independent CCSD reference did not converge"
        updated_singles, updated_doubles = coupled_cluster.update_amps(
            coupled_cluster.t1, coupled_cluster.t2, integrals
        )
        amplitude_update = coupled_cluster.amplitudes_to_vector(
            updated_singles, updated_doubles
        ) - coupled_cluster.amplitudes_to_vector(coupled_cluster.t1, coupled_cluster.t2)
        assert np.linalg.norm(amplitude_update) <= 1e-12
        result["ccsd(t)"] = float(coupled_cluster.e_tot + coupled_cluster.ccsd_t())
    return result


def reference_cbs(components: dict[str, dict[str, float]]) -> float:
    """Evaluate the DZ/TZ model algebra independently of native plan weights."""
    low, high = components["dz"], components["tz"]
    exponential_ratio = math.exp(1.63)
    hf_limit = (exponential_ratio * high["rhf"] - low["rhf"]) / (exponential_ratio - 1)
    correlation_limit = (
        27 * (high["mp2"] - high["rhf"]) - 8 * (low["mp2"] - low["rhf"])
    ) / 19
    return hf_limit + correlation_limit + low["ccsd(t)"] - low["mp2"]


@pytest.fixture(scope="module")
def reported_bases() -> tuple[CBSBasis, CBSBasis]:
    """Retain both complete orbital bases and the matching 28-electron ECP."""
    return tuple(
        CBSBasis(load_basis(INPUTS / f"native_basis_{cardinal}.json"), number)
        for cardinal, number in (("dz", 2), ("tz", 3))
    )


@pytest.fixture(scope="module")
def cuda_regression() -> None:
    """Long real-device qualification is opt-in and must use finite Slurm work."""
    if os.environ.get("GENERATIVEQC_ECP_CBS_3034_CUDA_TEST") != "1":
        pytest.skip("explicit iodine-ECP HF/CCSD(T)/CBS qualification")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU work must run through srun"


@pytest.fixture(scope="module")
def hf_probe(
    cuda_regression: None, request: pytest.FixtureRequest, tmp_path_factory: object
) -> object:
    """Build the required cached bridge when GPU qualification is requested."""
    native_cxx = request.getfixturevalue("required_native_cxx")
    calculator = Calculator(device="cuda")
    library = Path(calculator._library._name).resolve()
    directory = tmp_path_factory.mktemp("ecp-hf-probe")
    object_file, shared = directory / "probe.o", directory / "probe.so"
    native_cxx.compile_object(
        ROOT / "tests/native/conventional_ecp_hf_probe.cpp",
        object_file,
        args=(
            "-std=c++20",
            "-O2",
            "-fPIC",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
    )
    native_cxx.link([object_file], shared, args=("-shared", str(library)))
    bridge = ctypes.CDLL(str(shared))
    bridge.conventional_ecp_hf_probe.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_double),
    ]
    bridge.conventional_ecp_hf_probe.restype = ctypes.c_int
    return bridge


@pytest.mark.parametrize("case_name", ("hf_dz", "hf_tz"))
def test_reported_physical_hf_step(
    case_name: str, hf_probe: object, reported_bases: tuple[CBSBasis, CBSBasis]
) -> None:
    """Energy plateaus cannot substitute for converged physical references."""
    basis = reported_bases[int(case_name == "hf_tz")].basis
    calculator = Calculator(method="ccsd(t)", basis=basis, device="cuda")
    context = ctypes.c_void_p()
    _native.check(
        calculator._library,
        calculator._library.generativeqc_context_create(
            ctypes.byref(calculator._context_descriptor()), ctypes.byref(context)
        ),
    )
    system = ctypes.c_void_p()
    try:
        atoms = tuple(Atom.from_value(atom) for atom in case_atoms(case_name))
        system = calculator._create_native_system(
            context, atoms, CASES[case_name]["charge"], 1
        )
        diagnostics = (ctypes.c_double * 8)()
        status = hf_probe.conventional_ecp_hf_probe(system, 200, diagnostics)
        assert status == 0, list(diagnostics)
        (
            converged,
            physical,
            iterations,
            energy,
            change,
            density,
            residual,
            commutator,
        ) = diagnostics
        assert converged and physical and 0 < iterations <= 200
        assert density < 1e-11 and residual <= 1e-8 and commutator <= 1e-8
        archived = json.loads((INPUTS / "pyscf_and_native_results.json").read_text())
        cardinal = "tz" if case_name == "hf_tz" else "dz"
        assert math.isfinite(change)
        expected = next(
            item["energy_hartree"]
            for item in archived[case_name]["reference"]
            if item["basis"] == cardinal
        )
        assert energy == pytest.approx(expected, abs=ENERGY_ATOL, rel=0)
    finally:
        if system:
            calculator._library.generativeqc_system_destroy(system)
        calculator._library.generativeqc_context_destroy(context)


def test_reported_physical_ccsdt_force_step(
    cuda_regression: None, reported_bases: tuple[CBSBasis, CBSBasis]
) -> None:
    """Isolate the failed physical Z endpoint and independently qualify its force."""
    case_name = "response_dz"
    atoms = case_atoms(case_name)
    direction = direction_for(atoms)
    workflow = CCSDTCBSCalculator(
        reported_bases,
        auxiliary_bases=None,
        hf_reference="exact",
        device="cuda",
        policy="standard",
        preconvergence="auto",
        calculator_options={"correlation_memory_budget_bytes": 43 * 1024**3},
    )
    calculator = Calculator(
        method="ccsd(t)",
        basis=reported_bases[0].basis,
        device="cuda",
        **workflow._options,
    )
    try:
        result = calculator.singlepoint(
            atoms,
            charge=CASES[case_name]["charge"],
            multiplicity=1,
            properties=("energy", "forces"),
        )
    finally:
        calculator.clear_cache()
    assert result.converged and result.executed_backend == "cuda"
    assert result.forces is not None
    print(
        json.dumps(
            {"native_energy": result.energy, "native_forces": result.forces.tolist()}
        ),
        flush=True,
    )
    expected_energy = reference_components(case_name, "dz", atoms)["ccsd(t)"]
    print(
        f"physical CCSD(T): native={result.energy:.17g}, oracle={expected_energy:.17g}"
    )
    assert result.energy == pytest.approx(expected_energy, abs=ENERGY_ATOL, rel=0)
    np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=3e-8, rtol=0)
    projection = float(np.sum(result.forces * direction))
    for step in (2e-4, 6e-5):
        energies = [
            reference_components(
                case_name, "dz", displaced(atoms, direction, sign * step)
            )["ccsd(t)"]
            for sign in (-1, 1)
        ]
        expected_force = (energies[0] - energies[1]) / (2 * step)
        print(
            f"physical CCSD(T) force: step={step}, native={projection:.17g}, "
            f"oracle={expected_force:.17g}"
        )
        assert abs(expected_force) > 1e-5
        assert projection == pytest.approx(expected_force, abs=FORCE_ATOL, rel=0)


@pytest.mark.parametrize("case_name", tuple(CASES))
def test_reported_complete_ecp_cbs_force(
    case_name: str, cuda_regression: None, reported_bases: tuple[CBSBasis, CBSBasis]
) -> None:
    """Qualify every component and nonzero composite force against two-step FD."""
    atoms = case_atoms(case_name)
    direction = direction_for(atoms)
    workflow = CCSDTCBSCalculator(
        reported_bases,
        auxiliary_bases=None,
        hf_reference="exact",
        device="cuda",
        policy="standard",
        preconvergence="auto",
        calculator_options={"correlation_memory_budget_bytes": 43 * 1024**3},
    )
    result = workflow.singlepoint(
        atoms,
        charge=CASES[case_name]["charge"],
        multiplicity=1,
        properties=("energy", "forces"),
    )
    assert result.converged and result.executed_backend == "cuda"
    assert result.forces is not None
    np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=3e-8, rtol=0)
    print(
        json.dumps(
            {
                "case": case_name,
                "native_energy": result.energy,
                "native_forces": result.forces.tolist(),
                "components": [
                    {
                        "method": component.method,
                        "basis": component.basis_identity,
                        "energy": component.result.energy,
                        "forces": (
                            None
                            if component.result.forces is None
                            else component.result.forces.tolist()
                        ),
                    }
                    for component in result.components
                ],
            }
        ),
        flush=True,
    )
    oracles = {
        cardinal: reference_components(case_name, cardinal, atoms)
        for cardinal in ("dz", "tz")
    }
    assert result.energy == pytest.approx(
        reference_cbs(oracles), abs=3 * ENERGY_ATOL, rel=0
    )
    native_components = {}
    for component in result.components:
        cardinal = (
            "dz"
            if component.basis_identity == reported_bases[0].basis.identity
            else "tz"
        )
        native_components[cardinal, component.method] = component.result
        assert (
            component.result.converged and component.result.executed_backend == "cuda"
        )
        assert component.result.energy == pytest.approx(
            oracles[cardinal][component.method], abs=ENERGY_ATOL, rel=0
        )
        assert component.result.forces is not None
        np.testing.assert_allclose(
            component.result.forces.sum(axis=0), 0, atol=3e-8, rtol=0
        )
    assert set(native_components) == {
        ("dz", "rhf"),
        ("dz", "mp2"),
        ("dz", "ccsd(t)"),
        ("tz", "rhf"),
        ("tz", "mp2"),
    }
    for step in (2e-4, 6e-5):
        shifted = [
            {
                cardinal: reference_components(
                    case_name, cardinal, displaced(atoms, direction, sign * step)
                )
                for cardinal in ("dz", "tz")
            }
            for sign in (-1, 1)
        ]
        composite_force = (reference_cbs(shifted[0]) - reference_cbs(shifted[1])) / (
            2 * step
        )
        assert abs(composite_force) > 1e-5, (
            "direction failed to exercise a nonzero molecular force"
        )
        assert float(np.sum(result.forces * direction)) == pytest.approx(
            composite_force, abs=FORCE_ATOL, rel=0
        )
        for (cardinal, method), component in native_components.items():
            expected = (shifted[0][cardinal][method] - shifted[1][cardinal][method]) / (
                2 * step
            )
            assert float(np.sum(component.forces * direction)) == pytest.approx(
                expected, abs=FORCE_ATOL, rel=0
            )
