"""Full RI-JK/correlation E+F gates, independent of the production CUDA source."""

from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import Calculator, CBSBasis, CCSDTCBSCalculator, Primitive, Shell

from tools.generativeqc_validation.correlation_df_oracle import basis_snapshot, energies

if TYPE_CHECKING:
    from collections.abc import Iterator

HYDROGEN = ((1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7)))
WATER = ((8, (0.0, 0.0, 0.0)), (1, (1.4, 0.0, 1.1)), (1, (-1.4, 0.0, 1.1)))
CH_PLUS = ((6, (0.0, 0.0, 0.0)), (1, (0.3, -0.4, 2.4)))
CYANIDE = (
    (6, tuple(np.array((-0.074650273, 0.694596054, 0.378790432)) / 0.529177210903)),
    (7, tuple(np.array((-0.206743139, 1.592263011, 1.110970151)) / 0.529177210903)),
)


def retain_evidence(name: str, payload: dict) -> None:
    """Opt-in source-matched qualification artifacts, separate from required CI."""
    directory = os.environ.get("GENERATIVEQC_FULL_DF_EVIDENCE_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        payload["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
        payload["source_identity"] = os.environ.get("GENERATIVEQC_VALIDATION_SOURCE_ID")
        (path / (name + ".json")).write_text(
            json.dumps(payload, indent=2, default=lambda value: value.tolist()) + "\n"
        )


@contextmanager
def observed_device_memory(name: str) -> Iterator[dict]:
    """Sample this test process on its allocated GPU, including driver overhead.

    This is optional qualification evidence, not a replacement for the native
    numeric capacity budget. NVML is sampled at 100 ms and may miss short peaks.
    The module's allocation fixture runs before any test enters this scope.
    """
    directory = os.environ.get("GENERATIVEQC_FULL_DF_EVIDENCE_DIR")
    observation = {}
    if not directory:
        yield observation
        return
    assert os.environ.get("SLURM_JOB_ID")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    samples = path / (name + "-device-memory.csv")
    with samples.open("w") as output:
        sampler = subprocess.Popen(
            [
                "nvidia-smi",
                "--id=" + os.environ["CUDA_VISIBLE_DEVICES"],
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
                "--loop-ms=100",
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            yield observation
        finally:
            sampler.terminate()
            sampler.wait(timeout=15)
    values = []
    for line in samples.read_text().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(os.getpid()) and fields[1].isdigit():
            values.append(int(fields[1]) * (1 << 20))
    observation.update(
        {
            "sampling_interval_ms": 100,
            "process_id": os.getpid(),
            "samples": len(values),
            "sampled_device_process_peak_bytes": max(values, default=None),
            "includes": "native allocations, CUDA libraries and driver/context overhead",
        }
    )
    assert values, (
        "allocated process GPU memory sampler produced no numeric observations"
    )


@pytest.fixture(scope="module", autouse=True)
def allocated_cuda() -> None:
    """Never execute device qualification outside an explicit Slurm allocation."""
    if os.environ.get("GENERATIVEQC_FULL_DF_CUDA_TEST") != "1":
        pytest.skip("requires allocated full-DF CUDA qualification")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    pytest.importorskip("pyscf")


def displaced(atoms: tuple, direction: np.ndarray, step: float) -> list:
    """Move both orbital and auxiliary centers by changing the public geometry."""
    return [
        (number, np.asarray(position) + step * shift)
        for (number, position), shift in zip(atoms, direction, strict=True)
    ]


def test_explicit_preconvergence_rejects_legacy_batch() -> None:
    """A prepared batch must not shallow-copy away the execution extension."""
    owner = Calculator(
        method="df-ccsd(t)",
        basis="sto-3g",
        auxiliary_basis="def2-svp",
        density_fitting="cuda",
        device="cuda",
        _correlation_preconvergence="mixed",
    )
    try:
        with pytest.raises(
            NotImplementedError, match="does not support prepared batches"
        ):
            owner.prepare_batch([HYDROGEN], warm_start=False)
    finally:
        owner.clear_cache()


@pytest.mark.parametrize(
    "atoms,orbital,auxiliary,method,cartesian,charge",
    (
        (HYDROGEN, "sto-3g", "def2-svp", "mp2", False, 0),
        (WATER, "cc-pvdz", "cc-pvdz-jkfit", "mp2", False, 0),
        (WATER, "cc-pvtz", "cc-pvtz-jkfit", "mp2", False, 0),
        (WATER, "cc-pvdz", "cc-pvdz-jkfit", "mp2", True, 0),
        (WATER, "def2-svp", "def2-svp-jkfit", "df-ccsd(t)", False, 0),
        (CH_PLUS, "aug-cc-pvqz", "aug-cc-pvqz-jkfit", "df-ccsd(t)", False, 1),
        (CYANIDE, "aug-cc-pvqz", "aug-cc-pvqz-jkfit", "mp2", False, -1),
    ),
)
def test_full_df_complete_energy_force_and_companions(
    atoms: tuple,
    orbital: str,
    auxiliary: str,
    method: str,
    cartesian: bool,
    charge: int,
) -> None:
    """Qualify all companion energies and forces against reconverged PySCF."""
    elements = [atom[0] for atom in atoms]
    owner = Calculator(
        method=method,
        basis=replace(
            basis_snapshot(orbital, elements),
            representation="cartesian" if cartesian else "spherical",
        ),
        auxiliary_basis=replace(
            basis_snapshot(auxiliary, elements),
            representation="cartesian" if cartesian else "spherical",
        ),
        density_fitting="full_cuda",
        device="cuda",
        precision="fp64",
        correlation_memory_budget_bytes=2 << 30,
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-11,
    )
    triples = method == "df-ccsd(t)"
    try:
        with observed_device_memory(f"{orbital}-{method}-{charge}") as memory:
            started = time.monotonic()
            result = owner.singlepoint_components(
                atoms,
                charge=charge,
                companions=("mp2", "rhf") if triples else ("rhf",),
                properties=("energy", "forces"),
            )
            endpoint_seconds = time.monotonic() - started
        evidence = {
            "native_complete_endpoint_seconds": endpoint_seconds,
            "memory_observation": memory,
            "atoms_bohr": atoms,
            "charge": charge,
            "orbital": orbital,
            "auxiliary": auxiliary,
            "cartesian": cartesian,
            "basis_snapshots": {
                "orbital": asdict(basis_snapshot(orbital, elements)),
                "auxiliary": asdict(basis_snapshot(auxiliary, elements)),
            },
            "components": {
                name: asdict(component) for name, component in result.items()
            },
            "finite_differences": [],
        }
        retain_evidence(f"{orbital}-{method}-{charge}-partial", evidence)
        oracle_diagnostics = {}
        expected = energies(
            atoms,
            orbital,
            auxiliary,
            triples=triples,
            df_reference=True,
            cartesian=cartesian,
            charge=charge,
            diagnostics=oracle_diagnostics,
        )
        evidence["oracle_energies"] = expected
        evidence["oracle_root"] = oracle_diagnostics
        retain_evidence(f"{orbital}-{method}-{charge}-partial", evidence)
        for name, component in result.items():
            assert component.converged and component.executed_backend == "cuda"
            assert component.energy == pytest.approx(expected[name], abs=5e-9)
            assert component.forces is not None
            np.testing.assert_allclose(component.forces.sum(axis=0), 0.0, atol=3e-7)
        direction = np.random.default_rng(724).normal(size=(len(atoms), 3))
        direction /= np.linalg.norm(direction)
        for step in (1e-4, 3e-5):
            oracle = [
                energies(
                    displaced(atoms, direction, sign * step),
                    orbital,
                    auxiliary,
                    triples=triples,
                    df_reference=True,
                    cartesian=cartesian,
                    charge=charge,
                )
                for sign in (-1, 1)
            ]
            for name, component in result.items():
                slope = (oracle[1][name] - oracle[0][name]) / (2 * step)
                evidence["finite_differences"].append(
                    {
                        "step_bohr": step,
                        "method": name,
                        "direction": direction,
                        "displaced_energies": [value[name] for value in oracle],
                        "oracle_slope": slope,
                        "native_slope": -np.vdot(component.forces, direction),
                    }
                )
                retain_evidence(f"{orbital}-{method}-{charge}-partial", evidence)
                assert -np.vdot(component.forces, direction) == pytest.approx(
                    slope, abs=3e-6
                )
        replay = owner.singlepoint(
            atoms, charge=charge, properties=("energy", "forces")
        )
        assert replay.energy == pytest.approx(result[method].energy, abs=1e-10)
        np.testing.assert_allclose(
            replay.forces, result[method].forces, atol=1e-9, rtol=0
        )
        energy_only = owner.singlepoint(atoms, charge=charge, properties=("energy",))
        assert energy_only.forces is None
        assert energy_only.energy == pytest.approx(result[method].energy, abs=1e-10)
        retain_evidence(f"{orbital}-{method}-{charge}", evidence)
    finally:
        owner.clear_cache()


@pytest.mark.parametrize("correction_cardinal", ("d", "t"))
def test_augmented_tq_cbs_energy_and_force(correction_cardinal: str) -> None:
    """Unmodified heavy-atom T/Q extrapolation with a separately selected delta."""
    names = ("aug-cc-pvtz", "aug-cc-pvqz", f"aug-cc-pv{correction_cardinal}z")
    bases = tuple(basis_snapshot(name, [6, 1]) for name in names)
    fits = tuple(basis_snapshot(name + "-jkfit", [6, 1]) for name in names)
    workflow = CCSDTCBSCalculator(
        tuple(
            CBSBasis(basis, cardinal)
            for basis, cardinal in zip(bases[:2], (3, 4), strict=True)
        ),
        auxiliary_bases=fits[:2],
        correction_basis=bases[2],
        correction_auxiliary_basis=fits[2],
        hf_reference="df",
        device="cuda",
        preconvergence="fp64",
        policy="tight",
        calculator_options={"correlation_memory_budget_bytes": 2 << 30},
    )
    with observed_device_memory(
        f"augmented-tq-cbs-delta-{correction_cardinal}"
    ) as memory:
        result = workflow.singlepoint(CH_PLUS, charge=1)
    assert result.converged and result.executed_backend == "cuda"
    np.testing.assert_allclose(result.mp2_weights, [-27 / 37, 64 / 37], atol=1e-15)
    exponential = np.exp(-1.63)
    np.testing.assert_allclose(
        result.hf_weights,
        [-exponential / (1 - exponential), 1 / (1 - exponential)],
        atol=1e-15,
    )
    evidence = {
        "result": asdict(result),
        "atoms_bohr": CH_PLUS,
        "charge": 1,
        "basis_snapshots": {
            "orbital": [asdict(basis) for basis in bases],
            "auxiliary": [asdict(basis) for basis in fits],
        },
        "finite_differences": [],
        "memory_observation": memory,
    }
    label = f"augmented-tq-cbs-delta-{correction_cardinal}"
    retain_evidence(label + "-partial", evidence)

    def oracle(geometry: tuple | list) -> np.ndarray:
        values = {}
        for index in (2, 0, 1):
            if bases[index].identity not in values:
                root_diagnostics = {}
                values[bases[index].identity] = energies(
                    geometry,
                    names[index],
                    names[index] + "-jkfit",
                    charge=1,
                    df_reference=True,
                    triples=index == 2,
                    diagnostics=root_diagnostics,
                )
                evidence.setdefault("oracle_roots", []).append(
                    {
                        "basis": names[index],
                        "atoms_bohr": geometry,
                        "root": root_diagnostics,
                    }
                )
        return np.array(
            [
                values[component.basis_identity][component.method]
                for component in result.components
            ]
        )

    coefficients = np.array([component.coefficient for component in result.components])
    expected = oracle(CH_PLUS)
    assert abs(result.energy - coefficients @ expected) < 1e-8
    for index, component in enumerate(result.components):
        assert abs(component.result.energy - expected[index]) < 1e-8
    evidence["oracle_component_energies"] = expected
    direction = np.random.default_rng(3055).normal(size=(2, 3))
    direction /= np.linalg.norm(direction)
    for step in (1e-4, 3e-5):
        slopes = (
            oracle(displaced(CH_PLUS, direction, step))
            - oracle(displaced(CH_PLUS, direction, -step))
        ) / (2 * step)
        for index, component in enumerate(result.components):
            assert -np.vdot(component.result.forces, direction) == pytest.approx(
                slopes[index], abs=3e-6
            )
        assert -np.vdot(result.forces, direction) == pytest.approx(
            coefficients @ slopes, abs=3e-6
        )
        evidence["finite_differences"].append(
            {"step_bohr": step, "direction": direction, "component_slopes": slopes}
        )
        retain_evidence(label + "-partial", evidence)
    retain_evidence(label, evidence)


def test_orbital_g_with_low_angular_fitting_basis() -> None:
    """Orbital g must select raised moments even without a high-l auxiliary.

    This deliberately synthetic small orbital basis tests policy selection;
    it is separate from the unmodified augmented molecular qualification.
    """
    atoms = ((2, (0.2, -0.3, -0.7)), (1, (0.1, 0.2, 0.7)))
    orbital = {"He": [[0, [1.5, 1.0]], [4, [0.6, 1.0]]], "H": [[0, [1.2, 1.0]]]}
    basis = (
        Shell(0, 0, (Primitive(1.5, 1.0),)),
        Shell(0, 4, (Primitive(0.6, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    )
    owner = Calculator(
        method="mp2",
        basis=basis,
        basis_representation="spherical",
        auxiliary_basis=basis_snapshot("def2-svp", [2, 1]),
        device="cuda",
        density_fitting="full_cuda",
        correlation_memory_budget_bytes=512 << 20,
        energy_tolerance=1e-12,
        density_tolerance=1e-11,
    )
    try:
        result = owner.singlepoint_components(
            atoms, charge=1, companions=("rhf",), properties=("energy", "forces")
        )
        expected = energies(atoms, orbital, "def2-svp", charge=1, df_reference=True)
        for name, component in result.items():
            assert component.energy == pytest.approx(expected[name], abs=5e-9)
            np.testing.assert_allclose(component.forces.sum(axis=0), 0.0, atol=3e-7)
        direction = np.random.default_rng(3055).normal(size=(2, 3))
        direction /= np.linalg.norm(direction)
        for step in (1e-4, 3e-5):
            displaced_energies = [
                energies(
                    displaced(atoms, direction, sign * step),
                    orbital,
                    "def2-svp",
                    charge=1,
                    df_reference=True,
                )
                for sign in (-1, 1)
            ]
            for name, component in result.items():
                slope = (displaced_energies[1][name] - displaced_energies[0][name]) / (
                    2 * step
                )
                assert -np.vdot(component.forces, direction) == pytest.approx(
                    slope, abs=3e-6
                )
    finally:
        owner.clear_cache()


@pytest.mark.parametrize("policy", ("standard", "tight"))
@pytest.mark.parametrize("preconvergence", ("fp64", "mixed", "auto"))
@pytest.mark.parametrize("hf_reference", ("exact", "df"))
def test_full_df_cbs_complete_directional_derivative(
    policy: str,
    hf_reference: str,
    preconvergence: str,
) -> None:
    """The composite and every basis group use the same explicit RI-JK model."""
    bases = ("sto-3g", "cc-pvdz")
    auxiliaries = ("def2-svp", "cc-pvdz-jkfit")
    workflow = CCSDTCBSCalculator(
        tuple(
            CBSBasis(basis_snapshot(name, [1]), cardinal)
            for name, cardinal in zip(bases, (2, 3), strict=True)
        ),
        auxiliary_bases=tuple(basis_snapshot(name, [1]) for name in auxiliaries),
        hf_reference=hf_reference,
        policy=policy,
        preconvergence=preconvergence,
        device="cuda",
        calculator_options={"correlation_memory_budget_bytes": 2 << 30},
    )
    result = workflow.singlepoint(HYDROGEN)
    assert result.converged and result.forces is not None
    assert result.basis_metadata["hf_reference"] == hf_reference
    assert len(result.execution_groups) == 2
    cc = next(
        c.result.cc_performance for c in result.components if c.method == "df-ccsd(t)"
    )
    assert cc is not None
    mixed_requested = preconvergence == "mixed" or (
        preconvergence == "auto" and policy == "standard"
    )
    assert cc.ccsd_fp32_requested is mixed_requested
    assert cc.lambda_fp32_requested is mixed_requested
    assert cc.lambda_seconds > 0
    assert cc.lambda_auxiliary_batch_size > 0
    assert cc.lambda_numeric_capacity_bytes >= cc.lambda_owned_device_bytes > 0
    if mixed_requested:
        assert result.convergence["arithmetic_request"] == "df_ccsd_fp32_preconvergence"
        assert cc.ccsd_fp32_executed and not cc.ccsd_fp32_refused
        assert cc.ccsd_fp32_evaluations > 0 and cc.ccsd_fp32_gemm_calls > 0
        assert result.convergence["arithmetic"] == "mixed"
        assert (
            result.convergence["lambda_arithmetic_request"]
            == "df_lambda_fp32_preconvergence"
        )
        assert cc.lambda_fp32_executed and not cc.lambda_fp32_refused
        assert cc.lambda_fp32_actions > 0 and cc.lambda_fp32_gemm_calls > 0
        assert cc.lambda_fp64_actions > 0
    else:
        assert not cc.ccsd_fp32_executed and cc.ccsd_fp32_gemm_calls == 0
        assert not cc.lambda_fp32_executed and cc.lambda_fp32_gemm_calls == 0
    energy_only = workflow.singlepoint(HYDROGEN, properties=("energy",))
    assert energy_only.forces is None
    assert energy_only.energy == pytest.approx(result.energy, abs=1e-10)
    assert energy_only.convergence["lambda_arithmetic"] == "not_requested"
    for component in energy_only.components:
        if component.result.cc_performance is not None:
            assert component.result.cc_performance.lambda_fp32_requested is None

    def oracle(atoms: tuple | list) -> float:
        """Assemble the independent focal-point model, not production components."""
        low = energies(
            atoms,
            bases[0],
            auxiliaries[0],
            triples=True,
            df_reference=hf_reference == "df",
        )
        high = energies(
            atoms, bases[1], auxiliaries[1], df_reference=hf_reference == "df"
        )
        return (
            sum(
                weight * value["rhf"]
                for weight, value in zip(result.hf_weights, (low, high), strict=True)
            )
            + sum(
                weight * (value["mp2"] - value["rhf"])
                for weight, value in zip(result.mp2_weights, (low, high), strict=True)
            )
            + low["df-ccsd(t)"]
            - low["mp2"]
        )

    assert result.energy == pytest.approx(oracle(HYDROGEN), abs=1e-8)
    direction = np.asarray(((0.0, 0.0, -0.5), (0.0, 0.0, 0.5)))
    for step in (1e-4, 3e-5):
        slope = (
            oracle(displaced(HYDROGEN, direction, step))
            - oracle(displaced(HYDROGEN, direction, -step))
        ) / (2 * step)
        assert -np.vdot(result.forces, direction) == pytest.approx(slope, abs=3e-6)


@pytest.mark.parametrize("hf_reference", ("exact", "df"))
def test_strict_mixed_cbs_water_full_force_oracle(hf_reference: str) -> None:
    """Strict warmup preserves all force components, replays and moved centers.

    This compact pair exercises CBS algebra rather than a physical basis limit.
    Independent reconverged energies supply all nine Cartesian derivatives;
    the all-FP64 control cannot substitute for that scientific oracle.
    """
    names = ("sto-3g", "def2-svp")
    auxiliary_name = "def2-svp-jkfit"
    bases = tuple(
        CBSBasis(basis_snapshot(name, [1, 8]), n)
        for name, n in zip(names, (2, 3), strict=True)
    )
    auxiliary = basis_snapshot(auxiliary_name, [1, 8])
    owners = {
        mode: CCSDTCBSCalculator(
            bases,
            auxiliary_bases=(auxiliary, auxiliary),
            hf_reference=hf_reference,
            device="cuda",
            policy="tight",
            preconvergence=mode,
            calculator_options={"correlation_memory_budget_bytes": 2 << 30},
        )
        for mode in ("fp64", "mixed")
    }
    moved = displaced(WATER, np.asarray(((0, 0, 0), (1, 0.5, -1), (0, 0, 0))), 0.03)
    for atoms in (WATER, moved):
        controls = {mode: owner.singlepoint(atoms) for mode, owner in owners.items()}
        result = controls["mixed"]
        assert result.convergence["selected"] == "strict"
        assert result.convergence["budget"] is None
        assert result.convergence["limits"]["ccsd_residual_tolerance"] == 1e-10
        assert abs(result.energy - controls["fp64"].energy) < 1e-9
        np.testing.assert_allclose(
            result.forces, controls["fp64"].forces, atol=1e-8, rtol=0
        )
        replay = owners["mixed"].singlepoint(atoms)
        assert abs(replay.energy - result.energy) < 1e-10
        np.testing.assert_allclose(replay.forces, result.forces, atol=1e-9, rtol=0)
        energy_only = owners["mixed"].singlepoint(atoms, properties=("energy",))
        assert (
            energy_only.forces is None
            and abs(energy_only.energy - result.energy) < 1e-10
        )

        def oracle(
            geometry: tuple | list, components: tuple = result.components
        ) -> np.ndarray:
            low = energies(
                geometry,
                names[0],
                auxiliary_name,
                triples=True,
                df_reference=hf_reference == "df",
            )
            high = energies(
                geometry, names[1], auxiliary_name, df_reference=hf_reference == "df"
            )
            return np.asarray(
                [
                    (low if c.basis_identity == bases[0].basis.identity else high)[
                        c.method
                    ]
                    for c in components
                ]
            )

        expected = oracle(atoms)
        force = np.empty((len(result.components), len(atoms), 3))
        step = 3e-5
        for atom in range(len(atoms)):
            for axis in range(3):
                direction = np.zeros((len(atoms), 3))
                direction[atom, axis] = 1
                force[:, atom, axis] = -(
                    oracle(displaced(atoms, direction, step))
                    - oracle(displaced(atoms, direction, -step))
                ) / (2 * step)
        coefficients = np.asarray([c.coefficient for c in result.components])
        assert abs(result.energy - coefficients @ expected) < 1e-8
        np.testing.assert_allclose(
            result.forces,
            np.einsum("c,cij->ij", coefficients, force),
            atol=3e-6,
            rtol=0,
        )
        for index, component in enumerate(result.components):
            assert abs(component.result.energy - expected[index]) < 1e-8
            np.testing.assert_allclose(
                component.result.forces, force[index], atol=3e-6, rtol=0
            )


def test_routine_mixed_single_occupied_stretched_force() -> None:
    """One occupied orbital still needs physical pair symmetry at the FP64 handoff.

    The spectator-pair contraction optimization is inapplicable to this shape.
    It must not govern removal of FP32 warm-start noise before Lambda admission.
    """
    atoms = ((1, (0.0, 0.0, -4.0)), (1, (0.0, 0.0, 4.0)))
    names = ("cc-pvdz", "cc-pvtz")
    fits = tuple(name + "-jkfit" for name in names)
    bases = tuple(
        CBSBasis(basis_snapshot(name, [1]), n)
        for name, n in zip(names, (2, 3), strict=True)
    )
    kwargs = {
        "auxiliary_bases": tuple(basis_snapshot(name, [1]) for name in fits),
        "hf_reference": "df",
        "device": "cuda",
        "calculator_options": {"correlation_memory_budget_bytes": 2 << 30},
    }
    result = CCSDTCBSCalculator(bases, **kwargs).singlepoint(atoms)
    control = CCSDTCBSCalculator(bases, preconvergence="fp64", **kwargs).singlepoint(
        atoms
    )
    cc = next(c.result for c in result.components if c.method == "df-ccsd(t)")
    assert cc.cc_performance.ccsd_fp32_executed
    assert cc.cc_performance.lambda_fp64_actions > 0
    assert abs(result.energy - control.energy) < 1e-7
    np.testing.assert_allclose(result.forces, control.forces, atol=3e-6, rtol=0)

    def oracle(geometry: list | tuple) -> np.ndarray:
        lower = energies(geometry, names[0], fits[0], triples=True, df_reference=True)
        higher = energies(geometry, names[1], fits[1], df_reference=True)
        return np.array(
            [
                (higher if "_high" in c.roles[0] else lower)[c.method]
                for c in result.components
            ]
        )

    coefficients = np.array([c.coefficient for c in result.components])
    assert abs(result.energy - coefficients @ oracle(atoms)) < 1e-7
    expected = np.zeros((2, 3))
    for atom in range(2):
        direction = np.zeros((2, 3))
        direction[atom, 2] = 1
        expected[atom, 2] = (
            -coefficients
            @ (
                oracle(displaced(atoms, direction, 3e-5))
                - oracle(displaced(atoms, direction, -3e-5))
            )
            / 6e-5
        )
    np.testing.assert_allclose(result.forces, expected, atol=3e-6, rtol=0)
