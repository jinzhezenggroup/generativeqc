"""Native exactly-once B3LYP/D3(BJ) composition with elemental Cs identity."""

from dataclasses import replace
from time import perf_counter

import numpy as np
import pytest
from generativeqc import Calculator, KsOptions, evaluate_d3_correction
from generativeqc_compiler.method import DispersionCorrectionPrimitive
from generativeqc_compiler.method._generated_parameters import d3_parameters
from generativeqc_compiler.method.dispersion import D3Spec
from test_global_hybrid_ecp_basis import canonical_records as _canonical_records
from test_global_hybrid_ecp_forces import GRID, make_calculator, molecule_atoms
from test_integral_autodiff_second import require_native_library  # noqa: F401

ecp_records = _canonical_records


@pytest.mark.parametrize("charge,multiplicity", ((0, 1), (-1, 2)))
def test_cs_global_hybrid_composes_independent_bj_correction_once(
    charge: int,
    multiplicity: int,
    ecp_records: tuple,
    record_property: object,
) -> None:
    """ECP46 leaves nine valence electrons but dispersion must still receive Z=55."""
    interface = pytest.importorskip("dftd3.interface")
    _, _, record = ecp_records
    atoms = molecule_atoms("Cs")
    plain_calculator = make_calculator(record, multiplicity)
    specification = D3Spec(**d3_parameters("B3LYP-D3(BJ)"))
    assert specification.s9 == 0
    assert specification.cn_cutoff is None and specification.pair_cutoff is None
    graph = replace(
        plain_calculator.method_ir,
        identifier="physical-b3lyp-def2-ecp-with-bj",
        primitives=(
            *plain_calculator.method_ir.primitives,
            DispersionCorrectionPrimitive(specification),
        ),
    )
    corrected_calculator = Calculator(
        method=graph,
        basis=record,
        basis_representation="spherical",
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
    )
    plain = plain_calculator.singlepoint(
        atoms, charge=charge, multiplicity=multiplicity, properties=("energy", "forces")
    )
    started = perf_counter()
    corrected = corrected_calculator.singlepoint(
        atoms, charge=charge, multiplicity=multiplicity, properties=("energy", "forces")
    )
    record_property("whole_composed_endpoint_seconds", perf_counter() - started)
    assert plain.converged and corrected.converged
    numbers = np.array((55, 1))
    positions = np.array([center for _, center in atoms])
    damping = interface.RationalDampingParam(
        s6=specification.s6,
        s8=specification.s8,
        a1=specification.a1,
        a2=specification.a2,
        s9=0,
    )
    independent = interface.DispersionModel(numbers, positions).get_dispersion(
        damping, grad=True
    )
    native = evaluate_d3_correction(graph, numbers, positions)
    assert abs(native.energy - float(independent["energy"])) <= 2e-13
    np.testing.assert_allclose(
        native.gradient, independent["gradient"], atol=2e-12, rtol=0
    )
    assert corrected.dispersion is not None
    assert abs(corrected.energy - plain.energy - float(independent["energy"])) <= 2e-11
    np.testing.assert_allclose(
        corrected.forces, plain.forces - independent["gradient"], atol=2e-9, rtol=0
    )
    assert np.max(abs(independent["gradient"])) > 1e-7
    assert np.max(abs(corrected.forces - plain.forces)) > 1e-7
    assert (
        np.max(abs(corrected.forces - (plain.forces - 2 * independent["gradient"])))
        > 1e-7
    )
    wrong_identity = interface.DispersionModel(
        np.array((9, 1)), positions
    ).get_dispersion(damping, grad=True)
    assert np.max(abs(native.gradient - wrong_identity["gradient"])) > 1e-7
    record_property(
        "independent_d3_force_error",
        float(np.max(abs(corrected.forces - plain.forces + independent["gradient"]))),
    )
