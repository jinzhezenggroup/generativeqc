"""Independent native Gaussian-ECP snapshot and response-source contracts."""

import ctypes as ct
from dataclasses import replace

import numpy as np
import pytest
from generativeqc import BasisProvenance, _native
from generativeqc._dft_gradient import StationaryKsState
from generativeqc.ecp_hessian import ECPSecondOrder
from generativeqc_compiler.dft import NativeAO
from test_ecp import _record_from_resolved_basis
from test_global_hybrid_ecp_forces import make_calculator
from test_integral_autodiff_second import require_native_library  # noqa: F401


@pytest.fixture
def native_case() -> tuple:
    """Bind an actual native SCF to explicitly synthetic, test-only ECP data."""
    gto = pytest.importorskip("pyscf.gto")
    basis = {"Na": [[0, [0.65, 1.0]], [1, [0.45, 1.0]]], "H": [[0, [0.8, 1.0]]]}
    terms = [
        [channel, [[], [], [[0.67, -0.12], [1.23, 0.081]]]] for channel in (-1, 0, 1, 2)
    ]
    record = _record_from_resolved_basis(basis, {"Na": [10, terms]}, "spherical")
    record = replace(
        record,
        name="Synthetic Gaussian ECP snapshot oracle",
        provenance=BasisProvenance(
            "Explicit test-only Gaussian coefficients in test_ecp_gaussian_response.py",
            "1",
            "Repository test fixture",
            record.provenance.checksum,
        ),
    )
    atoms = [("Na", (0.13, -0.21, 0.17)), ("H", (0.43, 0.19, 3.2))]
    references = [
        gto.M(
            atom=atoms, basis=basis, ecp={"Na": [10, selected]}, unit="Bohr", verbose=0
        )
        for selected in ([terms[0]], terms[1:])
    ]
    calculator = make_calculator(record)
    with (
        calculator.prepare_batch([atoms], warm_start=False) as batch,
        NativeAO(atoms, basis=record, representation="spherical") as ao,
    ):
        batch.execute(strict=True, properties=("energy",))
        state = StationaryKsState.from_native(batch, ao)
        try:
            yield state, ao, batch, references
        finally:
            state._source.close()


def shifted_matrix(molecule: object, positions: np.ndarray) -> np.ndarray:
    """Only the independent PySCF ECP primal is used to differentiate the oracle."""
    shifted = molecule.copy()
    shifted.set_geom_(positions, unit="Bohr")
    return shifted.intor("ECPscalar")


def test_snapshot_panels_match_independent_primal_and_all_first_derivatives(
    native_case: tuple,
) -> None:
    state, _, _, references = native_case
    panels, work = state._source.ecp_gaussian(budget_bytes=32 << 20)
    assert not panels.flags.writeable
    assert work["kernel_blocks"] > 0
    assert work["kernel_blocks"] <= work["kernel_block_bound"]
    positions = references[0].atom_coords()
    offsets, weights = (-2, -1, 1, 2), (1, -8, 8, -1)
    for part, reference in enumerate(references):
        np.testing.assert_allclose(
            panels[part, 0], reference.intor("ECPscalar"), atol=2e-10, rtol=0
        )
        for step in (0.003, 0.001):
            for coordinate in range(positions.size):
                derivative = np.zeros_like(panels[part, 0])
                for offset, weight in zip(offsets, weights, strict=True):
                    moved = positions.copy()
                    moved.reshape(-1)[coordinate] += offset * step
                    derivative += (
                        weight * shifted_matrix(reference, moved) / (12 * step)
                    )
                np.testing.assert_allclose(
                    panels[part, 1 + coordinate], derivative, atol=2e-8, rtol=0
                )


@pytest.mark.parametrize(
    "direction",
    (np.eye(6)[0].reshape(2, 3), np.array([[0.2, -0.3, 0.4], [-0.1, 0.5, 0.2]])),
)
def test_native_response_sources_match_independent_complete_mixed_derivatives(
    native_case: tuple, direction: np.ndarray
) -> None:
    state, basis, _, references = native_case
    owner = ECPSecondOrder(state, basis, budget_bytes=32 << 20)
    density = state.density.sum(axis=0)
    random = np.random.default_rng(2847)
    moving_density = random.normal(size=density.shape)
    result = owner.hvp(direction, density, moving_density)
    assert owner.statistics["analytic_gaussian"]
    assert owner.statistics["radial_layers"] == 0
    positions = references[0].atom_coords()
    offsets, weights = (-2, -1, 1, 2), (1, -8, 8, -1)
    for part, (name, reference) in enumerate(
        zip(("ecp_local", "ecp_nonlocal"), references, strict=True)
    ):
        assert not result[name].flags.writeable
        for step in (0.003, 0.001):
            independent = np.zeros(positions.size)
            for coordinate in range(positions.size):
                for left, left_weight in zip(offsets, weights, strict=True):
                    for right, right_weight in zip(offsets, weights, strict=True):
                        moved = positions + right * step * direction
                        moved = moved.copy()
                        moved.reshape(-1)[coordinate] += left * step
                        matrix = shifted_matrix(reference, moved)
                        value = np.sum(
                            (density + right * step * moving_density) * matrix
                        )
                        independent[coordinate] += (
                            left_weight * right_weight * value / (144 * step**2)
                        )
            np.testing.assert_allclose(
                result[name].reshape(-1), independent, atol=2e-8, rtol=0
            )
    frozen = owner.frozen_fock(direction)
    for step in (0.003, 0.001):
        independent = np.zeros_like(frozen)
        for reference in references:
            for offset, weight in zip(offsets, weights, strict=True):
                independent += (
                    weight
                    * shifted_matrix(reference, positions + offset * step * direction)
                    / (12 * step)
                )
        np.testing.assert_allclose(frozen, independent, atol=2e-8, rtol=0)


def test_snapshot_budget_recovery_and_stale_owner_leave_caller_buffers_untouched(
    native_case: tuple,
) -> None:
    state, _, batch, references = native_case
    source = state._source
    baseline, _ = source.ecp_gaussian(budget_bytes=32 << 20)
    with pytest.raises(MemoryError):
        source.ecp_gaussian(budget_bytes=1)
    replay, _ = source.ecp_gaussian(budget_bytes=32 << 20)
    np.testing.assert_array_equal(replay, baseline)
    evaluate = source._library.generativeqc_ks_snapshot_ecp_gaussian_v1
    output = np.full_like(baseline, 173.0)
    work = np.full(8, 37, dtype=np.uint64)
    pointer = ct.POINTER(ct.c_double)

    def invoke(kernel_blocks: int) -> int:
        return evaluate(
            batch._batch,
            source._handle,
            0,
            None,
            0,
            32 << 20,
            kernel_blocks,
            output.ctypes.data_as(pointer),
            output.size,
            work.ctypes.data_as(ct.POINTER(ct.c_uint64)),
            work.size,
        )

    assert invoke(1) == _native.STATUS_RESOURCE_LIMIT
    np.testing.assert_array_equal(output, 173.0)
    np.testing.assert_array_equal(work, 37)
    moved = references[0].atom_coords()
    moved[1, 2] += 0.02
    batch.execute(coordinates=[moved], strict=True, properties=("energy",))
    assert invoke(64 << 20) == _native.STATUS_INVALID_ARGUMENT
    np.testing.assert_array_equal(output, 173.0)
    np.testing.assert_array_equal(work, 37)
