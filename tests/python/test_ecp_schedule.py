"""Radial batch tails, bounded fallback grids and strict workspace accounting."""

import typing

import numpy as np
import pytest
from generativeqc.ecp import ecp_integrals
from generativeqc.resources_hf import _ecp_workspace
from generativeqc_compiler.integral.ecp_policy import GAUSSIAN_MAX_SCRATCH_DOUBLES
from generativeqc_compiler.integral.ecp_schedule import cuda_radial_tile
from test_ecp import fixture, reference
from test_ecp_f import require_device


@pytest.mark.parametrize("primitives", (10, 10000))
def test_radial_workspace_public_vs_cartesian_counts(primitives: int) -> None:
    # A spherical f-shell case has 16 public AOs and 19 Cartesian components.
    # Native schedule selection uses the former; storage bounds use the latter.
    item = {
        "atoms": 2,
        "orbital": {
            "nbf": 16,
            "cartesian_nbf": 19,
            "primitives": primitives,
            "ecp_terms": 5,
        },
    }
    assert cuda_radial_tile(16, 44) == 4
    assert cuda_radial_tile(17, 44) == cuda_radial_tile(16, 45) == 1
    cpu = _ecp_workspace(item)
    batched_cuda = _ecp_workspace(item, cuda=True)
    item["orbital"]["nbf"] = 19
    scalar_cuda = _ecp_workspace(item, cuda=True)
    assert batched_cuda - scalar_cuda == 3 * 32 * 19 * (2 * 44 * 44 + 16)
    assert _ecp_workspace(item) == cpu
    # CPU reserves either grid or analytic storage; changing the public AO
    # count changes only CUDA's radial tile, not either CPU storage bound.
    analytic = (
        48 * 19**2 * (1 + 3 * 2)
        + 8 * GAUSSIAN_MAX_SCRATCH_DOUBLES
        + 144 * (32 + 225 * 8)
        + 2 * 43 * 100 * 16
        + 100 * 6 * 8
        + 256 * 5
        + 4096 * min(primitives, 19)
    )
    assert cpu == max(scalar_cuda, analytic)


@pytest.mark.parametrize("f_projector", [False, True])
@pytest.mark.parametrize("polar", [32, 44, 45])
@pytest.mark.parametrize("derivatives", [False, True])
def test_cuda_radial_tail_and_grid_fallback(
    polar: typing.Any, derivatives: typing.Any, f_projector: typing.Any
) -> None:
    require_device("cuda")
    atoms, basis, mol = fixture(f_projector=f_projector)
    # 161 leaves a one-layer tail in the four-layer schedule; 45 polar points
    # selects the bounded original schedule, independently of AO count.
    kwargs = {"radial_points": 161, "polar_points": polar, "derivatives": derivatives}
    gpu = ecp_integrals(atoms, basis, device="cuda", **kwargs)
    cpu = ecp_integrals(atoms, basis, **kwargs)
    np.testing.assert_allclose(gpu.matrix, reference(mol), atol=2e-9, rtol=0)
    for field in ("local", "nonlocal_", "local_derivative", "nonlocal_derivative"):
        if getattr(cpu, field) is None:
            assert getattr(gpu, field) is None
            continue
        np.testing.assert_allclose(
            getattr(gpu, field), getattr(cpu, field), atol=2e-11, rtol=0
        )
