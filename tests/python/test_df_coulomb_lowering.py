"""Independent Coulomb algebra and canonical CPU/CUDA request contracts."""

import typing

import numpy as np
import pytest
from generativeqc_compiler.tensor.df_coulomb import coulomb_program
from generativeqc_compiler.tensor.interpreter import execute
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter
from generativeqc_compiler.tensor.vector_lowering import coulomb_portfolio


@pytest.mark.parametrize("batch,nbf,naux", [(1, 1, 1), (2, 3, 7), (3, 5, 2)])
def test_dense_and_packed_coulomb_against_four_center_oracle(
    batch: int, nbf: int, naux: int
) -> None:
    rng = np.random.default_rng(8190)
    factor = rng.normal(size=(batch, nbf, nbf, naux))
    factor += factor.swapaxes(1, 2)
    # Deliberately nonsymmetric: packed weights must be D_ij + D_ji, not 2 D_ij.
    density = rng.normal(size=(batch, nbf, nbf))
    expected = np.zeros_like(density)
    for b in range(batch):
        for i in range(nbf):
            for j in range(nbf):
                for k in range(nbf):
                    for l in range(nbf):
                        integral = sum(
                            float(factor[b, i, j, q]) * float(factor[b, k, l, q])
                            for q in range(naux)
                        )
                        expected[b, i, j] += integral * density[b, k, l]
    pairs = [(i, j) for i in range(nbf) for j in range(i + 1)]
    packed_factor = np.stack([factor[:, i, j] for i, j in pairs], axis=1)
    packed_density = np.stack(
        [density[:, i, j] + (density[:, j, i] if i != j else 0) for i, j in pairs],
        axis=1,
    )
    dense = execute(
        coulomb_program(batch, nbf, naux, packed=False),
        {"factor": factor, "density": density},
    ).outputs
    packed = execute(
        coulomb_program(batch, nbf, naux, packed=True),
        {"factor": packed_factor, "density": packed_density},
    ).outputs
    np.testing.assert_allclose(dense["coulomb"], expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        packed["charge"], dense["charge"], atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        packed["coulomb"],
        np.stack([expected[:, i, j] for i, j in pairs], axis=1),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("output", ["charge", "coulomb"])
def test_same_semantics_and_precision_across_backends_and_candidates(
    packed: bool, output: str
) -> None:
    program = coulomb_program(2, 3, 5, packed=packed)
    adapter = TensorLoweringAdapter(program)
    cpu = adapter.request(program.outputs[output], backend="cpu")
    cuda = adapter.request(program.outputs[output], backend="cuda")
    assert cpu.scientific_identity == cuda.scientific_identity
    assert cpu.semantic_identity == cuda.semantic_identity
    _, _, request, candidates, _, _ = coulomb_portfolio(packed, output, "a" * 64)
    assert request.semantic_identity == cuda.semantic_identity
    assert len(candidates) == 2
    for candidate in candidates:
        assert candidate.request is request
        assert candidate.execution is not None
        assert candidate.execution.precision == request.precisions[0]
        assert candidate.execution.capture_safe
        assert candidate.cost is None  # Unknown costs cannot justify promotion.


@pytest.mark.parametrize("dimension", [0, -1, True, 1.5])
def test_invalid_scientific_extents_fail_before_generation(
    dimension: typing.Any,
) -> None:
    with pytest.raises(ValueError):
        coulomb_program(1, dimension, 2, packed=False)
