"""COSX value algebra shares canonical CPU/CUDA identities and seed effects."""

import numpy as np
import pytest
from generativeqc_compiler.dft.cosx_contraction import (
    cosx_matrix_program,
    emit_cosx_contractions,
)
from generativeqc_compiler.tensor.contraction_update import contraction_update_request
from generativeqc_compiler.tensor.interpreter import execute
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter


@pytest.mark.parametrize("points,columns", [(1, 1), (7, 13), (17, 5)])
@pytest.mark.parametrize("update", [False, True])
def test_nonsymmetric_matrix_algebra(points: int, columns: int, update: bool) -> None:
    """Asymmetric factors detect hidden density transpose/symmetry assumptions."""
    program = cosx_matrix_program(points, columns, update=update)
    rng = np.random.default_rng(1884)
    ao = rng.normal(size=(points, columns))
    other = rng.normal(size=(points if update else columns, columns))
    seed = rng.normal(size=(columns, columns))
    expected = np.zeros((columns if update else points, columns))
    for i in range(expected.shape[0]):
        for j in range(columns):
            value = np.longdouble(seed[i, j]) if update else np.longdouble(0)
            for k in range(points if update else columns):
                value += np.longdouble(
                    ao[k, i] if update else ao[i, k]
                ) * np.longdouble(other[k, j])
            expected[i, j] = value
    inputs = {"ao": ao, "potential" if update else "density": other}
    if update:
        inputs["seed"] = seed
    np.testing.assert_allclose(
        execute(program, inputs).outputs["result"], expected, atol=3e-13, rtol=3e-13
    )
    root = program.outputs["result"]
    adapter = TensorLoweringAdapter(program)
    requests = [
        contraction_update_request(adapter, root, backend=backend)
        if update
        else adapter.request(root, backend=backend)
        for backend in ("cpu", "cuda")
    ]
    assert requests[0].semantic_identity == requests[1].semantic_identity
    assert requests[0].scientific_identity == requests[1].scientific_identity
    if update:
        assert dict(requests[0].effects) == {
            "output": "donated",
            "donated_input": "input:2",
        }
        assert requests[0].operands[-1].alias_group == "donated-seed"


def test_emission_is_deterministic_and_provider_neutral() -> None:
    source = emit_cosx_contractions()
    assert source == emit_cosx_contractions()
    assert '"cublas"' in source and '"generated.cuda"' in source
    assert "cublasDgemm" not in source and "cublasCreate" not in source
    assert "PreparedContractionSites<4>" in source
