"""Public autodiff argument ownership and disconnected-input regressions."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.tensor import Program
from generativeqc_compiler.tensor.cpu import emit_cpu


@pytest.mark.parametrize("backend", ("reference", "cpu"))
def test_unused_gradient_tracks_argument_specialization(
    backend: str, tmp_path: Path
) -> None:
    def energy(values: Any, unused: Any) -> Any:
        return xp.sum(values * values)

    values = np.asarray([1.0, 2.0])
    derivative = xp.grad(energy, argnums=1, backend=backend, cache=tmp_path)
    combined = xp.value_and_grad(energy, argnums=1, backend=backend, cache=tmp_path)
    try:
        hashes = []
        for shape in ((3,), (2, 4), (0,)):
            unused = np.ones(shape)
            hashes.append(derivative.function.lower(values, unused).logical_hash)
            np.testing.assert_array_equal(derivative(values, unused), np.zeros(shape))
            value, gradient = combined(values, unused)
            assert float(value) == 5.0
            np.testing.assert_array_equal(gradient, np.zeros(shape))
        assert len(set(hashes)) == 1
        assert len(derivative.programs) == 3
    finally:
        derivative.close()
        combined.close()


@pytest.mark.parametrize("backend", ("reference", "cpu"))
def test_disconnected_jvp_vjp_and_seed_validation(backend: str, tmp_path: Path) -> None:
    def expression(values: Any, unused: Any) -> Any:
        return {"active": values * values, "constant": xp.ones_like(unused)}

    values, unused = np.asarray([1.0, 2.0]), np.ones(3)
    directions = (np.asarray([0.5, -0.5]), np.ones(3))
    primal, tangent = xp.jvp(
        expression, (values, unused), directions, backend=backend, cache=tmp_path
    )
    np.testing.assert_array_equal(primal["active"], [1.0, 4.0])
    np.testing.assert_array_equal(tangent["active"], [1.0, -2.0])
    np.testing.assert_array_equal(tangent["constant"], np.zeros(3))
    with pytest.raises(ValueError, match="tangent unused must be finite"):
        xp.jvp(
            expression,
            (values, unused),
            (directions[0], np.full(3, np.nan)),
            backend=backend,
            cache=tmp_path,
        )
    _, pullback = xp.vjp(expression, values, unused, backend=backend, cache=tmp_path)
    try:
        gradients = pullback({"active": np.ones(2), "constant": np.ones(3)})
        np.testing.assert_array_equal(gradients[0], [2.0, 4.0])
        np.testing.assert_array_equal(gradients[1], np.zeros(3))
        with pytest.raises(ValueError, match="cotangent constant must be finite"):
            pullback({"active": np.ones(2), "constant": np.full(3, np.inf)})
    finally:
        pullback.close()


def test_shape_only_function_has_zero_gradient_and_validates_primal() -> None:
    derivative = xp.grad(
        lambda values: xp.sum(xp.ones_like(values)), backend="reference"
    )
    np.testing.assert_array_equal(derivative(np.ones(3)), np.zeros(3))
    with pytest.raises(ValueError, match="non-finite compiled input: values"):
        derivative(np.full(3, np.nan))


def test_retained_unused_inputs_preserve_live_hash_and_cpu_source() -> None:
    compiled = xp.compile(lambda values, unused: values * values)
    program = compiled.lower(np.ones(2), np.ones(3))
    assert {node.attrs["name"] for node in program.nodes if node.op == "input"} == {
        "values",
        "unused",
    }
    live_only = Program(program.outputs, provenance=program.provenance)
    assert program.logical_hash == live_only.logical_hash
    assert emit_cpu(program) == emit_cpu(live_only)


def test_gradient_metadata_cannot_replace_transform_state() -> None:
    def energy(values: Any) -> Any:
        return xp.sum(values * values)

    energy.with_value = True  # type: ignore[attr-defined]
    energy.programs = {"unrelated": None}  # type: ignore[attr-defined]
    derivative = xp.grad(energy, backend="reference")
    assert derivative.with_value is False
    assert not derivative.programs
    np.testing.assert_array_equal(derivative(np.asarray([1.0, 2.0])), [2.0, 4.0])
