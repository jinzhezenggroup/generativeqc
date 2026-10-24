"""End-to-end native Array API compilation and quantum-chemistry derivatives."""

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp


@pytest.fixture
def forbid_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    from generativeqc_compiler.tensor import autodiff, interpreter

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("native Array API delegated to a reference oracle")

    monkeypatch.setattr(interpreter, "execute", forbidden)
    monkeypatch.setattr(autodiff, "jvp", forbidden)
    monkeypatch.setattr(autodiff, "vjp", forbidden)


def test_default_native_and_single_endpoint_value_and_grad(
    tmp_path: Path, forbid_reference: None
) -> None:
    @xp.compile(cache=tmp_path)
    def norm2(values: Any) -> Any:
        return xp.sum(values * values)

    values = xp.asarray([1.0, 2.0, 3.0])
    assert float(norm2(values)) == 14
    assert norm2.compilation_info()["backend"] == "cpu"
    assert norm2.compilation_info()["native_executables"] == 1
    transformed = xp.value_and_grad(norm2)
    program = transformed.lower(values)
    assert set(program.outputs) == {"output", "bar_values"}
    value, gradient = transformed(values)
    assert float(value) == 14
    np.testing.assert_array_equal(gradient, [2, 4, 6])
    transformed(values + 1)
    assert transformed.compilation_info()["native_executables"] == 1
    assert transformed.compilation_info()["specializations"] == 1
    transformed.close()
    with pytest.raises(RuntimeError, match="closed"):
        transformed(values)


def test_grad_argument_order_broadcast_and_keyword_binding(
    tmp_path: Path, forbid_reference: None
) -> None:
    def energy(matrix: Any, shift: Any) -> Any:
        return xp.sum((matrix + shift) * (matrix + shift))

    matrix = np.arange(6.0).reshape(2, 3)
    shift = np.asarray([0.2, 0.3, 0.4])
    derivative = xp.grad(energy, argnums=(-1, 0), cache=tmp_path)
    shift_gradient, matrix_gradient = derivative(matrix=matrix, shift=shift)
    np.testing.assert_allclose(matrix_gradient, 2 * (matrix + shift), atol=1e-14)
    np.testing.assert_allclose(
        shift_gradient, (2 * (matrix + shift)).sum(axis=0), atol=1e-14
    )


def test_native_nonlinear_gaussian_derivatives(
    tmp_path: Path, forbid_reference: None
) -> None:
    def energy(exponents: Any) -> Any:
        return xp.sum(
            xp.exp(-exponents * exponents)
            + xp.log(exponents)
            + xp.sqrt(exponents)
            + xp.pow(exponents, Fraction(3, 2))
        )

    exponents = np.asarray([0.2, 0.7, 1.4])
    value, derivative = xp.value_and_grad(energy, cache=tmp_path)(exponents)
    expected_value = np.sum(
        np.exp(-(exponents**2))
        + np.log(exponents)
        + np.sqrt(exponents)
        + exponents**1.5
    )
    expected_gradient = (
        -2 * exponents * np.exp(-(exponents**2))
        + 1 / exponents
        + 0.5 / np.sqrt(exponents)
        + 1.5 * np.sqrt(exponents)
    )
    np.testing.assert_allclose(value, expected_value, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(derivative, expected_gradient, rtol=2e-13, atol=2e-13)


def test_jvp_and_snapshot_vjp_native(tmp_path: Path, forbid_reference: None) -> None:
    def expression(values: Any) -> Any:
        return {"energy": xp.sum(values * values), "vector": values * values}

    values = xp.asarray([0.2, 0.7, 1.4])
    tangent = xp.asarray([0.3, -0.2, 0.4])
    primal, response = xp.jvp(expression, (values,), (tangent,), cache=tmp_path)
    np.testing.assert_allclose(primal["vector"], values * values)
    np.testing.assert_allclose(response["vector"], 2 * values * tangent)
    np.testing.assert_allclose(response["energy"], np.sum(2 * values * tangent))
    saved_values = np.array(values, copy=True)
    primal, pullback = xp.vjp(expression, values, cache=tmp_path)
    values[:] = 100
    cotangents = {"energy": xp.asarray(0.5), "vector": xp.asarray([1.0, 2.0, 3.0])}
    (gradient,) = pullback(cotangents)
    np.testing.assert_allclose(gradient, 2 * saved_values * (0.5 + np.arange(1.0, 4.0)))
    pullback.close()
    with pytest.raises(RuntimeError, match="closed"):
        pullback(cotangents)


def _rhf_energy(density: Any, hcore: Any, eri: Any) -> Any:
    return (
        xp.einsum("pq,pq->", density, hcore)
        + xp.einsum("pq,rs,pqrs->", density, density, eri, coefficient=Fraction(1, 2))
        - xp.einsum("pq,rs,prqs->", density, density, eri, coefficient=Fraction(1, 4))
    )


def test_scientific_rhf_density_gradient_is_fock(
    tmp_path: Path, forbid_reference: None
) -> None:
    rng = np.random.default_rng(1027)
    size = 3
    factors = rng.normal(size=(4, size, size))
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    eri = np.einsum("apq,ars->pqrs", factors, factors)
    density = rng.normal(size=(size, size))
    density = (density + density.T) / 2
    hcore = rng.normal(size=(size, size))
    hcore = (hcore + hcore.T) / 2
    ao = xp.IndexSpace("ao", "ao", size)
    matrix_spec = xp.TensorSpec(
        tuple(xp.Index(name, ao) for name in ("p", "q")), role="input"
    )
    eri_spec = xp.TensorSpec(
        tuple(xp.Index(name, ao) for name in ("p", "q", "r", "s")), role="input"
    )
    compiled = xp.compile(
        _rhf_energy,
        cache=tmp_path,
        specs={"density": matrix_spec, "hcore": matrix_spec, "eri": eri_spec},
    )
    transformed = xp.value_and_grad(compiled)
    value, fock = transformed(density, hcore, eri)
    coulomb = np.einsum("rs,pqrs->pq", density, eri)
    exchange = np.einsum("rs,prqs->pq", density, eri)
    expected_fock = hcore + coulomb - exchange / 2
    expected_energy = np.sum(density * (hcore + expected_fock)) / 2
    np.testing.assert_allclose(value, expected_energy, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(fock, expected_fock, rtol=2e-13, atol=2e-13)
    direction = rng.normal(size=density.shape)
    step = 1e-5
    finite_difference = (
        compiled(density + step * direction, hcore, eri)
        - compiled(density - step * direction, hcore, eri)
    ) / (2 * step)
    np.testing.assert_allclose(
        finite_difference, np.sum(expected_fock * direction), rtol=2e-9, atol=2e-9
    )
    original = compiled.lower(density, hcore, eri)
    assert all(
        index.space == ao for node in original.nodes for index in node.spec.indices
    )


def test_scientific_domains_and_budgets_fail_before_toolchain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.common import cpp_adapter

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("invalid semantic/resource request reached toolchain")

    monkeypatch.setattr(cpp_adapter, "CppCompilerAdapter", forbidden)
    ao = xp.IndexSpace("ao", "ao", 2)
    occupied = xp.IndexSpace("occupied", "occupied", 2)
    ao_spec = xp.TensorSpec((xp.Index("p", ao),), role="input")
    occupied_spec = xp.TensorSpec((xp.Index("i", occupied),), role="input")
    compiled = xp.compile(
        lambda left, right: xp.sum(left * right),
        specs={"left": ao_spec, "right": occupied_spec},
        cache=tmp_path,
    )
    with pytest.raises((TypeError, ValueError)):
        compiled(np.ones(2), np.ones(2))
    for options in ({"max_bytes": 1}, {"max_work": 0}, {"max_nodes": 1}):
        with pytest.raises(ValueError):
            xp.value_and_grad(
                lambda values: xp.sum(values * values), cache=tmp_path, **options
            )(np.ones(4))


def test_transform_validation_and_no_implicit_dtype_promotion(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scalar"):
        xp.grad(lambda values: values, cache=tmp_path)(np.ones(2))
    with pytest.raises(ValueError, match="distinct"):
        xp.grad(lambda values: xp.sum(values), argnums=(0, 0))
    with pytest.raises(ValueError, match="float64"):
        xp.grad(lambda values: xp.sum(values * values), cache=tmp_path)(
            np.ones(2, dtype=np.float32)
        )
    with pytest.raises(NotImplementedError, match="capture"):
        xp.grad(lambda values: xp.sum(xp.sin(values)), cache=tmp_path)(np.ones(2))
    with pytest.raises(ValueError, match="shape and dtype"):
        xp.jvp(
            lambda values: values * values, (np.ones(2),), (np.ones(3),), cache=tmp_path
        )
    with pytest.raises(ValueError, match="finite"):
        xp.compile(lambda values: values, compile_timeout=float("inf"))


def test_recompiling_preserves_requested_backend_budget_and_independent_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.common import cpp_adapter

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("over-budget recompiled function reached toolchain")

    monkeypatch.setattr(cpp_adapter, "CppCompilerAdapter", forbidden)
    ao = xp.IndexSpace("ao", "ao", 2)
    spec = xp.TensorSpec((xp.Index("p", ao),), role="input")
    original = xp.compile(
        lambda values: xp.sum(values * values),
        backend="reference",
        specs={"values": spec},
    )
    recompiled = xp.compile(original, backend="cpu", max_bytes=1, cache=tmp_path)
    assert recompiled.compilation_info()["backend"] == "cpu"
    assert recompiled._options["max_bytes"] == 1
    assert recompiled._execution is not original._execution
    assert recompiled._specs == original._specs
    with pytest.raises(ValueError, match="byte/work budget"):
        recompiled(np.ones(2))
    recompiled.close()
    assert float(original(np.ones(2))) == 2
    original.close()


def test_callable_metadata_cannot_replace_compilation_state(tmp_path: Path) -> None:
    def energy(values: Any) -> Any:
        return xp.sum(values * values)

    energy.__dict__.update(
        _backend="reference", _execution=object(), _options={"max_bytes": 1 << 40}
    )
    compiled = xp.compile(energy, backend="cpu", max_bytes=1, cache=tmp_path)
    assert compiled.compilation_info()["backend"] == "cpu"
    assert compiled._options["max_bytes"] == 1
    with pytest.raises(ValueError, match="byte/work budget"):
        compiled(np.ones(2))
    compiled.close()


def test_inactive_inputs_and_derivative_seeds_are_still_finite(tmp_path: Path) -> None:
    def zero(values: Any) -> Any:
        return xp.sum(values * 0)

    with pytest.raises(ValueError, match="non-finite compiled input: values"):
        xp.grad(zero, cache=tmp_path)(np.asarray([np.nan]))
    with pytest.raises(ValueError, match="tangent values must be finite"):
        xp.jvp(zero, (np.ones(1),), (np.asarray([np.nan]),), cache=tmp_path)
    value, pullback = xp.vjp(zero, np.ones(1), cache=tmp_path)
    assert float(value) == 0
    try:
        with pytest.raises(ValueError, match="cotangent output must be finite"):
            pullback(np.asarray(np.inf))
    finally:
        pullback.close()


@pytest.mark.parametrize(
    "operation,values", [(xp.log, [-1.0]), (xp.sqrt, [-1.0]), (xp.exp, [1000.0])]
)
def test_native_domain_and_overflow_fail_closed(
    operation: Any, values: Any, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="native CPU"):
        xp.compile(operation, cache=tmp_path)(np.asarray(values))
