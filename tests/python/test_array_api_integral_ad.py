"""Captured end-to-end AD, including nonlinear coordinate/basis transformations."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import Calculator, _native
from generativeqc.experimental import array_api as xp
from generativeqc.experimental.integral_autodiff import IntegralRHF

if TYPE_CHECKING:
    from pathlib import Path

XYZ = np.array([[0.1, 0.0, -0.7], [-0.1, 0.0, 0.7]])


@pytest.fixture(scope="module")
def owner(tmp_path_factory: pytest.TempPathFactory) -> IntegralRHF:
    # Source-only Array API conformance does not build the molecular library.
    # Native CI explicitly configures its path and must never skip a missing build.
    if not any(candidate.exists() for candidate in _native._candidate_paths()):
        if os.environ.get("GENERATIVEQC_LIBRARY"):
            pytest.fail("configured GENERATIVEQC_LIBRARY does not exist")
        pytest.skip("integral RHF acceptance requires a built/installed library")
    result = IntegralRHF(
        ("H", "H"), cache=tmp_path_factory.mktemp("capture-integral-ad")
    )
    yield result
    result.close()


def test_composed_coordinate_and_basis_chain_rule(
    owner: IntegralRHF, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = xp.MolecularModel(("H", "H"), method="rhf", differentiation="integral-ad")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("integral AD invoked Calculator's molecular endpoint")

    monkeypatch.setattr(Calculator, "singlepoint", forbidden)
    monkeypatch.setattr(Calculator, "batch_singlepoint", forbidden)

    def loss(r: Any, log_alpha: Any, coefficients: Any) -> Any:
        value = model(1.03 * r, exponents=xp.exp(log_alpha), coefficients=coefficients)
        return xp.square(value) + 0.2 * xp.sum(r * r)

    compiled = xp.value_and_grad(loss, argnums=(0, 1, 2), cache=tmp_path)
    try:
        value, gradients = compiled(XYZ, np.log(owner.exponents), owner.coefficients)
        energy, expected = owner.value_and_grad(1.03 * XYZ)
        np.testing.assert_allclose(
            value, energy * energy + 0.2 * np.sum(XYZ * XYZ), atol=2e-10
        )
        np.testing.assert_allclose(
            gradients[0],
            2 * energy * 1.03 * expected.coordinates + 0.4 * XYZ,
            atol=2e-9,
        )
        np.testing.assert_allclose(
            gradients[1], 2 * energy * expected.exponents * owner.exponents, atol=2e-9
        )
        np.testing.assert_allclose(
            gradients[2], 2 * energy * expected.coefficients, atol=2e-9
        )
        record = compiled.compilation_info()["molecular_calls"][0]
        assert record["force_calls"] == 0
        assert (
            record["integral_ad"]["derivative"]
            == "stationary-rhf-through-integral-ad-v1"
        )
    finally:
        compiled.close()


def test_integral_source_jvp_and_snapshot_vjp(
    owner: IntegralRHF, tmp_path: Path
) -> None:
    model = xp.MolecularModel(("H", "H"), method="rhf", differentiation="integral-ad")

    def energy(r: Any, alpha: Any, c: Any) -> Any:
        return model(r, exponents=alpha, coefficients=c)

    primals = (XYZ.copy(), owner.exponents.copy(), owner.coefficients.copy())
    rng = np.random.default_rng(722)
    tangents = tuple(rng.normal(size=value.shape) for value in primals)
    reference_energy, derivative = owner.value_and_grad(
        *primals[:1], exponents=primals[1], coefficients=primals[2]
    )
    expected = (derivative.coordinates, derivative.exponents, derivative.coefficients)
    value, tangent = xp.jvp(energy, primals, tangents, cache=tmp_path)
    np.testing.assert_allclose(value, reference_energy, atol=2e-10)
    np.testing.assert_allclose(
        tangent,
        sum(np.sum(g * d) for g, d in zip(expected, tangents, strict=True)),
        atol=2e-9,
    )
    _, pullback = xp.vjp(energy, *primals, cache=tmp_path)
    try:
        for array in primals:
            array[:] = 123  # replay must use the actual forward basis/geometry
        actual = pullback(xp.asarray(-2.5))
        for gradient, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(gradient, -2.5 * reference, atol=2e-9)
    finally:
        pullback.close()


def test_compiled_basis_updates_do_not_reuse_a_stale_state(
    owner: IntegralRHF, tmp_path: Path
) -> None:
    model = xp.MolecularModel(("H", "H"), method="rhf", differentiation="integral-ad")
    compiled = xp.compile(lambda r, alpha: model(r, exponents=alpha), cache=tmp_path)
    try:
        values = []
        for scale in (1.0, 1.2, 0.9):
            values.append(float(compiled(XYZ, scale * owner.exponents)))
            assert values[-1] == pytest.approx(
                owner.energy(XYZ, exponents=scale * owner.exponents), abs=2e-10
            )
        assert abs(values[0] - values[1]) > 1e-4
    finally:
        compiled.close()


def test_lowering_does_not_construct_native_integral_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("capture evaluated a molecular source")

    monkeypatch.setattr(xp.MolecularModel, "_calculator", forbidden)
    model = xp.MolecularModel(("H", "H"), method="rhf", differentiation="integral-ad")
    compiled = xp.value_and_grad(
        lambda r, alpha: model(r, exponents=alpha), argnums=(0, 1), cache=tmp_path
    )
    try:
        program = compiled.lower(XYZ, np.ones(6))
        assert program.provenance["molecular_sources"]
    finally:
        compiled.close()


@pytest.mark.parametrize(
    "options", [{"method": "uhf"}, {"backend": "cuda"}, {"density_fitting": "cpu"}]
)
def test_unqualified_integral_ad_domains_fail_at_capture(
    options: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="CPU direct RHF"):
        xp.MolecularModel(
            ("H", "H"), **{"method": "rhf", "differentiation": "integral-ad", **options}
        )


def test_basis_parameters_do_not_silently_enter_native_force_cut() -> None:
    model = xp.MolecularModel(("H", "H"), method="rhf")
    with pytest.raises(ValueError, match="continuous basis"):
        model(XYZ, exponents=np.ones(6))


def test_integral_ad_controls_are_explicit() -> None:
    model = xp.MolecularModel(
        ("H", "H"),
        method=" RHF ",
        density_fitting="NONE",
        differentiation="integral-ad",
        diis_history=12,
        screening_tolerance=0,
    )
    assert dict(model._controls)["screening_tolerance"] == 0
    assert dict(model._controls)["diis_history"] == 12
    for options in ({"diis_history": 33}, {"screening_tolerance": 1e-5}):
        with pytest.raises(ValueError, match="integral-ad"):
            xp.MolecularModel(
                ("H", "H"), method="rhf", differentiation="integral-ad", **options
            )


def test_nested_derivative_rejects_before_native_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("higher-order capture executed a molecular source")

    monkeypatch.setattr(xp.MolecularModel, "_calculator", forbidden)
    model = xp.MolecularModel(("H", "H"), method="rhf", differentiation="integral-ad")
    first = xp.grad(lambda r: model(r), cache=tmp_path)
    second = xp.grad(lambda r: xp.sum(first(r)), cache=tmp_path)
    try:
        with pytest.raises(TypeError, match="concrete arrays"):
            second(XYZ)
    finally:
        second.close()
        first.close()
