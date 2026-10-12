"""Device-free validation for the complete public CPU Hessian benchmark driver."""

import sys
import typing
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tools import benchmark_cpu_hessian as benchmark


def test_cpu_hessian_receipt_preserves_every_timing_sample() -> None:
    values = [1.3, 7.2, 0.8]
    result = benchmark._summary(values)
    assert result["seconds"] == values
    assert result["n"] == 3
    assert result["median_seconds"] == 1.3
    assert result["max_seconds"] == 7.2
    with pytest.raises(ValueError, match="nonnegative"):
        benchmark._summary([1.0, float("nan")])
    with pytest.raises(ValueError, match="nonempty"):
        benchmark._summary([])


def test_cpu_hessian_moved_geometry_does_not_change_source() -> None:
    atoms, _, direction = benchmark._fixture("h2")
    original = tuple(atoms)
    moved = benchmark._moved(atoms, direction, 0.02)
    assert tuple(atoms) == original
    assert moved[0][0] == atoms[0][0]
    np.testing.assert_allclose(
        np.asarray([row[1] for row in moved]) - np.asarray([row[1] for row in atoms]),
        0.02 * direction,
        atol=1e-15,
    )
    with pytest.raises(ValueError, match="displacement"):
        benchmark._moved(atoms, direction, 0.0)


def test_cpu_hessian_raw_matrix_is_not_symmetrized() -> None:
    direction = np.asarray([[0.25, -0.5, 0.75]])
    # Intentionally asymmetric to confirm that the driver records RAW symmetry.
    matrix = np.asarray(
        [
            [1.0, 0.4, 0.0],
            [-0.2, 2.0, 0.1],
            [0.0, 0.1, 3.0],
        ]
    )
    vector = matrix @ direction.ravel()
    hvp = SimpleNamespace(value=vector.reshape(1, 3))
    hessian = SimpleNamespace(matrix=matrix)
    result = benchmark._validate_pair(hvp, hessian, direction)
    assert result["hvp_full_max_abs_error"] < 1e-15
    assert result["raw_symmetry_max_abs_error"] == pytest.approx(0.6)
    assert result["posthoc_symmetrization"] is False
    with pytest.raises(ArithmeticError, match="disagree"):
        benchmark._validate_pair(
            SimpleNamespace(value=np.ones_like(direction)), hessian, direction
        )


def _mock_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure: tuple[str, str] | None = None,
    supported: bool = True,
) -> tuple[SimpleNamespace, list, dict]:
    """Install device-free public endpoints with geometry-dependent outputs."""
    calls = []
    configuration = {}
    elapsed = [0.0]
    durations = iter([1.0, 2.0, 17.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 9.0])
    counts = {"hvp": 0, "full_hessian": 0}

    class Calculator:
        def __init__(self, **kwargs: typing.Any) -> None:
            configuration.update(kwargs)
            self.capabilities = SimpleNamespace(supported_second_order=supported)

        def result(
            self,
            endpoint: str,
            atoms: typing.Any,
            kwargs: dict[str, typing.Any],
            direction: np.ndarray | None = None,
        ) -> SimpleNamespace:
            counts[endpoint] += 1
            calls.append((endpoint, tuple(atoms), kwargs))
            elapsed[0] += next(durations)
            xyz = np.asarray([atom[1] for atom in atoms])
            matrix = np.eye(xyz.size) * (2.0 + float(np.sum(xyz**2)))
            value = (
                matrix
                if direction is None
                else (matrix @ direction.ravel()).reshape(xyz.shape)
            )
            if failure is not None and endpoint == failure[0] and counts[endpoint] == 2:
                if failure[1] == "raise":
                    raise RuntimeError("injected endpoint failure")
                if failure[1] == "nan":
                    value = np.full_like(value, np.nan)
                elif failure[1] == "shape":
                    value = value.reshape(-1)
                elif failure[1] == "wrong":
                    value = value + 1.0
            return (
                SimpleNamespace(matrix=value)
                if direction is None
                else SimpleNamespace(value=value)
            )

        def hessian_vector_product(
            self, atoms: typing.Any, direction: np.ndarray, **kwargs: typing.Any
        ) -> SimpleNamespace:
            return self.result("hvp", atoms, kwargs, direction)

        def hessian(self, atoms: typing.Any, **kwargs: typing.Any) -> SimpleNamespace:
            return self.result("full_hessian", atoms, kwargs)

    module = ModuleType("generativeqc")
    module.Calculator = Calculator
    module.GridSpec = SimpleNamespace
    module.KsOptions = SimpleNamespace
    monkeypatch.setitem(sys.modules, "generativeqc", module)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: elapsed[0])
    monkeypatch.setattr(benchmark, "_source_sha", lambda: "mock-source-sha")
    args = SimpleNamespace(
        case="h2",
        method="pbe-rks",
        repeats=3,
        displacement=0.02,
        radial_points=10,
        angular_polar=4,
        angular_azimuth=8,
        cache=Path("unused-mock-cache"),
        integral_budget_bytes=12345,
        output_budget_bytes=67890,
        strategy="recycled",
    )
    return args, calls, configuration


def test_cpu_hessian_run_validates_every_retained_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, calls, configuration = _mock_run(monkeypatch)
    receipt = benchmark.run(args)
    assert receipt["timings"]["hvp"]["same_geometry"]["seconds"] == [17.0, 3.0, 5.0]
    assert receipt["timings"]["full_hessian"]["same_geometry"]["seconds"] == [
        4.0,
        6.0,
        8.0,
    ]
    assert len(receipt["validation"]["same_geometry"]) == 3
    assert receipt["measurement_order"]["endpoint_first_prior_public_calls"] == {
        "hvp": 0,
        "full_hessian": 1,
    }
    assert receipt["measurement_order"]["compiler_cold_claim"] is False
    assert all("process_first" not in timing for timing in receipt["timings"].values())
    original = tuple(receipt["configuration"]["atoms_bohr"])
    moved = tuple(receipt["configuration"]["changed_atoms_bohr"])
    assert original != moved
    for endpoint, geometry, options in calls:
        assert geometry in (original, moved)
        assert options["integral_budget_bytes"] == args.integral_budget_bytes
        assert options["cache"] == args.cache
        if endpoint == "full_hessian":
            assert options["output_budget_bytes"] == args.output_budget_bytes
            assert options["strategy"] == args.strategy
        else:
            assert "output_budget_bytes" not in options
    assert calls[-2][1] == calls[-1][1] == moved
    assert configuration["device"] == "cpu"
    assert configuration["precision"] == "fp64"
    assert configuration["basis"] == "sto-3g"
    assert configuration["max_iterations"] == 200
    assert configuration["energy_tolerance"] == 1e-13
    assert configuration["density_tolerance"] == 1e-11


@pytest.mark.parametrize("endpoint", ["hvp", "full_hessian"])
@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("nan", FloatingPointError),
        ("shape", ValueError),
        ("wrong", ArithmeticError),
        ("raise", RuntimeError),
    ],
)
def test_cpu_hessian_run_rejects_bad_warm_repeat(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, fault: str, error: type[Exception]
) -> None:
    args, _, _ = _mock_run(monkeypatch, failure=(endpoint, fault))
    with pytest.raises(error):
        benchmark.run(args)


def test_cpu_hessian_run_rejects_unsupported_before_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, calls, _ = _mock_run(monkeypatch, supported=False)
    with pytest.raises(NotImplementedError, match="not qualified"):
        benchmark.run(args)
    assert calls == []


@pytest.mark.parametrize("target", ["hvp", "hessian", "direction"])
def test_cpu_hessian_pair_rejects_complex_before_conversion(target: str) -> None:
    direction = np.ones((1, 3))
    vector = direction.copy()
    matrix = np.eye(3)
    if target == "hvp":
        vector = vector.astype(complex) + 1j
    elif target == "hessian":
        matrix = matrix.astype(complex) + 1j
    else:
        direction = direction.astype(complex) + 1j
    with pytest.raises(ValueError, match="real"):
        benchmark._validate_pair(
            SimpleNamespace(value=vector), SimpleNamespace(matrix=matrix), direction
        )


def test_cpu_hessian_source_sha_uses_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}

    def git(command: list[str], **kwargs: typing.Any) -> str:
        observed.update(kwargs)
        assert command == ["git", "rev-parse", "HEAD"]
        return "source-head\n"

    monkeypatch.setattr(benchmark.subprocess, "check_output", git)
    assert benchmark._source_sha() == "source-head"
    assert observed["cwd"] == Path(benchmark.__file__).resolve().parents[1]
