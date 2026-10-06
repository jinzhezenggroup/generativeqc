"""Small real-endpoint CodSpeed suite for routine CPU performance regression."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

import pytest
from generativeqc import Calculator, GridSpec, KsOptions

if TYPE_CHECKING:
    from collections.abc import Callable

_T = TypeVar("_T")
Atom = tuple[str, tuple[float, float, float]]


class _BenchmarkFixture(Protocol):
    def __call__(self, target: Callable[[], _T]) -> _T: ...


@dataclass(frozen=True)
class _Case:
    name: str
    method: str
    basis: str
    atoms: tuple[Atom, ...]
    pr_fast: bool = False
    pr_extra: str | None = None
    grid_shape: tuple[int, int, int] | None = None


_FORMALDEHYDE = (
    ("C", (0.0, 0.0, 0.0)),
    ("O", (0.0, 0.0, 2.28)),
    ("H", (1.75, 0.0, -1.05)),
    ("H", (-1.75, 0.0, -1.05)),
)
_WATER = (
    ("O", (0.0, 0.0, 0.0)),
    ("H", (1.43233673, 0.0, 1.10715266)),
    ("H", (-1.43233673, 0.0, 1.10715266)),
)
_CASES = (
    # Keep the always-on PR energy tier intentionally small while exercising
    # complete public RHF and semilocal DFT endpoints.
    _Case("water-rhf-sto3g", "rhf", "sto-3g", _WATER, pr_fast=True),
    _Case("water-pbe-sto3g", "pbe-rks", "sto-3g", _WATER, pr_fast=True),
    # WB97M-V stays targeted on PRs. A fixed small grid makes it a path sentinel,
    # not a production-grid performance claim.
    _Case(
        "water-wb97mv-smallgrid-sto3g",
        "wb97m-v-rks",
        "sto-3g",
        _WATER,
        pr_extra="wb97mv",
        grid_shape=(12, 4, 8),
    ),
    # The full scheduled/manual tier needs at least one case large enough to
    # expose shell scheduling, ERI reuse and cache-locality regressions that a
    # water/STO-3G endpoint cannot represent.
    _Case("formaldehyde-rhf-def2-svp", "rhf", "def2-svp", _FORMALDEHYDE),
)


def _active_cases() -> tuple[_Case, ...]:
    tier = os.environ.get("GENERATIVEQC_CODSPEED_TIER", "full")
    if tier == "full":
        return _CASES
    if tier == "pr":
        extras = frozenset(
            item.strip()
            for item in os.environ.get("GENERATIVEQC_CODSPEED_EXTRA_CASES", "").split(
                ","
            )
            if item.strip()
        )
        supported = frozenset(
            case.pr_extra for case in _CASES if case.pr_extra is not None
        )
        unknown = extras - supported
        if unknown:
            raise RuntimeError(
                "unknown GENERATIVEQC_CODSPEED_EXTRA_CASES=" + ",".join(sorted(unknown))
            )
        return tuple(case for case in _CASES if case.pr_fast or case.pr_extra in extras)
    raise RuntimeError(f"unknown GENERATIVEQC_CODSPEED_TIER={tier!r}")


def _calculator(case: _Case) -> Calculator:
    if case.grid_shape is None:
        return Calculator(method=case.method, basis=case.basis, device="cpu")

    radial, polar, azimuth = case.grid_shape
    return Calculator(
        method=case.method,
        basis=case.basis,
        device="cpu",
        ks_options=KsOptions(
            grid=GridSpec(
                radial_points=radial,
                angular_polar=polar,
                angular_azimuth=azimuth,
            )
        ),
        energy_tolerance=1e-10,
        density_tolerance=1e-8,
        max_iterations=160,
    )


def _validate_result(
    result: object,
    case_name: str,
    *,
    require_forces: bool = False,
) -> float:
    if not result.converged:
        raise RuntimeError(f"{case_name} did not converge")
    energy = float(result.energy)
    if not math.isfinite(energy):
        raise RuntimeError(f"{case_name} returned a non-finite energy")
    if require_forces:
        forces = result.forces
        if forces is None:
            raise RuntimeError(f"{case_name} did not return forces")
        if any(
            not math.isfinite(float(component)) for row in forces for component in row
        ):
            raise RuntimeError(f"{case_name} returned non-finite forces")
    return energy


def _shift_last_atom(
    atoms: tuple[Atom, ...],
    delta: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    coordinates = [list(position) for _, position in atoms]
    for axis, value in enumerate(delta):
        coordinates[-1][axis] += value
    return tuple(tuple(float(value) for value in row) for row in coordinates)


@pytest.mark.parametrize("case", _active_cases(), ids=lambda case: case.name)
def test_cpu_warm_endpoint_walltime(
    benchmark: _BenchmarkFixture,
    case: _Case,
) -> None:
    """Track complete warm energy endpoints on one-thread hosted CPU."""
    calculator = _calculator(case)

    warmup = calculator.singlepoint(case.atoms, properties=("energy",))
    _validate_result(warmup, case.name)

    def run_once() -> float:
        result = calculator.singlepoint(case.atoms, properties=("energy",))
        return _validate_result(result, case.name)

    energy = benchmark(run_once)
    assert math.isfinite(energy)


def test_cpu_pbe_force_walltime(benchmark: _BenchmarkFixture) -> None:
    """Track the public semilocal CPU analytic-force endpoint."""
    calculator = Calculator(method="pbe-rks", basis="sto-3g", device="cpu")
    case_name = "water-pbe-force-sto3g"

    warmup = calculator.singlepoint(_WATER, properties=("energy", "forces"))
    _validate_result(warmup, case_name, require_forces=True)

    def run_once() -> float:
        result = calculator.singlepoint(_WATER, properties=("energy", "forces"))
        return _validate_result(result, case_name, require_forces=True)

    energy = benchmark(run_once)
    assert math.isfinite(energy)


def test_cpu_rhf_changed_geometry_pair_walltime(
    benchmark: _BenchmarkFixture,
) -> None:
    """Measure repeated geometry rebuilds without folding preparation into timing."""
    calculator = Calculator(method="rhf", basis="sto-3g", device="cpu")
    case_name = "water-rhf-changed-geometry-pair-sto3g"
    plus = _shift_last_atom(_WATER, (0.0010, -0.0005, 0.0003))
    minus = _shift_last_atom(_WATER, (-0.0010, 0.0005, -0.0003))

    # A prepared batch retains the production warm state used by geometry
    # optimization/MD-style workloads. Each benchmark sample performs two
    # geometry changes so repeated fixture invocations cannot silently collapse
    # into a same-geometry warm replay.
    with calculator.prepare_batch([_WATER], warm_start=True) as batch:
        warmup = batch.execute(strict=True, properties=("energy",)).items[0]
        _validate_result(warmup, case_name)

        def run_once() -> float:
            first = batch.execute(
                coordinates=[plus],
                strict=True,
                properties=("energy",),
            ).items[0]
            _validate_result(first, case_name)
            second = batch.execute(
                coordinates=[minus],
                strict=True,
                properties=("energy",),
            ).items[0]
            return _validate_result(second, case_name)

        energy = benchmark(run_once)
    assert math.isfinite(energy)
