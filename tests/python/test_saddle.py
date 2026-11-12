"""Exercise the real native saddle ABI, not a Python optimizer substitute."""

import typing

import numpy as np
import pytest
from generativeqc import _native
from generativeqc.saddle import SaddleOptions, SaddleProgress, correct_saddle


@pytest.fixture
def native() -> None:
    try:
        library = _native.load_library()
    except (OSError, RuntimeError) as error:
        pytest.skip(f"native library unavailable: {error}")
    if not hasattr(library, "generativeqc_saddle_correct_v1"):
        pytest.skip("native library lacks saddle ABI")


def derivatives(
    curvatures: tuple[float, ...] = (-1.0, 2.0, 3.0),
) -> dict[str, typing.Any]:
    matrix = np.diag(curvatures)

    def energy_gradient(positions: np.ndarray) -> tuple[float, np.ndarray]:
        flat = positions.ravel()
        return float(0.5 * flat @ matrix @ flat), (matrix @ flat).reshape(
            positions.shape
        )

    def hessian(positions: np.ndarray) -> np.ndarray:
        return matrix.copy()

    return {"energy_gradient": energy_gradient, "hessian": hessian}


def test_native_saddle_converges_and_detaches(native: None) -> None:
    initial = np.array([[0.02, 0.02, 0.02]])
    observations = []
    result = correct_saddle(initial, **derivatives(), progress=observations.append)
    assert result.converged
    assert result.receipt["stationary_verified"]
    assert result.receipt["negative_curvatures"] == 1
    assert 0 < result.receipt["accepted_steps"] <= result.receipt["attempts"] <= 100
    np.testing.assert_allclose(result.coordinates, 0, atol=3e-4)
    np.testing.assert_array_equal(result.hessian, np.diag([-1.0, 2.0, 3.0]))
    assert result.receipt["energy_gradient_calls"] > result.receipt["accepted_steps"]
    assert result.receipt["hessian_calls"] > result.receipt["accepted_steps"]
    assert result.elapsed_seconds >= observations[-1].elapsed_seconds > 0
    assert not result.failures
    initial[:] = 99
    assert not np.any(result.coordinates == 99)
    for array in (
        result.coordinates,
        result.gradient,
        result.hessian,
        result.target_mode,
        result.curvatures,
        observations[0].coordinates,
    ):
        assert not array.flags.writeable
    with pytest.raises(TypeError):
        result.receipt["status"] = "success"


@pytest.mark.parametrize(
    "changes",
    [
        {"max_attempts": 101},
        {"max_attempts": 0},
        {"max_attempts": True},
        {"max_coordinates": 301},
        {"maximum_numeric_bytes": 0},
        {"project_rigid_motions": 1},
        {"gradient_rms": float("nan")},
        {"energy_change": 0},
        {"initial_radius": 1},
        {"minimum_radius": 0.2},
        {"minimum_mode_overlap": 1.1},
        {"acceptance_ratio": 1},
    ],
)
def test_invalid_options(changes: dict[str, typing.Any]) -> None:
    with pytest.raises(ValueError):
        SaddleOptions(**changes)


@pytest.mark.parametrize(
    "initial",
    [
        [],
        [[0, 0]],
        [[0, 0, float("nan")]],
        np.zeros((101, 3)),
        [[0j, 0, 0]],
        [[True, False, True]],
    ],
)
def test_invalid_geometry_before_callbacks(initial: typing.Any) -> None:
    def forbidden(*args: typing.Any) -> typing.NoReturn:
        pytest.fail("invalid coordinates reached derivative callback")

    with pytest.raises(ValueError):
        correct_saddle(initial, energy_gradient=forbidden, hessian=forbidden)


def test_capacity_refusal_before_callbacks(native: None) -> None:
    def forbidden(*args: typing.Any) -> typing.NoReturn:
        pytest.fail("capacity refusal reached derivative callback")

    with pytest.raises(MemoryError, match="maximum_numeric_bytes"):
        correct_saddle(
            [[0.1, 0, 0]],
            energy_gradient=forbidden,
            hessian=forbidden,
            options=SaddleOptions(maximum_numeric_bytes=1),
        )


@pytest.mark.parametrize(
    "error", [ValueError("SCF did not converge"), KeyboardInterrupt()]
)
def test_initial_callback_exception_has_no_valid_arrays(
    native: None, error: BaseException
) -> None:
    def failed(positions: np.ndarray) -> typing.NoReturn:
        raise error

    result = correct_saddle(
        [[0.1, 0, 0]], **{**derivatives(), "energy_gradient": failed}
    )
    assert not result.converged
    assert result.receipt["status"] == "evaluation_failed"
    assert not result.receipt["has_valid_state"]
    assert (
        result.gradient
        is result.hessian
        is result.target_mode
        is result.curvatures
        is None
    )
    assert result.failures == (("energy_gradient", type(error).__name__, str(error)),)
    np.testing.assert_array_equal(result.coordinates, [[0.1, 0, 0]])


def test_later_hessian_exception_retains_last_accepted_state(native: None) -> None:
    providers = derivatives()
    initial = np.array([[0.1, 0.1, 0.1]])
    hessians = []

    def hessian(positions: np.ndarray) -> np.ndarray:
        hessians.append(positions.copy())
        if len(hessians) > 1:
            raise RuntimeError("response did not converge")
        return providers["hessian"](positions)

    result = correct_saddle(initial, **{**providers, "hessian": hessian})
    assert not result.converged
    assert result.receipt["status"] == "hessian_failed"
    assert result.receipt["has_valid_state"]
    assert result.receipt["accepted_steps"] == 0
    np.testing.assert_array_equal(result.coordinates, initial)
    np.testing.assert_array_equal(result.hessian, providers["hessian"](initial))
    assert result.failures[0][0] == "hessian"


def test_unprintable_callback_exception_stays_inside_abi(native: None) -> None:
    class UnprintableError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("cannot format exception")

    def failed(positions: np.ndarray) -> typing.NoReturn:
        raise UnprintableError

    result = correct_saddle(
        [[0.1, 0, 0]], **{**derivatives(), "energy_gradient": failed}
    )
    assert not result.converged
    assert result.failures == (
        ("energy_gradient", "UnprintableError", "unprintable callback exception"),
    )
    assert result.receipt["energy"] is None


@pytest.mark.parametrize("stage", ["energy_gradient", "hessian"])
def test_invalid_derivative_output_is_contained(native: None, stage: str) -> None:
    providers = derivatives()
    providers[stage] = lambda positions: (
        (0, np.zeros(3)) if stage == "energy_gradient" else np.full((3, 3), np.nan)
    )
    result = correct_saddle([[0.1, 0, 0]], **providers)
    assert not result.converged
    assert result.failures[0][:2] == (stage, "ValueError")


def test_nonnegative_initial_mode_is_not_a_saddle(native: None) -> None:
    result = correct_saddle([[0.1, 0.1, 0.1]], **derivatives((1, 2, 3)))
    assert result.receipt["status"] == "target_not_negative"
    assert not result.receipt["stationary_verified"]
    assert result.receipt["attempts"] == 0
    assert not result.converged


def test_attempt_limit_is_nonconvergence(native: None) -> None:
    result = correct_saddle(
        [[0.1, 0.1, 0.1]],
        **derivatives(),
        options=SaddleOptions(max_attempts=1, initial_radius=0.001),
    )
    assert result.receipt["status"] == "attempt_limit"
    assert result.receipt["attempts"] == 1
    assert not result.converged


@pytest.mark.parametrize(
    "stage", ["before", "progress", "final", "cancel_error", "progress_error"]
)
def test_cancellation_cannot_publish_success(native: None, stage: str) -> None:
    def cancelled() -> bool:
        if stage == "cancel_error":
            raise RuntimeError("cancellation observer failed")
        return stage == "before"

    def progress(observation: SaddleProgress) -> bool:
        if stage == "progress_error":
            raise RuntimeError("progress observer failed")
        return stage == "progress" or (
            stage == "final" and observation.receipt["stationary_verified"]
        )

    result = correct_saddle(
        [[0.02, 0.02, 0.02]], **derivatives(), cancelled=cancelled, progress=progress
    )
    assert result.receipt["status"] == "cancelled"
    assert not result.receipt["stationary_verified"]
    assert not result.converged
    assert bool(result.failures) == stage.endswith("error")


def test_readonly_callback_geometry_cannot_corrupt_native_state(native: None) -> None:
    providers = derivatives()

    def energy_gradient(positions: np.ndarray) -> tuple[float, np.ndarray]:
        positions[:] = 99
        return providers["energy_gradient"](positions)

    result = correct_saddle(
        [[0.1, 0, 0]], **{**providers, "energy_gradient": energy_gradient}
    )
    assert result.failures[0][:2] == ("energy_gradient", "ValueError")
    np.testing.assert_array_equal(result.coordinates, [[0.1, 0, 0]])
