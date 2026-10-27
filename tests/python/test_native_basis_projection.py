"""Installed basis transport: independent science, ownership and explicit compatibility."""

import ctypes
import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import _native, _projection_native, projection


@pytest.fixture
def native_projection() -> typing.Any:
    library = _projection_native.optional_library()
    if library is None:
        pytest.skip("requires installed native basis projection protocol")
    return library


def ambient(seed: int = 42) -> typing.Any:
    generator = np.random.default_rng(seed)
    source_frame = generator.normal(size=(14, 6))
    target_frame = generator.normal(size=(14, 8))
    source = source_frame.T @ source_frame
    target = target_frame.T @ target_frame
    cross = target_frame.T @ source_frame
    values, vectors = np.linalg.eigh(source)
    coefficients = (vectors / np.sqrt(values))[:, :2]
    image = np.linalg.lstsq(target_frame, source_frame @ coefficients, rcond=1e-10)[0]
    norms, rotation = np.linalg.eigh(image.T @ target @ image)
    expected = image @ (rotation / np.sqrt(norms)) @ rotation.T
    return source, target, cross, coefficients, expected


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("occupation", (1, 2))
@pytest.mark.parametrize("density_input", (False, True))
def test_independent_ambient_least_squares(
    native_projection: typing.Any, seed: int, occupation: int, density_input: bool
) -> None:
    source, target, cross, coefficients, expected = ambient(seed)
    policy = projection.ProjectionPolicy(maximum_residual=1)
    if density_input:
        result = projection.project_density(
            source,
            target,
            cross,
            occupation * coefficients @ coefficients.T,
            occupied_orbitals=2,
            occupation=occupation,
            policy=policy,
        )
    else:
        result = projection.project_occupied(
            source, target, cross, coefficients, occupation=occupation, policy=policy
        )
    np.testing.assert_allclose(
        result.density, occupation * expected @ expected.T, atol=3e-11, rtol=0
    )
    np.testing.assert_allclose(
        result.coefficients.T @ target @ result.coefficients,
        np.eye(2),
        atol=3e-11,
        rtol=0,
    )
    assert result.diagnostics.electron_trace == pytest.approx(2 * occupation, abs=3e-11)
    assert result.diagnostics.normal_equation_residual < 3e-11
    assert result.diagnostics.metric_idempotency_error < 3e-11
    with pytest.raises(ValueError):
        result.density.setflags(write=True)


def test_normal_path_never_runs_python_linear_algebra(
    native_projection: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target, cross, coefficients, expected = ambient()

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("native transport retried Python projection science")

    monkeypatch.setattr(projection, "_project_occupied_python", forbidden)
    monkeypatch.setattr(projection, "_project_density_python", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(np.linalg, "norm", forbidden)
    policy = projection.ProjectionPolicy(maximum_residual=1)
    result = projection.project_occupied(
        source, target, cross, coefficients, policy=policy
    )
    assert result.execution_owner == "native_cpu"
    np.testing.assert_allclose(
        result.density, 2 * expected @ expected.T, atol=3e-11, rtol=0
    )
    reconstructed = projection.project_density(
        source,
        target,
        cross,
        2 * coefficients @ coefficients.T,
        occupied_orbitals=2,
        policy=policy,
    )
    np.testing.assert_allclose(
        reconstructed.density, result.density, atol=3e-11, rtol=0
    )


def test_native_failure_does_not_retry_and_closes_context(
    native_projection: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailedOperation:
        closed = False

        def __getattr__(self, name: str) -> typing.Any:
            return getattr(native_projection, name)

        def generativeqc_basis_project_cpu_v1(
            self, context: typing.Any, descriptor: typing.Any, output: typing.Any
        ) -> int:
            return _native.STATUS_NUMERICAL_FAILURE

        def generativeqc_context_destroy(self, context: typing.Any) -> None:
            native_projection.generativeqc_context_destroy(context)
            self.closed = True

    failure = FailedOperation()
    monkeypatch.setattr(_projection_native, "optional_library", lambda: failure)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("native error retried Python science")

    monkeypatch.setattr(projection, "_project_occupied_python", forbidden)
    with pytest.raises(RuntimeError, match="GENERATIVEQC error"):
        projection.project_occupied(np.eye(2), np.eye(2), np.eye(2), np.eye(2))
    assert failure.closed


def test_native_capacity_exact_and_one_byte_short(
    native_projection: typing.Any,
) -> None:
    descriptor = _projection_native.Descriptor(
        struct_size=ctypes.sizeof(_projection_native.Descriptor),
        abi_version=_native.ABI_VERSION,
        source_ao=2,
        target_ao=2,
        occupied=2,
        occupation=2,
        maximum_ao=4096,
        relative_threshold=1e-10,
        validation_tolerance=1e-7,
        maximum_residual=1,
    )
    required = _projection_native.Capacity(
        struct_size=ctypes.sizeof(_projection_native.Capacity),
        abi_version=_native.ABI_VERSION,
    )
    _native.check(
        native_projection,
        native_projection.generativeqc_basis_projection_query_v1(
            ctypes.byref(descriptor), ctypes.byref(required)
        ),
    )
    arguments = (np.eye(2), np.eye(2), np.eye(2), np.eye(2))
    keywords = {
        "occupied": None,
        "occupation": 2,
        "policy": projection.ProjectionPolicy(),
    }
    result = _projection_native.project(
        *arguments, **keywords, maximum_bytes=required.numeric_workspace_bytes
    )
    np.testing.assert_allclose(result.density, 2 * np.eye(2), atol=1e-12, rtol=0)
    with pytest.raises(MemoryError, match="maximum_workspace_bytes"):
        _projection_native.project(
            *arguments, **keywords, maximum_bytes=required.numeric_workspace_bytes - 1
        )


@pytest.mark.parametrize("kind", ("custom_policy", "custom_scalar", "unbounded"))
def test_explicit_custom_and_unbounded_compatibility(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    class CustomPolicy(projection.ProjectionPolicy):
        pass

    class CustomScalar(float):
        pass

    policy = {
        "custom_policy": CustomPolicy(),
        "custom_scalar": projection.ProjectionPolicy(
            relative_threshold=CustomScalar(1e-10)
        ),
        "unbounded": projection.ProjectionPolicy(maximum_ao=2**100),
    }[kind]

    def forbidden() -> typing.NoReturn:
        raise AssertionError("custom policy lost explicit compatibility")

    monkeypatch.setattr(_projection_native, "optional_library", forbidden)
    result = projection.project_density(
        np.eye(2),
        np.eye(2),
        np.eye(2),
        2 * np.eye(2),
        occupied_orbitals=2,
        policy=policy,
    )
    np.testing.assert_array_equal(result.density, 2 * np.eye(2))
    assert result.execution_owner == "python_compatibility"


def test_missing_protocol_is_compatibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "old-library.so"
    path.touch()
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [path])
    monkeypatch.setattr(ctypes, "CDLL", lambda selected: SimpleNamespace())
    result = projection.project_occupied(np.eye(2), np.eye(2), np.eye(2), np.eye(2))
    np.testing.assert_array_equal(result.density, 2 * np.eye(2))


def test_prepared_projection_honors_native_capacity(
    native_projection: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc import Calculator

    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    source = Calculator(basis="sto-3g")
    target = Calculator(basis="def2-svp")
    maximum = 1024 * 1024
    with source.prepare_batch([atoms]) as small, target.prepare_batch([atoms]) as large:
        small.execute(strict=True)
        monkeypatch.setattr(
            _projection_native, "numeric_capacity", lambda *args: maximum + 1
        )
        with pytest.raises(MemoryError, match="maximum_host_bytes"):
            large.initialize_from(small, strict=False, maximum_host_bytes=maximum)
        result = large.execute(strict=True).items[0]
        assert result.succeeded and result.converged and result.restart_origin == "cold"


@pytest.mark.parametrize("compatibility", (False, True))
def test_prepared_projection_reports_actual_owner(
    native_projection: typing.Any, compatibility: bool
) -> None:
    from generativeqc import Calculator

    class CustomPolicy(projection.ProjectionPolicy):
        pass

    policy = CustomPolicy() if compatibility else projection.ProjectionPolicy()
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    with (
        Calculator(basis="sto-3g").prepare_batch([atoms]) as source,
        Calculator(basis="def2-svp").prepare_batch([atoms]) as target,
    ):
        source.execute(strict=True)
        report = target.initialize_from(source, policy=policy)
        expected = (
            "python_compatibility"
            if compatibility
            else (
                "native_cpu_system"
                if hasattr(native_projection, "generativeqc_hf_system_project_cpu_v1")
                else "native_cpu"
            )
        )
        assert report["items"][0]["projection_owners"] == [expected]
        assert report["target_verification"] == "pending_execute"


def test_selected_loader_error_never_tries_a_later_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first, second = tmp_path / "first.so", tmp_path / "second.so"
    first.touch()
    second.touch()
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [first, second])

    def load(selected: str) -> typing.NoReturn:
        if Path(selected) == second:
            raise AssertionError("selected loader failure tried another native library")
        raise OSError("selected library cannot load")

    monkeypatch.setattr(ctypes, "CDLL", load)
    with pytest.raises(OSError, match="cannot load"):
        projection.project_occupied(np.eye(2), np.eye(2), np.eye(2), np.eye(2))


def test_selected_abi_mismatch_is_not_compatibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library = SimpleNamespace(
        generativeqc_basis_projection_query_v1=lambda *args: 0,
        generativeqc_basis_project_cpu_v1=lambda *args: 0,
        generativeqc_get_abi_version=lambda: _native.ABI_VERSION + 1,
    )
    monkeypatch.setattr(ctypes, "CDLL", lambda selected: library)
    with pytest.raises(RuntimeError, match="ABI version mismatch"):
        _projection_native._load(str(tmp_path / "mismatched.so"))
