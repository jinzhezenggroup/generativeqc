"""Run production inventory/ABI decisions without loading a native CUDA library."""

from __future__ import annotations

import ctypes
import typing
from types import SimpleNamespace

import pytest
from generativeqc import GridSpec, KsOptions, _native, resources_ks

ATOMS = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]


@pytest.fixture(autouse=True)
def no_native_library_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resource discovery in this suite uses only explicit library doubles."""

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("shape-only inventory attempted to load a native library")

    monkeypatch.setattr(_native, "load_library", forbidden)


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        grid=SimpleNamespace(radial_points=4, angular_polar=3),
        tile_points=16,
        xc_schedule="fused",
        has_nonlocal_correlation=False,
        has_nondefault_composition=False,
    )


def _item(n: int) -> dict[str, typing.Any]:
    return {
        "atoms": 3,
        "spins": 1,
        "grid_points": 64,
        "orbital": {
            "nbf": n,
            "cartesian_nbf": n,
            "shells": 4,
            "primitives": 8,
            "maximum_angular": 1,
        },
    }


@pytest.mark.parametrize(
    "backend,n", [("cpu", 17), ("cuda", 16), ("cuda", 17), ("cuda", 32)]
)
def test_host_inventory_preserves_current_model_and_solver_charge(
    backend: str, n: int
) -> None:
    row = resources_ks._item_host_inventory(
        _item(n),
        diis_history=8,
        max_iterations=100,
        requires_first_ao_derivatives=True,
        backend=backend,
        model=_model(),
    )
    expected = (1 << 20) + 16 * 8 * n * n if backend == "cuda" and n > 16 else 0
    assert row["solver_host"] == expected
    assert row["solver_binding_host"] == (16 * 1024 if backend == "cuda" else 0)
    assert row["matrix_binding_host"] == (16 * 1024 if backend == "cuda" else 0)
    assert row["structured_binding_host"] == (32 * 1024 if backend == "cuda" else 0)
    retained_fields = (
        "metadata",
        "grid",
        "basis",
        "warm_and_matrices",
        "history",
        "provider",
        "solver_host",
        "solver_binding_host",
        "matrix_binding_host",
        "structured_binding_host",
        "xc_schedule_staging",
        "nonlocal_provider",
    )
    assert row["retained"] == sum(row[key] for key in retained_fields)


@pytest.mark.parametrize("schedule", ["fused", "host_unfused"])
@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("backend,n", [("cpu", 17), ("cuda", 16), ("cuda", 17)])
def test_matrix_binding_host_envelope_is_independent_of_provider_and_xc(
    backend: str, n: int, spins: int, schedule: str
) -> None:
    model = _model()
    model.xc_schedule = schedule
    row = resources_ks._item_host_inventory(
        {**_item(n), "spins": spins},
        diis_history=8,
        max_iterations=100,
        requires_first_ao_derivatives=True,
        backend=backend,
        model=model,
    )
    assert row["matrix_binding_host"] == (16 * 1024 if backend == "cuda" else 0)
    assert row["structured_binding_host"] == (32 * 1024 if backend == "cuda" else 0)


def _library(quadrature_status: int = 0) -> SimpleNamespace:
    def query(*args: typing.Any) -> int:
        output = args[-2]
        output[:] = [8 << 20] * 3
        return 0

    def quadrature(atoms: int, points: int, output: typing.Any) -> int:
        assert (atoms, points) == (3, 64)
        output._obj.value = 320 << 20
        return quadrature_status

    def matrix_provider(nao: int, output: typing.Any) -> int:
        output._obj.value = (96 << 20) if nao >= 17 else 0
        return 0

    return SimpleNamespace(
        generativeqc_resource_ks_cuda_v1=query,
        generativeqc_resource_ks_matrix_provider_cuda_v1=matrix_provider,
        generativeqc_resource_quadrature_cuda_v1=quadrature,
    )


def test_quadrature_setup_coexists_with_prepared_coulomb() -> None:
    result = resources_ks._cuda_item_inventory(
        _library(),
        _item(32),
        diis_history=8,
        requires_first_ao_derivatives=True,
        functional_code=1,
        requires_tau=False,
        tile=16,
    )
    assert result["quadrature_setup"] == 320 << 20
    assert result["setup"] == 328 << 20


@pytest.mark.parametrize("missing", [False, True])
def test_unavailable_quadrature_inventory_fails_closed(missing: bool) -> None:
    library = _library(quadrature_status=1)
    if missing:
        del library.generativeqc_resource_quadrature_cuda_v1
    with pytest.raises(NotImplementedError, match="quadrature"):
        resources_ks._cuda_item_inventory(
            library,
            _item(32),
            diis_history=8,
            requires_first_ao_derivatives=True,
            functional_code=1,
            requires_tau=False,
            tile=16,
        )


def _abi_library(version: int | None = 1) -> SimpleNamespace:
    library = SimpleNamespace(generativeqc_ks_resource_inventory_version_v1=lambda: 1)
    if version is not None:
        library.generativeqc_ks_options_version = lambda: version
    return library


@pytest.mark.parametrize("version", [None, 0, 1, 3, 5])
def test_native_semantic_abi_version_is_checked_even_for_default_model(
    version: int | None,
) -> None:
    request = resources_ks.ks_resource_request([ATOMS], library=_abi_library(version))
    if version == 1:
        assert request.unsupported_reason is None
        assert request.candidates
    else:
        assert not request.candidates
        assert "semantic KS execution-plan ABI" in request.unsupported_reason


@pytest.mark.parametrize("method", ["pbe-rks", "pbe0-rks", "b3lyp-rks"])
def test_current_models_plan_without_retired_version_properties(method: str) -> None:
    """Resolve actual models and complete planning through the supported ABI."""
    request = resources_ks.ks_resource_request(
        [ATOMS],
        method=method,
        library=_abi_library(),
        ks_options=KsOptions(grid=GridSpec(24, 8, 16)),
    )
    assert request.unsupported_reason is None
    assert request.candidates
    assert request.identity.method == method


@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize(
    "backend,spins,requires_first_ao_derivatives",
    [("cpu", 1, True), ("cpu", 2, True), ("cpu", 1, False), ("cuda", 1, True)],
)
def test_cpu_ao_cache_boundary_keeps_exact_transient_charge(
    delta: int, backend: str, spins: int, requires_first_ao_derivatives: bool
) -> None:
    inventory = resources_ks._item_host_inventory
    cap = 64 << 20
    n = 8
    points = cap // 32 // n + delta
    item = {**_item(n), "grid_points": points, "spins": spins}
    row = inventory(
        item,
        diis_history=8,
        max_iterations=100,
        requires_first_ao_derivatives=requires_first_ao_derivatives,
        backend=backend,
        model=_model(),
    )
    expected = (
        32 * points * n
        if backend == "cpu"
        and requires_first_ao_derivatives
        and spins == 1
        and delta <= 0
        else 0
    )
    assert row["ao_grid_cache"] == expected
    matrix_work = 8 * spins * n * n * (128 + 2 * 9) + 16 * 9 * 9
    tile = (
        8 * min(points, 16) * n * (4 if requires_first_ao_derivatives else 1)
        + 8 * spins * n * n
        if backend == "cpu"
        else 0
    )
    assert row["scf_workspace"] == matrix_work + tile + expected


@pytest.mark.parametrize("symbol", [None, 1])
def test_unusable_semantic_abi_symbol_is_explicitly_unsupported(symbol: object) -> None:
    library = _abi_library()
    library.generativeqc_ks_options_version = symbol
    request = resources_ks.ks_resource_request([ATOMS], library=library)
    assert not request.candidates
    assert "semantic KS execution-plan ABI" in request.unsupported_reason


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("n", [-1, True, 1 << 32])
def test_host_inventory_rejects_invalid_or_overflowing_dimensions(
    backend: str,
    n: int,
) -> None:
    """Use the real checked arithmetic, without allocating the claimed arrays."""
    with pytest.raises(ValueError, match="dimension|byte product"):
        resources_ks._item_host_inventory(
            _item(n),
            diis_history=8,
            max_iterations=100,
            requires_first_ao_derivatives=True,
            backend=backend,
            model=_model(),
        )


def test_cuda_inventory_rejects_shape_overflow_before_native_query() -> None:
    library = _library()

    def forbidden(*args: typing.Any) -> typing.NoReturn:
        pytest.fail("overflowing shape reached the native inventory query")

    library.generativeqc_resource_ks_cuda_v1 = forbidden
    n = 1 << (8 * ctypes.sizeof(ctypes.c_size_t))
    with pytest.raises(ValueError, match="host size_t ABI"):
        resources_ks._cuda_item_inventory(
            library,
            _item(n),
            diis_history=8,
            requires_first_ao_derivatives=True,
            functional_code=1,
            requires_tau=False,
            tile=16,
        )


@pytest.mark.parametrize("method", ["pbe0-rks", "b3lyp-rks"])
def test_nondefault_cuda_composition_without_inventory_fails_closed(
    method: str,
) -> None:
    request = resources_ks.ks_resource_request(
        [ATOMS],
        method=method,
        backend="cuda",
        library=_abi_library(),
        ks_options=KsOptions(grid=GridSpec(24, 8, 16)),
    )
    assert not request.candidates
    assert "scaled/global-hybrid execution" in request.unsupported_reason
