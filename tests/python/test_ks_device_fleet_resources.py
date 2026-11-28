"""Execute native CUDA fleet composition without claiming GPU leaf qualification."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import Primitive, ResourceBudget, Shell, _native, resources_ks
from generativeqc.ks import _native_semilocal_code, resolve_ks_options
from generativeqc.resources_planning import plan_resources


@pytest.fixture(scope="module")
def library() -> Any:
    """Require the actual compiled bridge even in a CPU-only library."""
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_resource_ks_cuda_fleet_v1")
    return loaded


def topology(cartesian_aos: int, atoms: int = 2) -> dict[str, Any]:
    return {
        "atoms": atoms,
        "orbital": {
            "cartesian_nbf": cartesian_aos,
            "shells": atoms,
            "primitives": 3 * atoms,
        },
    }


@pytest.mark.parametrize("cores", ((0, 0), (10, 0), (28, 46)))
def test_stationary_shape_reduces_per_atom_ecp_core_counts(
    cores: tuple[int, ...],
) -> None:
    """The seven-scalar native shape receives the total, never an atom tuple."""
    observed = []

    class Query:
        def __call__(
            self, descriptor: Any, shape: Any, count: int, output: Any, size: int
        ) -> int:
            observed.append(tuple(shape))
            assert count == 7 and size == 9
            return _native.STATUS_NOT_IMPLEMENTED

    item = {
        "atoms": 2,
        "grid_points": 768,
        "orbital": {
            "nbf": 7,
            "primitives": 21,
            "maximum_angular": 1,
            "ecp_terms": 3,
            "ecp_core_electrons": cores,
        },
    }
    library = SimpleNamespace(generativeqc_resource_ks_stationary_cuda_v1=Query())
    descriptor = _native.MethodDescriptor()
    assert (
        resources_ks._native_cuda_stationary_inventory(library, item, descriptor)
        is None
    )
    assert observed == [(2, 7, 21, 768, 1, 3, sum(cores))]


def leaves(item: dict[str, Any], *, ecp: bool = False) -> dict[str, int]:
    atoms = item["atoms"]
    return {
        "state": atoms * 100,
        "xc": atoms * 200,
        "coulomb": atoms * 300,
        "quadrature_setup": atoms * 400,
        "matrix_provider": atoms * 500,
        "ecp_workspace": (1 << 20) if ecp else 0,
    }


def test_fitted_fock_capacity_keeps_source_metric_in_setup(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A large source metric can dominate construction without staying live."""
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", "auto")
    query = library.generativeqc_resource_prepared_df_fock_v1
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    requested, exact, auxiliary_aos = 1 << 20, 64 << 10, 2048
    orbital_shape = (2, 2, 2, 6)
    auxiliary_shape = (auxiliary_aos,) * 4
    row = (ctypes.c_uint64 * 14)(
        2, *orbital_shape, *auxiliary_shape, requested, exact, 1, 1, 0
    )
    values = (ctypes.c_uint64 * 2)()
    assert query(row, len(row), values, len(values)) == 0
    retained, setup = values
    assert retained == requested
    assert setup >= exact + 8 * auxiliary_aos**2 > retained

    row[10] = 0
    assert query(row, len(row), values, len(values)) == 0
    assert tuple(values) == (retained, setup - exact)
    row[9] = 0
    assert query(row, len(row), values, len(values)) == 0
    assert values[0] > retained and values[1] >= setup - exact

    saved = tuple(values)
    row[11] = 3
    assert query(row, len(row), values, len(values)) != 0
    assert tuple(values) == saved
    row[11] = 1
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", "invalid")
    assert query(row, len(row), values, len(values)) != 0
    assert tuple(values) == saved


@pytest.mark.parametrize("spins", (1, 2))
def test_rebuild_state_inventory_counts_radial_exchange_without_changing_xc(
    library: Any, spins: int
) -> None:
    """The shape query adds each mutable K panel independently of its source."""
    query = library.generativeqc_resource_ks_cuda_rebuild_v1
    query.argtypes = [ctypes.c_size_t] * 10 + [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    method = "pbe0-rks" if spins == 1 else "pbe0-uks"
    functional = _native_semilocal_code(resolve_ks_options(method).method_ir)
    shape = (12, 4, 30, 768, 8, spins, functional, 256)
    values = (ctypes.c_uint64 * 2)(123, 456)
    status = query(*shape, 0, 0, values, len(values))
    if status == 2:
        assert tuple(values) == (123, 456)
        pytest.skip("native library has no CUDA state allocator")
    assert status == 0
    j_state, xc = values
    assert j_state > 0 and xc > 0
    assert query(*shape, 1, 0, values, len(values)) == 0
    hybrid_state, hybrid_xc = values
    assert hybrid_state > j_state
    assert hybrid_xc == xc
    assert query(*shape, 0, 1, values, len(values)) == 0
    range_state, range_xc = values
    assert range_state > j_state
    assert range_xc == xc
    assert query(*shape, 1, 1, values, len(values)) == 0
    assert tuple(values) == (hybrid_state + range_state - j_state, xc)
    assert query(*shape, 1, 2, values, len(values)) != 0
    assert tuple(values) == (hybrid_state + range_state - j_state, xc)


def test_fitted_rvv10_accounts_for_distinct_equal_size_grid(library: Any) -> None:
    """Different quadratures can have the same shape but own separate buffers."""
    query = library.generativeqc_resource_ks_cuda_nonlocal_v1
    query.argtypes = [ctypes.c_size_t] * 10 + [
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    shape = (12, 4, 30, 768, 8, 1, 256, 768)
    values = (ctypes.c_uint64 * 4)(123, 456, 789, 1011)
    status = query(*shape, 0, 0, 1 << 22, values, len(values))
    if status == _native.STATUS_NOT_IMPLEMENTED:
        assert tuple(values) == (123, 456, 789, 1011)
        pytest.skip("native library has no CUDA variational r2SCAN owner")
    assert status == 0
    shared_state, shared_xc, shared_quadrature, shared_host = values
    assert shared_quadrature == 0
    assert query(*shape, 1, 0, 1 << 22, values, len(values)) == 0
    state, xc, quadrature, host = values
    assert xc > shared_xc and quadrature > 0
    assert state == shared_state and host == shared_host
    assert query(*shape, 1, 0, 1, values, len(values)) == _native.STATUS_OUT_OF_MEMORY
    assert tuple(values) == (state, xc, quadrature, host)


@pytest.mark.parametrize(
    ("model", "default"), ((1, 2), (2, 0)), ids=("wb97mv", "range-pbe")
)
def test_radial_auto_envelopes_split_fit_and_exact_retry(
    library: Any, monkeypatch: pytest.MonkeyPatch, model: int, default: int
) -> None:
    """Auto quotes either transaction, and a large second metric is late scratch."""
    query = library.generativeqc_resource_ks_cuda_radial_fock_v1
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    budget = (1 << 20) + 1
    shape = (ctypes.c_uint64 * 14)(2, 2, 2, 2, 6, *([2048] * 4), budget, 1, 1, model, 0)
    values = (ctypes.c_uint64 * 5)(1, 2, 3, 4, 5)
    candidates = []
    for mode, expected in (("direct", 0), ("df", 1), ("auto", 2)):
        monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", mode)
        status = query(shape, len(shape), values, len(values))
        if status == _native.STATUS_NOT_IMPLEMENTED:
            assert tuple(values) == (1, 2, 3, 4, 5)
            pytest.skip("native library has no CUDA Fock inventory")
        assert status == 0 and values[4] == expected
        candidates.append(tuple(values[:4]))
    exact, fitted, automatic = candidates
    assert exact[0] > fitted[0] == budget
    assert fitted[2] > 0
    assert automatic == tuple(map(max, zip(exact, fitted, strict=True)))
    monkeypatch.delenv("GENERATIVEQC_DF_RANGE_EXCHANGE", raising=False)
    assert query(shape, len(shape), values, len(values)) == 0
    assert tuple(values) == (*candidates[default], default)
    monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "df")
    shape[9] = 1
    saved = tuple(values)
    assert query(shape, len(shape), values, len(values)) == _native.STATUS_OUT_OF_MEMORY
    assert tuple(values) == saved


def test_direct_nonlocal_fock_rejects_large_topology_without_publication(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An LR-free model ignores the range policy but still bounds Direct packing."""
    monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "df")
    query = library.generativeqc_resource_ks_cuda_radial_fock_v1
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    shape = (ctypes.c_uint64 * 14)(2, 2, 2, 2, 6)
    values = (ctypes.c_uint64 * 5)(1, 2, 3, 4, 5)
    status = query(shape, len(shape), values, len(values))
    if status == _native.STATUS_NOT_IMPLEMENTED:
        assert tuple(values) == (1, 2, 3, 4, 5)
        pytest.skip("native library has no CUDA Fock inventory")
    assert status == 0
    retained, initial_setup, late_setup, host, range_policy = saved = tuple(values)
    assert initial_setup >= retained > 0 and host > 0
    assert late_setup == range_policy == 0
    # Valid signed inputs, but the host shell-quartet envelope cannot fit.
    shape[1:5] = [1 << 18] * 4
    assert query(shape, len(shape), values, len(values)) != 0
    assert tuple(values) == saved


def test_radial_ecp_setup_is_late_only_for_separate_range_owner(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ECP scratch retires, but late setup cannot borrow earlier owners' space."""
    query = library.generativeqc_resource_ks_cuda_radial_fock_v1
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    values = (ctypes.c_uint64 * 5)()
    for fitted, policy in ((False, "direct"), (True, "direct"), (True, "df")):
        monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", policy)
        auxiliary = (2, 2, 2, 6) if fitted else (0, 0, 0, 0)
        shape = (ctypes.c_uint64 * 14)(2, 2, 2, 2, 6, *auxiliary, 0, 0, fitted, 1, 0)
        status = query(shape, len(shape), values, len(values))
        if status == _native.STATUS_NOT_IMPLEMENTED:
            pytest.skip("native library has no CUDA Fock inventory")
        assert status == 0
        retained, setup, late, host, mode = values
        shape[13] = 1 << 21
        assert query(shape, len(shape), values, len(values)) == 0
        assert values[0] == retained and values[4] == mode
        assert values[1] > setup and values[3] > host
        if fitted:
            assert values[2] > late
        else:
            assert late == values[2] == 0


@pytest.mark.parametrize("method", ("r2scan+rvv10-rks", "wb97m-v-rks"))
def test_cuda_nonlocal_resource_plan_rejects_force_request(method: str) -> None:
    """The Hessian SCF inventory does not certify a force consumer's lifetime."""
    atoms = [(1, (0, 0, -0.7)), (1, (0, 0, 0.7))]
    with pytest.raises(NotImplementedError, match="does not cover forces"):
        resources_ks.estimate_ks_resources(
            [atoms], backend="cuda", method=method, include_forces=True
        )


@pytest.mark.parametrize(
    "auxiliary_angular,include_forces",
    ((4, True), (5, False)),
    ids=("forces", "auxiliary-h"),
)
def test_fitted_ks_resource_plans_reject_unqualified_shape_or_property(
    auxiliary_angular: int, include_forces: bool
) -> None:
    """The f/g energy inventory neither widens forces nor implies h response."""
    atoms = [(1, (0, 0, -0.7)), (1, (0, 0, 0.7))]
    auxiliary = (
        Shell(0, 0, (Primitive(0.8, 1.0),)),
        Shell(1, auxiliary_angular, (Primitive(0.6, 1.0),)),
    )
    with pytest.raises(NotImplementedError, match="orbital/auxiliary angular domain"):
        resources_ks.ks_resource_request(
            [atoms],
            backend="cuda",
            method="pbe-rks",
            basis_representation="spherical",
            include_forces=include_forces,
            _fitted=(auxiliary, 0, 1e-10),
        )


@pytest.mark.parametrize("late_setup", (0, 8192))
def test_fleet_v2_composes_fitted_setup_and_deferred_replay(
    library: Any, late_setup: int
) -> None:
    """Setup peaks retire per item; deferred scratch overlaps all owners."""
    query = library.generativeqc_resource_ks_cuda_fleet_v2
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    first = (2, 2, 2, 6, 100, 200, 300, 400, 500, 0, 1, 0, 0, 4096, late_setup)
    second = (1, 1, 1, 3, 10, 20, 30, 40, 50, 0, 1, 0, 0, 2048, 0)
    rows = (ctypes.c_uint64 * 30)(*first, *second)
    values = (ctypes.c_uint64 * 21)()
    assert query(rows, len(rows), values, len(values)) == 0
    retained = (110, 220, 330, 550)
    setup_extra = max(4096 - 600, 2048 - 60, late_setup)
    assert tuple(values[:5]) == (*retained, setup_extra)
    assert values[7] == values[8] == sum(retained) + setup_extra
    assert values[12] == 4096 and values[18] == 2048


@pytest.mark.parametrize("fused", (False, True))
@pytest.mark.parametrize("ecp", (False, True))
@pytest.mark.parametrize("stationary", (False, True))
@pytest.mark.parametrize("cartesian_aos", (2, 6, 18, 1024))
def test_ragged_fleet_matches_independent_portable_lifetimes(
    library: Any,
    monkeypatch: pytest.MonkeyPatch,
    fused: bool,
    ecp: bool,
    stationary: bool,
    cartesian_aos: int,
) -> None:
    """Vary Cartesian expansion, ECP, XC schedule and stationary candidates."""
    items = [topology(cartesian_aos), topology(2, atoms=1)]
    monkeypatch.setattr(
        resources_ks,
        "_cuda_leaf_inventory",
        lambda _, item, **__: leaves(item, ecp=ecp),
    )
    consumers = [{"minimum": 700, "cached": 900}, {"minimum": 10, "cached": 20}]
    if not stationary:
        consumers = [None, consumers[1]]
    expected = [resources_ks._cuda_item_inventory(library, item) for item in items]
    if not fused:
        for inventory in expected:
            inventory["xc"] = 0
    if ecp and stationary:
        # The force's ECP quadrature and bridge temporaries coexist with SCF;
        # the old energy setup retires those arenas before allocating scratch.
        for item, inventory in zip(items, expected, strict=True):
            shape = item["orbital"]
            n, a, s, p = (
                shape["cartesian_nbf"],
                item["atoms"],
                shape["shells"],
                shape["primitives"],
            )
            bridge = 64 * (1 + a + s + p + n + s * (s + 1) // 2 + n * n) + 48 * a
            retained = sum(inventory[key] for key in ("state", "xc", "coulomb"))
            inventory["setup"] = max(inventory["setup"], retained + (1 << 20) + bridge)
    actual, aggregate = resources_ks._native_cuda_fleet_inventory(
        library, items, consumers, device_fused_xc=fused
    )
    assert actual == expected
    for key in ("state", "xc", "coulomb", "matrix_provider"):
        assert aggregate[key] == sum(inventory[key] for inventory in expected)
    assert aggregate["setup_extra"] == max(
        max(
            0,
            inventory["setup"]
            - sum(inventory[key] for key in ("state", "xc", "coulomb")),
        )
        for inventory in expected
    )
    common = sum(
        aggregate[key]
        for key in ("state", "xc", "coulomb", "matrix_provider", "setup_extra")
    )
    for mode in ("minimum", "cached"):
        assert aggregate[mode] == (
            sum(consumer[mode] for consumer in consumers) if stationary else 0
        )
        assert aggregate[f"{mode}_peak"] == common + aggregate[mode]


@pytest.mark.parametrize("minimum,cached", ((700, 900), (900, 700), (0, 0)))
def test_cache_candidates_remain_independent(
    library: Any, monkeypatch: pytest.MonkeyPatch, minimum: int, cached: int
) -> None:
    """Composition must not invent a cache ordering or scientific policy."""
    monkeypatch.setattr(
        resources_ks, "_cuda_leaf_inventory", lambda _, item, **__: leaves(item)
    )
    _, aggregate = resources_ks._native_cuda_fleet_inventory(
        library,
        [topology(2)],
        [{"minimum": minimum, "cached": cached}],
        device_fused_xc=True,
    )
    assert aggregate["minimum"] == minimum
    assert aggregate["cached"] == cached
    assert aggregate["minimum_peak"] - minimum == aggregate["cached_peak"] - cached


def test_missing_symbol_does_not_query_leaves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing optional composition is independent from execution support."""

    def forbidden(*args: Any, **options: Any) -> Any:
        raise AssertionError("missing bridge unexpectedly requested CUDA leaves")

    monkeypatch.setattr(resources_ks, "_cuda_leaf_inventory", forbidden)
    assert (
        resources_ks._native_cuda_fleet_inventory(
            SimpleNamespace(), [topology(2)], [None], device_fused_xc=True
        )
        is None
    )


@pytest.mark.parametrize(
    "corruption",
    (
        "refusal",
        "overflow",
        "leaf",
        "peak",
        "setup",
        "setup_aggregate",
        "minimum",
        "cached",
        "swapped_candidates",
    ),
)
def test_native_refusal_or_corruption_never_retries_portable_composition(
    library: Any, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    """Exercise the real ABI first, then corrupt its publication at the boundary."""
    query = library.generativeqc_resource_ks_cuda_fleet_v1
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int

    def corrupted(shape: Any, count: int, output: Any, size: int) -> int:
        assert query(shape, count, output, size) == 0
        if corruption == "refusal":
            return -1
        if corruption == "swapped_candidates":
            output[5], output[6] = output[6], output[5]
        elif corruption in ("setup_aggregate", "minimum", "cached"):
            slot = {"setup_aggregate": 4, "minimum": 5, "cached": 6}[corruption]
            output[slot] = 0
        else:
            slot = {"overflow": 0, "leaf": 9, "peak": 7, "setup": 12}[corruption]
            output[slot] = 2**64 - 1 if corruption == "overflow" else 0
        if corruption in ("setup_aggregate", "minimum", "cached", "swapped_candidates"):
            # Keep both internal peak equations coherent while contradicting
            # returned setup diagnostics or the supplied stationary leaves.
            output[7] = sum(output[:5]) + output[5]
            output[8] = sum(output[:5]) + output[6]
        return 0

    monkeypatch.setattr(
        resources_ks, "_cuda_leaf_inventory", lambda _, item, **__: leaves(item)
    )

    def forbidden(*args: Any, **options: Any) -> Any:
        raise AssertionError("selected native failure retried portable composition")

    monkeypatch.setattr(resources_ks, "_cuda_item_inventory", forbidden)
    with pytest.raises(ValueError):
        resources_ks._native_cuda_fleet_inventory(
            SimpleNamespace(generativeqc_resource_ks_cuda_fleet_v1=corrupted),
            [topology(2)],
            [{"minimum": 700, "cached": 900}],
            device_fused_xc=True,
        )


@pytest.mark.parametrize("mode", ("minimum", "cached"))
def test_stationary_candidates_admit_exact_signed_int64_peak(
    library: Any, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Validate the wire relations without imposing ordering or losing one byte."""
    monkeypatch.setattr(
        resources_ks, "_cuda_leaf_inventory", lambda _, item, **__: leaves(item)
    )
    _, empty = resources_ks._native_cuda_fleet_inventory(
        library,
        [topology(2)],
        [{"minimum": 0, "cached": 0}],
        device_fused_xc=True,
    )
    maximum = 2**63 - 1
    stationary = {"minimum": 0, "cached": 0}
    stationary[mode] = maximum - empty[f"{mode}_peak"]
    _, exact = resources_ks._native_cuda_fleet_inventory(
        library, [topology(2)], [stationary], device_fused_xc=True
    )
    assert exact[f"{mode}_peak"] == maximum
    stationary[mode] += 1
    with pytest.raises(ValueError):
        resources_ks._native_cuda_fleet_inventory(
            library, [topology(2)], [stationary], device_fused_xc=True
        )


def test_public_candidate_and_exact_budget_preserve_legacy_bounds(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only leaf metadata is doubled; native fleet and public admission execute."""
    monkeypatch.setattr(resources_ks, "_cuda_library_identity", lambda _: {})
    monkeypatch.setattr(
        resources_ks, "_cuda_leaf_inventory", lambda _, item, **__: leaves(item)
    )
    monkeypatch.setattr(
        resources_ks,
        "_native_cuda_stationary_inventory",
        lambda _, item, descriptor: {
            "minimum": 700 * item["atoms"],
            "cached": 900 * item["atoms"],
        },
    )
    common = {
        "generativeqc_ks_options_version": lambda: 1,
        "generativeqc_resource_compose_v1": library.generativeqc_resource_compose_v1,
        "generativeqc_ks_resource_inventory_version_v1": lambda: 1,
    }
    native_library = SimpleNamespace(
        **common,
        generativeqc_resource_ks_cuda_fleet_v1=library.generativeqc_resource_ks_cuda_fleet_v1,
    )
    portable_library = SimpleNamespace(**common)
    systems = [[(1, (0, 0, -0.7)), (1, (0, 0, 0.7))], [(2, (0, 0, 0))]]
    native_request = resources_ks.ks_resource_request(
        systems, backend="cuda", library=native_library
    )
    portable_request = resources_ks.ks_resource_request(
        systems, backend="cuda", library=portable_library
    )
    assert native_request.unsupported_reason is None
    assert portable_request.unsupported_reason is None
    assert len(native_request.candidates) == len(portable_request.candidates) == 2
    for native, portable in zip(
        native_request.candidates, portable_request.candidates, strict=True
    ):
        assert native.estimates == portable.estimates
        native_decisions = dict(native.decisions)
        portable_decisions = dict(portable.decisions)
        assert (
            native_decisions.pop("cuda_fleet_composition") == "native-declared-capacity"
        )
        assert (
            portable_decisions.pop("cuda_fleet_composition")
            == "legacy-leaf-compatibility"
        )
        assert native_decisions == portable_decisions
    request = replace(native_request, candidates=(native_request.candidates[0],))
    portable = replace(portable_request, candidates=(portable_request.candidates[0],))
    plan = plan_resources(
        (request,), ResourceBudget(), library=native_library
    ).require_feasible()
    reference = plan_resources(
        (portable,), ResourceBudget(), library=portable_library
    ).require_feasible()
    assert plan.peak_bytes == reference.peak_bytes
    exact = ResourceBudget(
        host_bytes=plan.peak_bytes["host"], device_bytes=plan.peak_bytes["device"]
    )
    plan_resources((request,), exact, library=native_library).require_feasible()
    short = replace(exact, device_bytes=exact.device_bytes - 1)
    with pytest.raises(MemoryError):
        plan_resources((request,), short, library=native_library).require_feasible()
