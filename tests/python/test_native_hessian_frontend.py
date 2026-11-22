"""Host-only admission/dispatch checks; fake native results are not GPU evidence."""

from __future__ import annotations

import ctypes as ct
import importlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NoReturn, Self

import numpy as np
import pytest
from generativeqc import (
    Atom,
    Calculator,
    GridSpec,
    KsOptions,
    Primitive,
    Shell,
    _native,
)
from generativeqc._native_hessian import (
    _Options,
    _options,
    _Receipt,
    checked_direction,
    has_cuda_r2scan_stationary_domain,
    linked_cuda_split_hybrid_domain,
    run_native_cuda,
)
from generativeqc.ks_diagnostics import (
    CUDA_M062X_STATIONARY_DOMAIN_VERSION,
    CUDA_MN15_STATIONARY_DOMAIN_VERSION,
    CUDA_R2SCAN_STATIONARY_DOMAIN_VERSION,
    CUDA_SPLIT_HYBRID_STATIONARY_DOMAINS,
)
from generativeqc.response_solver import GMRESOptions

from tools.generativeqc_validation.native_registry_stub import (
    install_registry_stub,
    install_scf_defaults_stub,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class NativeFunction:
    def __init__(self, callback: Callable[..., Any]) -> None:
        self.callback = callback

    def __call__(self, *arguments: Any) -> Any:
        return self.callback(*arguments)


@pytest.fixture
def frontend() -> tuple[Calculator, list[str]]:
    events = []

    class Batch:
        _batch = ct.c_void_p(123)
        _context = ct.c_void_p(456)
        _model_signature = "native-frontend-model-fixture"
        _native_dispersion = None

        def __enter__(self) -> Self:
            events.append("enter")
            return self

        def execute(
            self, *, strict: bool, properties: tuple[str, ...]
        ) -> SimpleNamespace:
            assert strict and properties == ("energy",)
            events.append("scf")
            return SimpleNamespace(
                items=(
                    SimpleNamespace(
                        ks_diagnostic=SimpleNamespace(scf_domain="incumbent")
                    ),
                )
            )

        def __exit__(self, *_: object) -> None:
            events.append("close")

    def native(
        batch: Any,
        index: int,
        options: Any,
        direction: Any,
        count: int,
        output: Any,
        size: int,
        receipt: Any,
    ) -> int:
        assert batch.value == 123 and index == 0
        selected = ct.cast(options, ct.POINTER(_Options)).contents
        assert selected.struct_size == ct.sizeof(_Options)
        assert selected.response_bytes == 64 << 20
        assert count == (6 if direction else 0)
        np.ctypeslib.as_array(output, shape=(size,))[:] = 0.125
        result = ct.cast(receipt, ct.POINTER(_Receipt)).contents
        result.completed_directions = 1 if direction else 6
        result.response_rhs_norm = 1.0
        result.response_residual_norm = 1e-13
        if selected.response_fp32_proposal:
            assert not direction
            result.cpks_operator_bytes = 36
            result.cpks_proposal_seeds = 2
        events.append("native")
        return 0

    def prepare(*arguments: Any, **keywords: Any) -> Batch:
        assert len(arguments[0]) == 1
        assert keywords == {"charges": [0], "multiplicities": [1], "warm_start": False}
        events.append("prepare")
        return Batch()

    calculator = object.__new__(Calculator)
    calculator._composite_rks_second_order = True
    calculator._cuda_rks_second_order = True
    calculator._d3_second_order_spec = None
    calculator._dispersion_method_ir = None
    calculator._device_name = "cuda"
    calculator._checked_rks_second_order_atoms = lambda atoms, **_: tuple(atoms)
    calculator.prepare_batch = prepare
    calculator._library = SimpleNamespace(
        generativeqc_ks_hessian_cuda_version_v2=NativeFunction(lambda: 2),
        generativeqc_ks_hessian_cuda_v2=NativeFunction(native),
        generativeqc_status_message=lambda status: b"injected response failure",
        generativeqc_context_get_last_detail=lambda context: (
            b"injected response failure"
        ),
    )
    return calculator, events


@pytest.mark.parametrize(
    ("method", "symbol", "version"),
    (
        (
            "r2SCAN",
            "r2scan",
            CUDA_R2SCAN_STATIONARY_DOMAIN_VERSION,
        ),
        (
            "M06-2X",
            "m062x",
            CUDA_M062X_STATIONARY_DOMAIN_VERSION,
        ),
        ("MN15", "mn15", CUDA_MN15_STATIONARY_DOMAIN_VERSION),
    ),
)
@pytest.mark.parametrize("state", ("missing", "off", "other", "linked"))
def test_stationary_domain_is_selected_by_the_loaded_library(
    method: str,
    symbol: str,
    version: int,
    state: str,
) -> None:
    library = SimpleNamespace()
    if state != "missing":
        setattr(
            library,
            f"generativeqc_ks_hessian_cuda_{symbol}_domain_v1",
            NativeFunction(lambda: {"off": 0, "other": 1, "linked": version}[state]),
        )
    if method == "r2SCAN":
        assert has_cuda_r2scan_stationary_domain(library) == (state == "linked")
    else:
        expected = (
            CUDA_SPLIT_HYBRID_STATIONARY_DOMAINS[version] if state == "linked" else None
        )
        assert linked_cuda_split_hybrid_domain(library, method) == expected


@pytest.mark.parametrize("model", ("r2SCAN", "M06-2X", "MN15"))
def test_variational_hessian_refuses_incumbent_scf(
    frontend: tuple[Calculator, list[str]],
    model: str,
) -> None:
    calculator, events = frontend
    if model == "r2SCAN":
        calculator._cuda_r2scan_stationary = True
    else:
        version = (
            CUDA_MN15_STATIONARY_DOMAIN_VERSION
            if model == "MN15"
            else CUDA_M062X_STATIONARY_DOMAIN_VERSION
        )
        calculator._cuda_split_hybrid_stationary = (
            model,
            CUDA_SPLIT_HYBRID_STATIONARY_DOMAINS[version],
        )
    atoms = (Atom(1, (0, 0, 0)), Atom(1, (0, 0, 1.4)))
    with pytest.raises(RuntimeError, match="requires the variational SCF domain"):
        calculator.hessian(atoms)
    assert events == ["prepare", "enter", "scf", "close"]


@pytest.mark.parametrize(
    "case",
    (
        "rks",
        "uks",
        "unlinked",
        "auto",
        "non-becke3",
        "implicit-df",
        "ecp",
        "ecp-implicit-df",
        "cartesian-d",
        "spherical-f",
        "spherical-g",
    ),
)
@pytest.mark.parametrize("selector", ("m06-2x", "mn15"))
def test_variational_split_hybrid_public_admission(
    monkeypatch: pytest.MonkeyPatch, case: str, selector: str
) -> None:
    """Capability policy does not run SCF, even when an input is rejected."""

    def available(method: int, output: object) -> int:
        ct.cast(output, ct.POINTER(ct.c_int32))[0] = 1
        return _native.STATUS_SUCCESS

    library = SimpleNamespace(
        generativeqc_ks_options_version=NativeFunction(lambda: 1),
        generativeqc_method_available=NativeFunction(available),
        generativeqc_ks_hessian_cuda_version_v2=NativeFunction(lambda: 2),
        generativeqc_ks_hessian_cuda_m062x_domain_v1=NativeFunction(
            lambda: (
                CUDA_M062X_STATIONARY_DOMAIN_VERSION
                if case != "unlinked" and selector == "m06-2x"
                else 0
            )
        ),
        generativeqc_ks_hessian_cuda_mn15_domain_v1=NativeFunction(
            lambda: (
                CUDA_MN15_STATIONARY_DOMAIN_VERSION
                if case != "unlinked" and selector == "mn15"
                else 0
            )
        ),
        generativeqc_ks_hessian_cuda_v2=NativeFunction(
            lambda *_: pytest.fail("unexpected Hessian execution")
        ),
    )
    install_registry_stub(library, frozenset({"energy"}))
    install_scf_defaults_stub(library)
    monkeypatch.setattr(_native, "load_library", lambda **_: library)
    angular = {"cartesian-d": 2, "spherical-f": 3, "spherical-g": 4}.get(case)
    basis = "sto-3g"
    if angular is not None:
        basis = (Shell(0, angular, (Primitive(1.0, 1.0),)),)
    elif case.startswith("ecp"):
        from test_wb97mv_ecp_hessian import synthetic_basis

        basis = synthetic_basis()
    calculator = Calculator(
        method=f"{selector}-{'uks' if case == 'uks' else 'rks'}",
        device="cuda",
        basis=basis,
        basis_representation="spherical"
        if case.startswith(("spherical-", "ecp"))
        else "cartesian",
        precision="auto" if case == "auto" else "fp64",
        density_fitting="cuda" if case.endswith("implicit-df") else "none",
        ks_options=KsOptions(grid=GridSpec(partition_iterations=2))
        if case == "non-becke3"
        else None,
    )

    class PreparationReached(Exception):
        """A legal shell passed the public gate without starting real SCF."""

    def intercept_preparation(*_args: object, **_kwargs: object) -> None:
        if case == "spherical-f":
            raise PreparationReached
        pytest.fail("unexpected SCF")

    monkeypatch.setattr(calculator, "prepare_batch", intercept_preparation)
    if case in ("rks", "uks", "cartesian-d", "spherical-f", "ecp"):
        assert calculator.second_order_capabilities == {"hvp", "hessian"}
    else:
        assert not calculator.second_order_capabilities
    if case.startswith("ecp"):
        assert "forces" not in calculator.capabilities.supported_properties
    if angular is not None:
        atoms = (Atom(1, (0, 0, 0)), Atom(1, (0, 0, 1.4)))
        if case == "spherical-f":
            with pytest.raises(PreparationReached):
                calculator.hessian(atoms)
        elif case == "cartesian-d":
            with pytest.raises(NotImplementedError, match="orbital shells through f"):
                calculator.hessian(atoms)
        else:
            with pytest.raises(
                NotImplementedError, match="does not expose public hessian"
            ):
                calculator.hessian(atoms)


def test_native_hessian_refuses_externally_owned_dispersion(
    frontend: tuple[Calculator, list[str]],
) -> None:
    calculator, events = frontend
    calculator._dispersion_method_ir = object()
    with pytest.raises(
        NotImplementedError, match="dispersion owned by the native KS batch"
    ):
        run_native_cuda(calculator, (object(),), charge=0, multiplicity=1)
    assert events == ["prepare", "enter", "close"]


@pytest.mark.parametrize("matrix", (False, True))
def test_public_cuda_dispatch_never_enters_legacy_numerics(
    frontend: tuple[Calculator, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    matrix: bool,
) -> None:
    calculator, events = frontend
    atoms = (Atom(1, (0, 0, 0)), Atom(1, (0, 0, 1.4)))

    def reject_legacy(*arguments: Any, **keywords: Any) -> NoReturn:
        raise AssertionError("public CUDA route entered Python numerical owner")

    calculator._run_public_rks_second_order = reject_legacy
    imports = importlib.import_module("builtins")
    original_import = imports.__import__

    def guarded_import(name: str, *arguments: Any, **keywords: Any) -> Any:
        if name.startswith(("generativeqc_compiler", "cupy")) or name in (
            "rks_hessian",
            "rks_hessian_integrals",
            "composite_rks_hessian",
            "_ks_snapshot",
        ):
            raise AssertionError(f"production numerical import: {name}")
        return original_import(name, *arguments, **keywords)

    monkeypatch.setattr(imports, "__import__", guarded_import)
    result = (
        calculator.hessian(atoms)
        if matrix
        else calculator.hessian_vector_product(atoms, np.arange(6).reshape(2, 3))
    )
    values = result.matrix if matrix else result.value
    assert values.shape == ((6, 6) if matrix else (2, 3))
    np.testing.assert_array_equal(values, 0.125)
    with pytest.raises(ValueError):
        values.setflags(write=True)
    diagnostics = result.diagnostics
    diagnostics["response_algorithm"] = "mutated"
    assert result.diagnostics["response_algorithm"] == "native-restarted-gmres"
    assert result.diagnostics["completed_directions"] == (6 if matrix else 1)
    assert result.diagnostics["component_publication"] == "aggregate-only"
    assert events == ["prepare", "enter", "scf", "native", "close"]


def test_fp32_proposal_bridge_selection_and_receipt(
    frontend: tuple[Calculator, list[str]],
) -> None:
    calculator, events = frontend
    atoms = (Atom(1, (0, 0, 0)), Atom(1, (0, 0, 1.4)))
    result = calculator.hessian(atoms, response_cache_precision="fp32-proposal")
    assert result.diagnostics["cpks_cache_precision"] == "fp32-proposal"
    assert result.diagnostics["cpks_operator_bytes"] == 36
    assert result.diagnostics["cpks_proposal_seeds"] == 2
    assert events == ["prepare", "enter", "scf", "native", "close"]


def test_fp32_proposal_requires_a_dense_cuda_matrix_before_scf(
    frontend: tuple[Calculator, list[str]],
) -> None:
    calculator, events = frontend
    atoms = (Atom(1, (0, 0, 0)), Atom(1, (0, 0, 1.4)))
    with pytest.raises(ValueError, match="requires a dense Hessian"):
        run_native_cuda(
            calculator,
            atoms,
            charge=0,
            multiplicity=1,
            direction=np.ones((2, 3)),
            response_cache_precision="fp32-proposal",
        )
    with pytest.raises(ValueError, match="response_cache_precision"):
        calculator.hessian(atoms, response_cache_precision="fp16")
    calculator._cuda_rks_second_order = False
    with pytest.raises(NotImplementedError, match="requires CUDA"):
        calculator.hessian(atoms, response_cache_precision="fp32-proposal")
    assert not events


@pytest.mark.parametrize("version", (0, 1, None))
def test_missing_aot_package_fails_before_scf(
    frontend: tuple[Calculator, list[str]], version: int | None
) -> None:
    calculator, events = frontend
    calculator._library.generativeqc_ks_hessian_cuda_version_v2 = (
        None if version is None else NativeFunction(lambda: version)
    )
    with pytest.raises(NotImplementedError, match="no Python/JIT fallback"):
        run_native_cuda(calculator, (object(),), charge=0, multiplicity=1)
    assert not events


@pytest.mark.parametrize(
    ("version", "has_execute"),
    ((None, True), (0, True), (1, True), (2, False), (2, True)),
)
def test_cuda_hessian_capabilities_follow_loaded_package(
    monkeypatch: pytest.MonkeyPatch, version: int | None, has_execute: bool
) -> None:
    """The public query and invocation agree without preparing or running SCF."""

    def available(method: int, output: object) -> int:
        ct.cast(output, ct.POINTER(ct.c_int32))[0] = 1
        return _native.STATUS_SUCCESS

    library = SimpleNamespace(
        generativeqc_ks_options_version=NativeFunction(lambda: 1),
        generativeqc_method_available=NativeFunction(available),
    )
    install_registry_stub(library, frozenset({"energy"}))
    install_scf_defaults_stub(library)
    if version is not None:
        library.generativeqc_ks_hessian_cuda_version_v2 = NativeFunction(
            lambda: version
        )
    if has_execute:
        library.generativeqc_ks_hessian_cuda_v2 = NativeFunction(
            lambda *_: pytest.fail("unexpected Hessian execution")
        )
    monkeypatch.setattr(_native, "load_library", lambda **_: library)
    calculator = Calculator(method="pbe-rks", basis="sto-3g", device="cuda")
    cpu = Calculator(method="pbe-rks", basis="sto-3g", device="cpu")
    assert cpu.second_order_capabilities == {"hvp", "hessian"}
    if version == 2 and has_execute:
        assert calculator.second_order_capabilities == {"hvp", "hessian"}
    else:
        assert not calculator.second_order_capabilities
        monkeypatch.setattr(
            calculator, "prepare_batch", lambda *_, **__: pytest.fail("unexpected SCF")
        )
        atoms = (Atom(1, (0, 0, 0)), Atom(1, (0, 0, 1.4)))
        with pytest.raises(NotImplementedError, match="no Python/JIT fallback"):
            calculator.hessian(atoms)
        with pytest.raises(NotImplementedError, match="no Python/JIT fallback"):
            calculator.hessian_vector_product(atoms, np.ones((2, 3)))


@pytest.mark.parametrize("invalid", (0, -1, True, 1.5, 2**63))
@pytest.mark.parametrize(
    "argument",
    (
        "integral_budget_bytes",
        "grid_budget_bytes",
        "output_budget_bytes",
        "max_nonlocal_pairs",
    ),
)
def test_invalid_capacities_fail_before_scf(
    frontend: tuple[Calculator, list[str]], argument: str, invalid: Any
) -> None:
    calculator, events = frontend
    with pytest.raises(ValueError, match="positive int64"):
        run_native_cuda(
            calculator, (object(),), charge=0, multiplicity=1, **{argument: invalid}
        )
    assert not events


def test_nonconverged_native_call_closes_batch_without_result(
    frontend: tuple[Calculator, list[str]],
) -> None:
    calculator, events = frontend
    calculator._library.generativeqc_ks_hessian_cuda_v2 = NativeFunction(lambda *_: 4)
    with pytest.raises(RuntimeError, match="injected response failure"):
        run_native_cuda(calculator, (object(),), charge=0, multiplicity=1)
    assert events[-1] == "close"


@pytest.mark.parametrize("strategy", ("recycled", "independent", "block", "invalid"))
def test_requested_non_native_strategy_is_not_silently_ignored(
    frontend: tuple[Calculator, list[str]], strategy: str
) -> None:
    calculator, events = frontend
    with pytest.raises(NotImplementedError, match="no Krylov recycling"):
        calculator.hessian((object(),), strategy=strategy)
    assert not events


@pytest.mark.parametrize(
    "settings",
    (
        {"true_residual_every": 2},
        {"breakdown_tolerance": 1e-12},
        {"stagnation_window": 2},
        {"stagnation_tolerance": 1e-12},
    ),
)
def test_unsupported_solver_controls_are_explicit(settings: dict[str, Any]) -> None:
    with pytest.raises(NotImplementedError):
        _options(64 << 20, 128 << 20, 64 << 20, 100, GMRESOptions(**settings))


def test_native_solver_options_preserve_supported_controls() -> None:
    selected = _options(
        10,
        20,
        30,
        40,
        GMRESOptions(
            rtol=1e-12,
            atol=1e-13,
            restart=17,
            max_iterations=43,
            reorthogonalize=3,
            max_workspace_bytes=123456,
        ),
    )
    assert (selected.relative_tolerance, selected.absolute_tolerance) == (1e-12, 1e-13)
    assert (
        selected.restart,
        selected.max_iterations,
        selected.reorthogonalizations,
    ) == (17, 43, 3)
    assert selected.response_bytes == 123456


def test_native_solver_default_handles_symmetry_zero_rhs() -> None:
    default = _options(10, 20, 30, 40, None)
    assert default.absolute_tolerance == 1e-12
    explicit = _options(10, 20, 30, 40, GMRESOptions(atol=0.0))
    assert explicit.absolute_tolerance == 0.0


@pytest.mark.parametrize(
    "direction",
    (
        [[1, 2]],
        [[True, False, True]],
        [[1j, 0, 0]],
        [[np.nan, 0, 0]],
        [[np.inf, 0, 0]],
    ),
)
def test_direction_admission(direction: Any) -> None:
    with pytest.raises(ValueError, match="finite"):
        checked_direction(direction, 1)


def test_direction_is_detached_before_scf() -> None:
    source = np.arange(6.0).reshape(2, 3)
    direction = checked_direction(source, 2)
    source[:] = -99
    np.testing.assert_array_equal(direction, np.arange(6).reshape(2, 3))
    with pytest.raises(ValueError):
        direction.setflags(write=True)


def test_result_identity_binds_geometry_even_when_values_match(
    frontend: tuple[Calculator, list[str]],
) -> None:
    calculator, _ = frontend
    first = calculator.hessian((Atom(1, (0, 0, 0)), Atom(1, (0, 0, 1.4))))
    second = calculator.hessian((Atom(1, (0, 0, 0)), Atom(1, (0, 0, 1.5))))
    np.testing.assert_array_equal(first.matrix, second.matrix)
    assert first.identity != second.identity


def test_global_resource_plan_requires_the_linked_ledger_contract(
    frontend: tuple[Calculator, list[str]],
) -> None:
    calculator, events = frontend
    calculator._resource_budget = object()
    with pytest.raises(NotImplementedError, match="ResourceBudget admission"):
        calculator.hessian((object(),))
    assert not events


def test_hessian_resource_plan_counts_scf_and_result_lifetimes() -> None:
    from generativeqc._hessian_resources import plan_cuda_hessian
    from generativeqc_compiler.common.resources import (
        ResourceBudget,
        ResourceCandidate,
        ResourceEstimate,
        ResourceIdentity,
        ResourceRequest,
    )

    scf_request = ResourceRequest(
        "ks",
        ResourceIdentity(
            "pbe-rks", "direct", "cuda", "fp64", '{"ao":2}', ("energy",), "scalar"
        ),
        (
            ResourceCandidate(
                "resident",
                "resident",
                (
                    ResourceEstimate("SCF device", 2048, "device:1", 2, 3),
                    ResourceEstimate("SCF host", 1024, "pageable", 1, 2),
                    ResourceEstimate("SCF pinned", 256, "pinned", 2, 3),
                ),
            ),
        ),
    )
    calculator = SimpleNamespace(
        _device_id=1,
        _library=None,
        _effective_ks_options=lambda *_args, **_kw: None,
        _resource_request=lambda *_args, **_kw: scf_request,
        _dispersion_resource_request=lambda *_args, **_kw: None,
    )
    kwargs = {
        "charge": 0,
        "multiplicity": 1,
        "dense": True,
        "options": _options(64 << 20, 128 << 20, 64 << 20, 2**24, None),
    }
    atoms = (Atom(1, (0, 0, 0)), Atom(1, (0, 0, 1.4)))
    budget = ResourceBudget(
        device_bytes=8192,
        per_device_bytes=((1, 4096),),
        device_reserve_bytes=4096,
        host_bytes=2048,
        host_reserve_bytes=192,
        pinned_host_bytes=256,
    )
    plan, response, capacity = plan_cuda_hessian(
        calculator, atoms, budget=budget, **kwargs
    )
    assert capacity == 2048
    assert plan.peak_bytes["device:1"] == plan.peak_bytes["device"] == 4096
    assert plan.peak_bytes["host"] == 1024 + 256 + 2 * 6**2 * 8 == 1856
    assert response.identity.observables == ("hessian",)
    for insufficient in (
        ResourceBudget(host_bytes=2047, host_reserve_bytes=192),
        ResourceBudget(pinned_host_bytes=255),
        ResourceBudget(per_device_bytes=((1, 2048 + 2 * 6**2 * 8 - 1),)),
    ):
        with pytest.raises(MemoryError, match="no supported plan fits"):
            plan_cuda_hessian(calculator, atoms, budget=insufficient, **kwargs)


@pytest.mark.parametrize("operator", ("df_metric", "df_three_center"))
def test_ks_auxiliary_g_scope_is_values_only(operator: str) -> None:
    from generativeqc import Shell
    from generativeqc.basis_capabilities import basis_capability
    from generativeqc.calculator import Primitive

    basis = (Shell(0, 4, (Primitive(0.8, 1.0),)),)
    request = {"backend": "cuda", "operator": operator, "role": "auxiliary"}
    assert not basis_capability(basis, [("H", (0, 0, 0))], **request)["eligible"]
    assert basis_capability(
        basis, [("H", (0, 0, 0))], consumer="wb97mv_df_values", **request
    )["eligible"]
    with pytest.raises(ValueError, match="source-only"):
        basis_capability(
            basis,
            [("H", (0, 0, 0))],
            derivative_order=1,
            consumer="wb97mv_df_values",
            **request,
        )
    with pytest.raises(ValueError, match="source-only"):
        basis_capability(
            basis,
            [("H", (0, 0, 0))],
            consumer="wb97mv_df_values",
            **{**request, "role": "orbital"},
        )


@pytest.mark.parametrize("scalar_type", (np.int32, np.int64, np.uint32, np.uint64))
@pytest.mark.parametrize("field", ("charge", "multiplicity"))
@pytest.mark.parametrize("matrix", (False, True))
def test_native_result_identity_normalizes_integer_inputs(
    frontend: tuple[Calculator, list[str]],
    scalar_type: type,
    field: str,
    matrix: bool,
) -> None:
    """Integral scalars accepted by native preparation retain the same identity."""
    calculator, _ = frontend
    atoms = (Atom(1, (0, 0, 0)), Atom(1, (0, 0, 1.4)))
    values = {"charge": 0, "multiplicity": 1}
    operation = calculator.hessian if matrix else calculator.hessian_vector_product
    arguments = (atoms,) if matrix else (atoms, np.arange(6).reshape(2, 3))
    expected = operation(*arguments, **values)
    values[field] = scalar_type(values[field])
    actual = operation(*arguments, **values)
    assert actual.identity == expected.identity
    np.testing.assert_array_equal(
        actual.matrix if matrix else actual.value,
        expected.matrix if matrix else expected.value,
    )
