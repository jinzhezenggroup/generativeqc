"""Host-only admission/dispatch checks; fake native results are not GPU evidence."""

from __future__ import annotations

import ctypes as ct
import importlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NoReturn, Self

import numpy as np
import pytest
from generativeqc import Atom, Calculator
from generativeqc._native_hessian import (
    _Options,
    _options,
    _Receipt,
    checked_direction,
    run_native_cuda,
)
from generativeqc.response_solver import GMRESOptions

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

        def __enter__(self) -> Self:
            events.append("enter")
            return self

        def execute(self, *, strict: bool, properties: tuple[str, ...]) -> None:
            assert strict and properties == ("energy",)
            events.append("scf")

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
        events.append("native")
        return 0

    def prepare(*arguments: Any, **keywords: Any) -> Batch:
        assert len(arguments[0]) == 1
        assert keywords == {"charges": [0], "multiplicities": [1], "warm_start": False}
        events.append("prepare")
        return Batch()

    calculator = object.__new__(Calculator)
    calculator._composite_rks_second_order = True
    calculator._device_name = "cuda"
    calculator._checked_rks_second_order_atoms = lambda atoms, **_: tuple(atoms)
    calculator.prepare_batch = prepare
    calculator._library = SimpleNamespace(
        generativeqc_ks_hessian_cuda_version_v1=NativeFunction(lambda: 1),
        generativeqc_ks_hessian_cuda_v1=NativeFunction(native),
        generativeqc_status_message=lambda status: b"injected response failure",
        generativeqc_context_get_last_detail=lambda context: (
            b"injected response failure"
        ),
    )
    return calculator, events


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


@pytest.mark.parametrize("version", (0, 2, None))
def test_missing_aot_package_fails_before_scf(
    frontend: tuple[Calculator, list[str]], version: int | None
) -> None:
    calculator, events = frontend
    calculator._library.generativeqc_ks_hessian_cuda_version_v1 = (
        None if version is None else NativeFunction(lambda: version)
    )
    with pytest.raises(NotImplementedError, match="no Python/JIT fallback"):
        run_native_cuda(calculator, (object(),), charge=0, multiplicity=1)
    assert not events


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
    calculator._library.generativeqc_ks_hessian_cuda_v1 = NativeFunction(lambda *_: 4)
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


def test_global_resource_plan_is_not_silently_undercharged(
    frontend: tuple[Calculator, list[str]],
) -> None:
    calculator, events = frontend
    calculator._resource_budget = object()
    with pytest.raises(NotImplementedError, match="global ResourceBudget"):
        calculator.hessian((object(),))
    assert not events


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
