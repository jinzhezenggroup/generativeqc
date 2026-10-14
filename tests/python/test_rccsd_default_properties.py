"""Exercise output selection through real facades without native SCF execution."""

from __future__ import annotations

import ctypes
import typing
from types import SimpleNamespace

import pytest
from generativeqc import Calculator, MethodCapabilities, _native
from generativeqc import calculator as calculator_module

SUPPORTED = frozenset({"energy", "forces"})
ATOMS = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]


class _NativeRequest(Exception):
    """Stop at the native execution boundary, recording its output contract."""

    def __init__(self, outputs: typing.Iterable[typing.Any]) -> None:
        self.force_buffers = tuple(
            (bool(output.forces), output.force_count) for output in outputs
        )


@pytest.fixture(params=("singlepoint", "batch"))
def selector(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> tuple[bool, typing.Callable[..., tuple[tuple[bool, int], ...]]]:
    batched = request.param == "batch"

    def handle(*args: typing.Any) -> int:
        args[-1]._obj.value = 1
        return _native.STATUS_SUCCESS

    def available(method: int, output: typing.Any) -> int:
        output._obj.value = 1
        return _native.STATUS_SUCCESS

    def execute_calculation(handle: object, output: typing.Any) -> int:
        raise _NativeRequest((output._obj,))

    def execute_batch(
        handle: object, inputs: object, count: int, outputs: typing.Any, size: int
    ) -> int:
        raise _NativeRequest(outputs)

    def success(*args: object) -> int:
        return _native.STATUS_SUCCESS

    library = SimpleNamespace(
        generativeqc_method_available=available,
        generativeqc_context_create=handle,
        generativeqc_context_destroy=success,
        generativeqc_system_destroy=success,
        generativeqc_calculation_prepare=handle,
        generativeqc_calculation_execute=execute_calculation,
        generativeqc_calculation_destroy=success,
        generativeqc_batch_prepare=handle,
        generativeqc_batch_execute=execute_batch,
        generativeqc_batch_destroy=success,
        generativeqc_batch_get_system_count=lambda owner: 1,
    )
    monkeypatch.setattr(_native, "load_library", lambda **kwargs: library)
    monkeypatch.setattr(
        calculator_module,
        "method_capabilities",
        lambda method: MethodCapabilities(
            method, "post_hartree_fock", True, True, SUPPORTED
        ),
    )
    # Basis/system construction is outside this selection contract. The real
    # facade still validates requests and builds the native output descriptors.
    monkeypatch.setattr(
        Calculator, "_create_native_system", lambda *args, **kwargs: ctypes.c_void_p(1)
    )

    def select(
        method: str,
        properties: typing.Any = None,
        *,
        fitted: bool = False,
    ) -> tuple[tuple[bool, int], ...]:
        calculator = Calculator(
            method=method, device="cpu", density_fitting="cpu" if fitted else "none"
        )
        try:
            with pytest.raises(_NativeRequest) as caught:
                if batched:
                    with calculator.prepare_batch([ATOMS], warm_start=False) as batch:
                        batch.execute(properties=properties)
                else:
                    calculator.singlepoint(ATOMS, properties=properties)
            return caught.value.force_buffers
        finally:
            calculator.clear_cache()

    return batched, select


def test_rccsd_omitted_properties_remain_energy_only(selector: typing.Any) -> None:
    _, select = selector
    assert select("rccsd") == ((False, 0),)


@pytest.mark.parametrize("properties", (("energy",), ("energy", "forces")))
def test_explicit_rccsd_requests_are_not_replaced(
    selector: typing.Any, properties: tuple[str, ...]
) -> None:
    _, select = selector
    expected = (True, 6) if "forces" in properties else (False, 0)
    assert select("rccsd", properties) == (expected,)


@pytest.mark.parametrize("method", ("rhf", "rccsd(t)"))
def test_other_method_defaults_remain_unchanged(
    selector: typing.Any, method: str
) -> None:
    _, select = selector
    assert select(method) == ((True, 6),)


@pytest.mark.parametrize("fitted", (False, True))
def test_existing_mp2_default_policy_is_preserved(
    selector: typing.Any, fitted: bool
) -> None:
    batched, select = selector
    expected = (False, 0) if fitted and not batched else (True, 6)
    assert select("mp2", fitted=fitted) == (expected,)


@pytest.mark.parametrize(
    ("properties", "error", "message"),
    (
        ("energy", TypeError, "iterable of property names"),
        ((), ValueError, "must include 'energy'"),
        (("forces",), ValueError, "must include 'energy'"),
        (
            ("energy", "dipole"),
            ValueError,
            "unsupported properties|does not support properties",
        ),
        ([[]], TypeError, "hashable property names"),
    ),
)
def test_invalid_rccsd_requests_fail_before_native_execution(
    selector: typing.Any,
    properties: typing.Any,
    error: type[Exception],
    message: str,
) -> None:
    _, select = selector
    with pytest.raises(error, match=message):
        select("rccsd", properties)
