"""Exercise Hessian publication accounting without a native SCF dependency."""

from __future__ import annotations

import typing
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

import tools.vibeqc_hessian.analytic as hessian_analytic
from tools.vibeqc_hessian import rks_molecular


def _stub_operator(monkeypatch: pytest.MonkeyPatch) -> typing.Any:
    class Operator:
        xc_kernel = SimpleNamespace(basis=SimpleNamespace(natom=2))
        _source = SimpleNamespace(nbf=2, shell_sizes=(1, 1), atoms=(None, None))
        state = SimpleNamespace(
            identity=SimpleNamespace(
                method="lda-rks",
                to_payload=lambda: {"fixture": "output-budget-only"},
            )
        )

        def validate_current(self) -> None:
            pass

    monkeypatch.setattr(rks_molecular, "NativeRKSResponse", Operator)
    monkeypatch.setattr(rks_molecular, "_checked_plan", lambda _: None)
    return Operator()


def test_symmetry_check_reuses_scratch_before_immutable_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _stub_operator(monkeypatch)
    expected = np.arange(36, dtype=np.float64).reshape(6, 6)
    expected_error = float(np.max(np.abs(expected - expected.T)))
    calls = []
    scratch_refs = []
    original_abs = np.abs
    original_immutable = rks_molecular.immutable

    def fake_many(
        _operator: typing.Any,
        directions: typing.Any,
        **kwargs: typing.Any,
    ) -> SimpleNamespace:
        vectors = np.asarray(directions).reshape(len(directions), 6)
        calls.append(len(directions))
        return SimpleNamespace(
            values=(vectors @ expected.T).reshape(len(directions), 2, 3),
            identity=f"test-block-{len(calls)}",
            diagnostics={"multi_rhs_calls": 1},
        )

    def checked_abs(
        value: typing.Any,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        if np.shape(value) == expected.shape:
            assert kwargs.get("out") is value, "second dense scratch allocation"
            scratch_refs.append(weakref.ref(value))
        return original_abs(value, *args, **kwargs)

    def checked_immutable(value: typing.Any, **kwargs: typing.Any) -> typing.Any:
        assert scratch_refs, "symmetry check was not exercised"
        assert all(ref() is None for ref in scratch_refs)
        return original_immutable(value, **kwargs)

    monkeypatch.setattr(rks_molecular, "rks_hvp_many", fake_many)
    monkeypatch.setattr(rks_molecular.np, "abs", checked_abs)
    monkeypatch.setattr(rks_molecular, "immutable", checked_immutable)
    result = rks_molecular.rks_hessian(
        operator,
        block_size=2,
        output_budget_bytes=2 * expected.nbytes,
    )
    monkeypatch.setattr(rks_molecular.np, "abs", original_abs)

    np.testing.assert_array_equal(result.matrix, expected)
    assert calls == [2, 2, 2]
    assert result.diagnostics["raw_symmetry_error"] == expected_error
    assert result.diagnostics["output_peak_bound_bytes"] == 2 * expected.nbytes
    assert not result.matrix.flags.writeable


def test_one_byte_short_output_budget_refuses_before_block_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _stub_operator(monkeypatch)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("HVP started before output admission")

    monkeypatch.setattr(rks_molecular, "rks_hvp_many", forbidden)
    with pytest.raises(ValueError, match="output_budget_bytes"):
        rks_molecular.rks_hessian(
            operator,
            block_size=2,
            output_budget_bytes=2 * 6 * 6 * np.dtype(np.float64).itemsize - 1,
        )


@pytest.mark.parametrize(
    ("budget", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, ValueError),
        (2**63, ValueError),
        (1.5, ValueError),
        (1, MemoryError),
    ],
)
def test_weighted_provider_admission_has_no_legacy_size_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: typing.Any
) -> None:
    """Keep the removed 12-AO/four-atom gate covered without a native HVP."""
    atoms = tuple(
        SimpleNamespace(atomic_number=2, position=(float(index), 0.0, 0.0))
        for index in range(5)
    )
    shells = tuple(
        SimpleNamespace(
            angular_momentum=0,
            atom_index=index % len(atoms),
            primitives=(SimpleNamespace(exponent=1.0 + 0.01 * index, coefficient=1.0),),
        )
        for index in range(13)
    )
    source = SimpleNamespace(
        _check_open=lambda: None,
        representation="cartesian",
        auxiliary_shells=(),
        nbf=13,
        atoms=atoms,
        shells=shells,
        shell_sizes=(1,) * len(shells),
    )
    sentinel = object()
    monkeypatch.setattr(
        hessian_analytic,
        "_checked_second_hvp_options",
        lambda *args: (sentinel, 0, sentinel),
    )

    data = hessian_analytic._provider_data_from_source(source, tmp_path)

    assert data["state"].nat == 5
    assert len(data["shells"]) == 13
    assert data["output_accumulator_bytes"] == 5 * 3 * np.dtype(np.float64).itemsize


def test_integral_budget_refuses_before_dense_output_allocation(
    monkeypatch: pytest.MonkeyPatch,
    budget: typing.Any,
    error_type: type[Exception],
) -> None:
    operator = _stub_operator(monkeypatch)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError(
            "dense allocation or HVP started before integral admission"
        )

    monkeypatch.setattr(rks_molecular.np, "empty", forbidden)
    monkeypatch.setattr(rks_molecular, "rks_hvp_many", forbidden)
    with pytest.raises(error_type, match="integral_budget_bytes"):
        rks_molecular.rks_hessian(operator, integral_budget_bytes=budget)


@pytest.mark.parametrize("batched", (False, True))
def test_integral_accumulator_refuses_before_response(
    monkeypatch: pytest.MonkeyPatch, batched: bool
) -> None:
    operator = _stub_operator(monkeypatch)
    # Isolate the accumulator branch from the plan-weight envelope. This is a
    # resource-control fixture, not a physical basis or molecular calculation.
    operator._source = SimpleNamespace(nbf=1, shell_sizes=(1,), atoms=(None,) * 7)
    operator.xc_kernel = SimpleNamespace(basis=SimpleNamespace(natom=7))
    direction = np.zeros((7, 3), dtype=np.float64)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("CPKS started before integral accumulator admission")

    monkeypatch.setattr(rks_molecular, "directional_rks_response", forbidden)
    monkeypatch.setattr(rks_molecular, "directional_rks_responses", forbidden)
    with pytest.raises(MemoryError, match="output accumulator"):
        if batched:
            rks_molecular.rks_hvp_many(
                operator, direction[None], integral_budget_bytes=128
            )
        else:
            rks_molecular.rks_hvp(operator, direction, integral_budget_bytes=128)


def test_integral_accumulator_accepts_exact_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _stub_operator(monkeypatch)
    operator._source = SimpleNamespace(nbf=1, shell_sizes=(1,), atoms=(None,) * 7)
    required = 7 * 3 * np.dtype(np.float64).itemsize
    assert rks_molecular._checked_integral_budget(operator, required) == 128
    with pytest.raises(MemoryError, match="output accumulator"):
        rks_molecular._checked_integral_budget(operator, required - 1)
