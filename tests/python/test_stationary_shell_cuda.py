"""Source-executing prepared-shell handoff tests without a CUDA dependency."""

from __future__ import annotations

import ast
import ctypes as ct
import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_join() -> typing.Any:
    path = ROOT / "python/generativeqc/_stationary_shell_cuda.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))

    def check(_library: typing.Any, status: int, **_kwargs: typing.Any) -> None:
        if status:
            raise RuntimeError(f"native failure {status}")

    namespace = {
        "ct": ct,
        "np": np,
        "typing": typing,
        "_native": SimpleNamespace(check=check),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[function.name]


def fixture(
    *, exchange: bool = False, status: int = 0, bad: bool = False
) -> typing.Any:
    source_names = ("one_electron", "coulomb", "overlap_pulay") + (
        ("exact_exchange",) if exchange else ()
    )
    data = np.arange(12, dtype=float).reshape(2, 2, 3)
    if not exchange:
        data[1] = 0
    if bad:
        data[0, 0, 0] = np.nan
    calls = []

    def execute(
        _batch: typing.Any,
        _snapshot: typing.Any,
        pointer: typing.Any,
        count: int,
        _budget: typing.Any,
        retained: typing.Any,
    ) -> int:
        calls.append("native")
        np.ctypeslib.as_array(pointer, shape=(count,))[:] = data.ravel()
        ct.cast(retained, ct.POINTER(ct.c_uint64))[0] = 512
        return status

    def seed(*_args: typing.Any) -> None:
        pass

    def call(name: str, _handle: typing.Any, pointer: typing.Any, count: int) -> None:
        calls.append((name, np.ctypeslib.as_array(pointer, shape=(count,)).copy()))

    source = SimpleNamespace(
        _library=SimpleNamespace(
            generativeqc_ks_snapshot_cuda_full_range_shell_gradient_v1=execute
        ),
        _batch=SimpleNamespace(_batch=1, _context=2),
        _handle=3,
    )
    owner = SimpleNamespace(
        library=SimpleNamespace(stationary_seed_sources_v1=seed), handle=4, _call=call
    )
    return (SimpleNamespace(_source=source), owner, source_names, data, calls)


@pytest.mark.parametrize("exchange", [False, True])
def test_separate_sources_seeded_without_recomputing_weights(exchange: bool) -> None:
    state, owner, names, data, calls = fixture(exchange=exchange)
    selected, work = load_join()(
        state, owner, names, 2, ecp=False, maximum_host_bytes=4096
    )
    assert selected == (("coulomb", "exact_exchange") if exchange else ("coulomb",))
    assert calls[0] == "native" and len(calls) == 2
    seeds = calls[1][1].reshape(len(names), 2, 3)
    np.testing.assert_array_equal(seeds[names.index("coulomb")], data[0])
    if exchange:
        np.testing.assert_array_equal(seeds[names.index("exact_exchange")], data[1])
    np.testing.assert_array_equal(seeds[0], 0)
    assert work["retained_device_bytes"] == 512
    assert work["source_seed_h2d_bytes"] == seeds.nbytes
    assert work["shell_work_counts"] is None


@pytest.mark.parametrize("missing", ["native", "seed", "ecp"])
def test_unavailable_capability_never_executes_or_seeds(missing: str) -> None:
    state, owner, names, _, calls = fixture()
    if missing == "native":
        del state._source._library.generativeqc_ks_snapshot_cuda_full_range_shell_gradient_v1
    if missing == "seed":
        del owner.library.stationary_seed_sources_v1
    selected, work = load_join()(
        state, owner, names, 2, ecp=missing == "ecp", maximum_host_bytes=4096
    )
    assert selected == () and work["route"] == "bounded-public-ao" and (calls == [])


def test_only_not_implemented_selects_capability_fallback() -> None:
    state, owner, names, _, calls = fixture(status=3)
    selected, work = load_join()(
        state, owner, names, 2, ecp=False, maximum_host_bytes=4096
    )
    assert selected == () and work["route"] == "bounded-public-ao"
    assert calls == ["native"]


@pytest.mark.parametrize("status", [1, 2, 4, 5, 6, 7, 8])
def test_failures_do_not_disappear_into_generic_reexecution(status: int) -> None:
    state, owner, names, _, calls = fixture(status=status)
    with pytest.raises(RuntimeError, match="native failure"):
        load_join()(state, owner, names, 2, ecp=False, maximum_host_bytes=4096)
    assert calls == ["native"]


def test_nonfinite_sources_never_seed_stationary_owner() -> None:
    state, owner, names, _, calls = fixture(bad=True)
    with pytest.raises(RuntimeError, match="nonfinite"):
        load_join()(state, owner, names, 2, ecp=False, maximum_host_bytes=4096)
    assert calls == ["native"]


def test_missing_exchange_source_cannot_drop_a_nonzero_k() -> None:
    state, owner, names, _, calls = fixture(exchange=True)
    with pytest.raises(RuntimeError, match="unexpected exchange"):
        load_join()(state, owner, names[:-1], 2, ecp=False, maximum_host_bytes=4096)
    assert calls == ["native"]


def test_seed_failure_propagates_without_reporting_shell_success() -> None:
    state, owner, names, _, calls = fixture()

    def fail(*_args: typing.Any) -> None:
        raise RuntimeError("seed failed")

    owner._call = fail
    with pytest.raises(RuntimeError, match="seed failed"):
        load_join()(state, owner, names, 2, ecp=False, maximum_host_bytes=4096)
    assert calls == ["native"]
