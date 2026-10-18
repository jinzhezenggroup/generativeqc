"""Selected public fixed-mask pullbacks against independent NumPy oracles."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp


@pytest.fixture(params=("reference", "cpu"))
def backend(request: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "cpu":
        from generativeqc_compiler.tensor import autodiff, interpreter

        def forbidden(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("native VJP delegated to a reference oracle")

        monkeypatch.setattr(interpreter, "execute", forbidden)
        monkeypatch.setattr(autodiff, "vjp", forbidden)
        monkeypatch.setattr(autodiff, "jvp", forbidden)
    return request.param


def expression(x: Any, y: Any, mask: Any, unused: Any) -> dict:
    gathered = xp.take(x, (2, 0, 2), axis=0)
    selected = xp.where(mask, gathered * gathered, y * y)
    return {
        "vector": selected,
        "norm": xp.sum(selected * selected),
        "constant": xp.ones_like(unused),
    }


def oracle(x: np.ndarray, y: np.ndarray, mask: np.ndarray, seeds: dict) -> tuple:
    gathered = x[[2, 0, 2]]
    selected = np.where(mask, gathered**2, y**2)
    weight = seeds["vector"] + 2 * seeds["norm"] * selected
    dx = np.zeros_like(x)
    np.add.at(dx, [2, 0, 2], np.where(mask, 2 * gathered * weight, 0))
    dy = np.where(mask, 0, 2 * y * weight).sum(axis=0)
    objective = np.sum(seeds["vector"] * selected) + seeds["norm"] * np.sum(selected**2)
    objective += seeds["constant"].sum()
    return selected, dx, dy, objective


def test_multi_output_snapshot_order_replay_finite_difference_and_dot(
    backend: str, tmp_path: Path
) -> None:
    x = np.asarray([[0.4, 0.8], [1.1, 0.3], [0.5, 1.3]])
    y = np.asarray([0.7, 1.2])
    mask = np.asarray([[True], [False], [True]])
    unused = np.ones(4)
    source = xp.compile(expression, backend=backend, cache=tmp_path)
    saved = tuple(np.array(value, copy=True) for value in (x, y, mask, unused))
    primal, pullback = xp.vjp(
        source, x, y, mask, unused, argnums=(-3, 0, -1), allow_external_mask_where=True
    )
    try:
        seeds = {
            "vector": np.asarray([[0.2, 0.8], [-0.3, 0.5], [0.4, -0.2]]),
            "norm": np.asarray(0.3),
            "constant": np.ones(4),
        }
        selected, dx, dy, _ = oracle(*saved[:3], seeds)
        np.testing.assert_allclose(primal["vector"], selected, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(
            primal["norm"], np.sum(selected**2), rtol=2e-13, atol=2e-13
        )
        x[:] = 2
        y[:] = 3
        mask[:] = ~mask
        unused[:] = 99
        # A caller owner can be closed without closing the snapshot's executor.
        source.close()
        actual_dy, actual_dx, actual_unused = pullback(seeds)
        np.testing.assert_allclose(actual_dx, dx, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(actual_dy, dy, rtol=2e-13, atol=2e-13)
        np.testing.assert_array_equal(actual_unused, np.zeros(4))
        direction_x = np.asarray([[0.1, -0.2], [0.3, 0.1], [-0.4, 0.2]])
        direction_y = np.asarray([0.3, -0.2])
        step = 1e-5
        fd = (
            oracle(
                saved[0] + step * direction_x,
                saved[1] + step * direction_y,
                saved[2],
                seeds,
            )[3]
            - oracle(
                saved[0] - step * direction_x,
                saved[1] - step * direction_y,
                saved[2],
                seeds,
            )[3]
        ) / (2 * step)
        dot = np.sum(actual_dx * direction_x) + np.sum(actual_dy * direction_y)
        np.testing.assert_allclose(fd, dot, rtol=2e-8, atol=2e-8)
        second_seeds = {name: seed * -0.5 for name, seed in seeds.items()}
        second_dy, second_dx, _ = pullback(second_seeds)
        np.testing.assert_allclose(second_dx, -0.5 * dx, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(second_dy, -0.5 * dy, rtol=2e-13, atol=2e-13)
        assert not np.shares_memory(actual_dx, second_dx)
        assert not np.shares_memory(actual_dx, x)
        pullback.close()
        with pytest.raises(RuntimeError, match="closed"):
            pullback(seeds)
    finally:
        source.close()
        pullback.close()


@pytest.mark.parametrize("shape", ((), (0, 2), (2, 3)))
@pytest.mark.parametrize("argnums", (0, (1, 0)))
def test_scalar_empty_disconnected_inputs_and_all_output_cotangents(
    backend: str, shape: tuple, argnums: Any, tmp_path: Path
) -> None:
    def function(x: Any, unused: Any, mask: Any) -> dict:
        return {
            "out": xp.where(mask, x * x, x),
            "empty": xp.ones((0,)),
            "constant": xp.ones_like(unused),
        }

    x, unused, mask = np.full(shape, 0.5), np.ones(4), np.full(shape, True)
    _, pullback = xp.vjp(
        function,
        x,
        unused,
        mask,
        argnums=argnums,
        backend=backend,
        allow_external_mask_where=True,
        cache=tmp_path,
    )
    seeds = {"out": np.ones(shape), "constant": np.ones(4), "empty": np.ones(0)}
    try:
        result = pullback(seeds)
        assert isinstance(result, tuple)
        if type(argnums) is int:
            assert len(result) == 1
            np.testing.assert_array_equal(result[0], 2 * x)
        else:
            np.testing.assert_array_equal(result[0], np.zeros_like(unused))
            np.testing.assert_array_equal(result[1], 2 * x)
        for bad in (
            {"out": seeds["out"]},
            {**seeds, "constant": np.full(4, np.inf)},
            {**seeds, "empty": np.ones(0, dtype=np.float32)},
            {**seeds, "out": np.ones((1, *shape))},
        ):
            with pytest.raises(ValueError):
                pullback(bad)
        # Failed seeds do not mutate or close a reusable pullback.
        np.testing.assert_array_equal(pullback(seeds)[-1], 2 * x)
    finally:
        pullback.close()


@pytest.mark.parametrize(
    "operation,bad",
    ((xp.log, -1.0), (xp.sqrt, 0.0), (xp.reciprocal, 0.0), (xp.exp, 1000.0)),
)
@pytest.mark.parametrize("unused_shape", ((3,), (0,)))
def test_unselected_forward_domains_survive_zero_pullback(
    backend: str, operation: Any, bad: float, unused_shape: tuple, tmp_path: Path
) -> None:
    def function(x: Any, unused: Any, mask: Any) -> Any:
        return xp.where(mask, x * x, operation(x))

    pullback = None
    try:
        with pytest.raises(ValueError):
            _, pullback = xp.vjp(
                function,
                np.asarray([bad]),
                np.ones(unused_shape),
                np.asarray([True]),
                backend=backend,
                argnums=1,
                allow_external_mask_where=True,
                cache=tmp_path,
            )
            pullback(np.ones(1))
    finally:
        if pullback is not None:
            pullback.close()
    _, recovered = xp.vjp(
        function,
        np.asarray([2.0]),
        np.ones(unused_shape),
        np.asarray([True]),
        backend=backend,
        argnums=1,
        allow_external_mask_where=True,
        cache=tmp_path,
    )
    try:
        np.testing.assert_array_equal(recovered(np.ones(1))[0], np.zeros(unused_shape))
    finally:
        recovered.close()


def test_selected_primals_resolve_omitted_defaults_and_preserve_tuple_abi(
    backend: str, tmp_path: Path
) -> None:
    default_scale = np.asarray([2.0, 3.0])

    def function(x: Any, scale: Any = default_scale) -> Any:
        return x * scale

    _, pullback = xp.vjp(
        function, np.ones(2), backend=backend, argnums=-1, cache=tmp_path
    )
    try:
        result = pullback(np.ones(2))
        assert isinstance(result, tuple) and len(result) == 1
        np.testing.assert_array_equal(result[0], [2.0, 3.0])
    finally:
        pullback.close()
    for argnums in (1, -2, (), (0, -1), True, [0], ("x",)):
        with pytest.raises((TypeError, ValueError), match="argnums"):
            xp.vjp(
                function, np.ones(2), backend=backend, argnums=argnums, cache=tmp_path
            )
    _, all_inputs = xp.vjp(
        function, np.ones(2), np.ones(2), backend=backend, cache=tmp_path
    )
    try:
        assert len(all_inputs(np.ones(2))) == 2
    finally:
        all_inputs.close()


def test_invalid_admission_fails_before_native_compiler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from generativeqc_compiler.common import cpp_adapter

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("invalid VJP reached compiler discovery")

    monkeypatch.setattr(cpp_adapter, "CppCompilerAdapter", forbidden)

    def function(x: Any, y: Any, mask: Any) -> Any:
        return xp.where(mask, x * x, y * y)

    x, y, mask = np.ones(2), np.ones(2), np.ones(2, dtype=bool)
    for options in (
        {},
        {"argnums": (0, 1)},
        {"argnums": 2, "allow_external_mask_where": True},
    ):
        with pytest.raises((TypeError, ValueError)):
            xp.vjp(function, x, y, mask, cache=tmp_path, **options)
    for flag in (1, None, "true", np.bool_(True)):
        with pytest.raises(TypeError, match="must be Boolean"):
            xp.vjp(function, x, y, mask, allow_external_mask_where=flag)
    with pytest.raises(ValueError, match="reference or CPU"):
        xp.vjp(
            function,
            x,
            y,
            mask,
            backend="cuda",
            argnums=(0, 1),
            allow_external_mask_where=True,
        )
    with pytest.raises(ValueError, match="float64"):
        xp.vjp(
            function,
            x.astype(np.float32),
            y.astype(np.float32),
            mask,
            argnums=(0, 1),
            allow_external_mask_where=True,
        )
    with pytest.raises(ValueError, match="external Boolean mask"):
        xp.vjp(lambda x: xp.where(x > 0, x, x * x), x, allow_external_mask_where=True)
    for options in ({"max_bytes": 1}, {"max_work": 0}, {"max_nodes": 1}):
        source = xp.compile(function, cache=tmp_path, **options)
        try:
            with pytest.raises(ValueError, match="budget"):
                xp.vjp(
                    source, x, y, mask, argnums=(0, 1), allow_external_mask_where=True
                )
            assert not source._execution.closed
        finally:
            source.close()


def test_disconnected_scientific_and_float32_selection_reject(backend: str) -> None:
    def function(x: Any, unused: Any, mask: Any) -> Any:
        return xp.where(mask, x * x, x)

    with pytest.raises(ValueError, match="float64"):
        xp.vjp(
            function,
            np.ones(2),
            np.ones(3, dtype=np.float32),
            np.ones(2, dtype=bool),
            argnums=1,
            backend=backend,
            allow_external_mask_where=True,
        )
    ao = xp.IndexSpace("ao", "ao", 3)
    spec = xp.TensorSpec((xp.Index("p", ao),), role="input")
    source = xp.compile(function, backend=backend, specs={"unused": spec})
    try:
        with pytest.raises(ValueError, match="generic"):
            xp.vjp(
                source,
                np.ones(2),
                np.ones(3),
                np.ones(2, dtype=bool),
                argnums=1,
                allow_external_mask_where=True,
            )
    finally:
        source.close()


def test_compiler_failure_closes_failed_owner_and_preserves_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc.experimental._array_api_execution import _Executor

    source = xp.compile(lambda x, mask: xp.where(mask, x * x, x))
    failed_owners = []

    def fail(self: Any, program: Any) -> Any:
        failed_owners.append(self)
        raise RuntimeError("injected native compiler failure")

    monkeypatch.setattr(_Executor, "_build", fail)
    try:
        with pytest.raises(RuntimeError, match="injected native compiler failure"):
            xp.vjp(
                source,
                np.ones(2),
                np.ones(2, dtype=bool),
                argnums=0,
                allow_external_mask_where=True,
            )
        assert failed_owners and all(owner.closed for owner in failed_owners)
        assert not source._execution.closed
    finally:
        source.close()


def test_capability_names_explicit_fixed_mask_pullback_contract() -> None:
    capability = xp.capabilities()["external_mask_vjp"]
    assert capability["default"] is False
    assert capability["backends"] == ("reference", "cpu")
    assert capability["real_dtype"] == "float64"


def test_selected_cotangent_bits_survive_public_packaging(
    backend: str, tmp_path: Path
) -> None:
    largest = np.finfo(np.float64).max
    tiny = np.nextafter(0.0, 1.0)
    seed = np.asarray([-0.0, 0.0, -tiny, tiny, -largest, largest])
    mask = np.asarray([True, False, False, True, True, False])
    _, pullback = xp.vjp(
        lambda x, y, mask: xp.where(mask, x, y),
        np.ones(6),
        np.ones(6),
        mask,
        argnums=(0, 1),
        backend=backend,
        allow_external_mask_where=True,
        cache=tmp_path,
    )
    try:
        dx, dy = pullback(seed)
        np.testing.assert_array_equal(
            np.asarray(dx).view(np.uint64), np.where(mask, seed, 0.0).view(np.uint64)
        )
        np.testing.assert_array_equal(
            np.asarray(dy).view(np.uint64), np.where(mask, 0.0, seed).view(np.uint64)
        )
    finally:
        pullback.close()


def test_non_selected_live_inputs_are_snapshotted(backend: str, tmp_path: Path) -> None:
    x, y, mask = (
        np.asarray([0.2, 0.7]),
        np.asarray([0.3, 0.8]),
        np.asarray([True, False]),
    )
    _, pullback = xp.vjp(
        lambda x, y, mask: xp.where(mask, x * y, y * y),
        x,
        y,
        mask,
        argnums=0,
        backend=backend,
        allow_external_mask_where=True,
        cache=tmp_path,
    )
    try:
        x[:] = np.inf
        y[:] = np.nan
        mask[:] = ~mask
        np.testing.assert_array_equal(pullback(np.ones(2))[0], [0.3, 0.0])
    finally:
        pullback.close()
