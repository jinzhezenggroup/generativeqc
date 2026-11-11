"""Gaussian ECP input admission before native arithmetic or oversized copies."""

import ctypes as ct
import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import _native
from generativeqc._ks_snapshot import NativeKsSnapshot


class NativeBoundary:
    """Observe ABI inputs without asserting any ECP numerical qualification."""

    def __init__(self) -> None:
        self.calls = 0
        self.status = _native.STATUS_SUCCESS
        self.current = True
        self.expire_on_evaluation = False
        self.weights = None

    def __call__(self, *args: object) -> int:
        self.calls += 1
        if self.status != _native.STATUS_SUCCESS:
            return self.status
        weights, count = args[3:5]
        self.weights = (
            None
            if weights is None
            else np.ctypeslib.as_array(weights, shape=(count,)).copy()
        )
        output, size, work, work_size = args[7:]
        np.ctypeslib.as_array(output, shape=(size,))[:] = (
            1.0 if self.weights is None else self.weights.sum()
        )
        np.ctypeslib.as_array(work, shape=(work_size,))[:] = np.arange(work_size)
        if self.expire_on_evaluation:
            self.current = False
        return _native.STATUS_SUCCESS


@pytest.fixture
def snapshot(
    request: pytest.FixtureRequest,
) -> tuple[NativeKsSnapshot, NativeBoundary]:
    """Use the real Python token checks around an observed, test-only native ABI."""
    boundary = NativeBoundary()
    source = object.__new__(NativeKsSnapshot)
    source.backend = "cpu"
    source.hamiltonian = "scalar-semilocal-ecp"
    source.metadata = (0, getattr(request, "param", 2), 0, 2)
    source._handle = 17
    source._batch = SimpleNamespace(_batch=29, _context=None, _ensure_open=lambda: None)
    source._library = SimpleNamespace(
        generativeqc_ks_snapshot_ecp_gaussian_v1=boundary,
        generativeqc_ks_snapshot_check_v1=lambda *args: int(not boundary.current),
        generativeqc_ks_snapshot_destroy_v1=lambda handle: None,
    )
    return source, boundary


@pytest.mark.parametrize(
    "weights",
    (
        np.array([[1 + 2j, 0], [0, 1]]),
        np.ones((2, 2), dtype=bool),
        np.ones((2, 2), dtype=object),
        np.full((2, 2), "1"),
        [[1 + 2j, 0], [0, 1]],
        [[True, 0], [0, 1]],
        [["1", 0], [0, 1]],
        np.ones((1, 4)),
        np.ones((2, 2, 1)),
        [[1, 2]],
        [[1], [2, 3]],
        1,
        [[float("nan"), 0], [0, 1]],
        [[float("inf"), 0], [0, 1]],
        [[10**400, 0], [0, 1]],
    ),
)
def test_invalid_weights_never_reach_native_arithmetic(
    snapshot: tuple, weights: object
) -> None:
    source, boundary = snapshot
    with pytest.raises(ValueError, match="ordered AO matrix"):
        source.ecp_gaussian(budget_bytes=65536, weights=weights)
    assert boundary.calls == 0
    result, _ = source.ecp_gaussian(budget_bytes=65536, weights=np.eye(2))
    assert np.isfinite(result).all() and boundary.calls == 1


@pytest.mark.parametrize("nested", (False, True))
def test_wrong_shape_is_rejected_without_materializing_large_input(
    snapshot: tuple, nested: bool
) -> None:
    source, boundary = snapshot
    weights = (
        [[1.0] * 524288] * 2 if nested else np.broadcast_to(np.array(1.0), (1024, 1024))
    )
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="ordered AO matrix"):
            source.ecp_gaussian(budget_bytes=65536, weights=weights)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1 << 20
    assert boundary.calls == 0


@pytest.mark.parametrize(
    "weights",
    (
        [[1, 2], [3, 4]],
        ((1.0, 2.0), (3.0, 4.0)),
        [np.array([1, 2]), np.array([3, 4])],
        np.arange(4, dtype=np.int32).reshape(2, 2),
        np.asfortranarray(np.arange(4.0).reshape(2, 2)),
        np.arange(16.0).reshape(4, 4)[::2, ::2],
        np.arange(4.0).reshape(2, 2).astype(">f8"),
        np.broadcast_to(np.array(2.0), (2, 2)),
    ),
)
def test_valid_real_layouts_preserve_order_and_detach_output(
    snapshot: tuple, weights: object
) -> None:
    source, boundary = snapshot
    result, work = source.ecp_gaussian(budget_bytes=65536, weights=weights)
    np.testing.assert_array_equal(boundary.weights, np.asarray(weights).ravel())
    np.testing.assert_array_equal(result, np.asarray(weights).sum())
    assert result.shape == (2, 43) and not result.flags.writeable
    assert work["workspace_bound_bytes"] == 6 + 2 * 2 * ct.sizeof(ct.c_double)


def test_input_and_output_capacity_reject_before_native_work(snapshot: tuple) -> None:
    source, boundary = snapshot
    required = (2 * (1 + 6 + 6**2) + 2**2) * 8
    with pytest.raises(MemoryError, match="publication"):
        source.ecp_gaussian(budget_bytes=required, weights=np.eye(2))
    assert boundary.calls == 0


def test_native_failure_and_token_expiration_never_publish(snapshot: tuple) -> None:
    source, boundary = snapshot
    boundary.status = _native.STATUS_OUT_OF_MEMORY
    with pytest.raises(MemoryError, match="endpoint"):
        source.ecp_gaussian(budget_bytes=65536, weights=np.eye(2))
    boundary.status = _native.STATUS_SUCCESS
    result, _ = source.ecp_gaussian(budget_bytes=65536, weights=np.eye(2))
    assert np.isfinite(result).all()
    boundary.expire_on_evaluation = True
    with pytest.raises(ValueError, match="stale"):
        source.ecp_gaussian(budget_bytes=65536, weights=np.eye(2))
    calls = boundary.calls
    with pytest.raises(ValueError, match="stale"):
        source.ecp_gaussian(budget_bytes=65536, weights=np.eye(2))
    assert boundary.calls == calls


@pytest.mark.parametrize("snapshot", (1024,), indirect=True)
@pytest.mark.parametrize("finite", (False, True))
def test_finite_validation_reuses_the_admitted_output_envelope(
    snapshot: tuple, finite: bool
) -> None:
    source, boundary = snapshot
    count = source.metadata[1]
    weights = np.ones((count, count), dtype=np.float32)
    if not finite:
        weights[-1, -1] = np.nan
    # Model the native workspace refusal after Python validates a valid matrix.
    # The native stub returns before copying weights or publishing any output.
    boundary.status = _native.STATUS_OUT_OF_MEMORY
    budget = count**2 * 8 + 2 * (1 + 6 + 6**2) * 8 + 1
    tracemalloc.start()
    try:
        with pytest.raises(MemoryError if finite else ValueError):
            source.ecp_gaussian(budget_bytes=budget, weights=weights)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Allow Python object overhead, not another full AO-sized boolean panel.
    assert peak < budget + (64 << 10)
    assert bool(boundary.calls) is finite
