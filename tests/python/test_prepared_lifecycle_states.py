"""The artifact lease and native endpoint use the same legal host states."""

from threading import RLock

import pytest
from generativeqc import _stationary_cuda
from generativeqc_compiler.common.prepared_execution import (
    PreparedArtifactBinding,
    PreparedExecutionLease,
    PreparedExecutionRequest,
)
from generativeqc_compiler.common.provenance import canonical_hash


def test_stationary_reentrant_refusal_preserves_active_transaction() -> None:
    """Reject nested entry without revoking the outer consumer's live lease."""
    request = PreparedExecutionRequest(
        "test",
        *(
            canonical_hash(name)
            for name in ("science", "target", "schedule", "workspace")
        ),
    )
    lease = PreparedExecutionLease()
    lease.install(
        request,
        (
            PreparedArtifactBinding(
                canonical_hash("artifact"), canonical_hash("binary")
            ),
        ),
        host_bytes=0,
        device_bytes=0,
    )
    lease.begin(request, max_host_bytes=0, max_device_bytes=0)
    owner = object.__new__(_stationary_cuda.PreparedStationaryCudaExecution)
    owner._lock = RLock()
    owner._lease = lease
    owner._stack = None
    try:
        with pytest.raises(RuntimeError, match="running"):
            _stationary_cuda.complete_rks_cuda_gradient_diagnostic(
                None, None, compiler=None, cache=None, prepared=owner
            )
        assert lease.running and not lease.failed
        lease.mark_success()
    finally:
        if lease.running:
            lease.mark_failure()
        owner.close()


def test_bound_running_recovery_and_terminal_close() -> None:
    request = PreparedExecutionRequest(
        "test",
        *(
            canonical_hash(name)
            for name in ("science", "target", "schedule", "workspace")
        ),
    )
    lease = PreparedExecutionLease()
    assert lease.state == "unbound"
    lease.install(
        request,
        (
            PreparedArtifactBinding(
                canonical_hash("artifact"), canonical_hash("binary")
            ),
        ),
        host_bytes=64,
        device_bytes=128,
    )
    assert lease.state == "bound"
    lease.begin(request, max_host_bytes=64, max_device_bytes=128)
    assert lease.state == "running"
    for operation in (
        lambda: lease.begin(request, max_host_bytes=64, max_device_bytes=128),
        lease.mark_refresh,
        lease.invalidate,
        lease.close,
    ):
        with pytest.raises(RuntimeError, match="running"):
            operation()
    lease.mark_success()
    assert lease.state == "warm"
    lease.begin(request, max_host_bytes=64, max_device_bytes=128)
    lease.mark_failure()
    assert lease.state == "failed"
    with pytest.raises(RuntimeError, match="refresh"):
        lease.begin(request, max_host_bytes=64, max_device_bytes=128)
    lease.mark_refresh()
    lease.begin(request, max_host_bytes=64, max_device_bytes=128)
    lease.mark_success()
    assert lease.state == "warm" and lease.executions == 2
    lease.close()
    lease.close()
    assert lease.state == "closed" and lease.contract is None
    with pytest.raises(RuntimeError, match="closed"):
        lease.require(request, max_host_bytes=64, max_device_bytes=128)
    with pytest.raises(RuntimeError, match="closed"):
        lease.install(request, (), host_bytes=0, device_bytes=0)
