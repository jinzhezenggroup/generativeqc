"""Check scheduler admission without probing or allocating a real GPU."""

import pytest

from benchmarks.array_api_rhf_endpoint import _scheduler_guard


@pytest.mark.parametrize(
    "node", ("n1", "n2", "n4", "n5", "node1", "node2", "node4", "node5")
)
def test_cuda_guard_accepts_verified_node_aliases(
    node: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "test-admission-only")
    monkeypatch.setenv("SLURMD_NODENAME", node)
    _scheduler_guard("cuda")


@pytest.mark.parametrize("node", ("n3", "node3", "", "unknown"))
def test_cuda_guard_rejects_maintenance_and_unknown_nodes(
    node: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "test-admission-only")
    monkeypatch.setenv("SLURMD_NODENAME", node)
    with pytest.raises(RuntimeError, match="srun"):
        _scheduler_guard("cuda")


def test_cuda_guard_rejects_unscheduled_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURMD_NODENAME", "node2")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="srun"):
        _scheduler_guard("cuda")
    _scheduler_guard("cpu")
