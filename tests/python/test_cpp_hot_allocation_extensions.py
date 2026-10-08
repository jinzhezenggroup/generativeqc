"""CPU-only lexical allocation-role controls for native replay loops."""

from __future__ import annotations

from collections import Counter

from tools.audit_native_work import audit_native


def test_pinned_host_memory_realloc_release_and_setup_are_separate_roles() -> None:
    source = """
void replay(int n) {
  cudaMallocHost(&setup, n);
  cudaFreeHost(setup);
  for (int tile = 0; tile < n; ++tile) {
    cudaMallocHost(&pinned, n);
    cudaHostAlloc(&mapped, n, 0);
    cudaMalloc(&device, n);
    cudaMallocAsync(&async_device, n, stream);
    realloc(work, n);
    std::realloc(more_work, n);
    cudaFree(device);
    cudaFreeAsync(async_device, stream);
    cudaFreeHost(pinned);
    std::free(mapped);
  }
}
"""
    rows = audit_native(source)
    assert Counter(row["rule_id"] for row in rows) == {
        "loop-host-allocation": 2,
        "loop-device-allocation": 2,
        "loop-host-reallocation": 2,
        "loop-device-release": 2,
        "loop-host-release": 2,
    }
    assert all(row["details"]["count_kind"] == "static-site-not-runtime-count" for row in rows)
    assert all("replay" == row["function"] for row in rows)


def test_same_file_callee_retains_role_without_runtime_proof() -> None:
    source = """
void make(int n) { cudaMallocHost(&p, n); realloc(q, n); cudaFreeHost(p); }
void replay(int n) {
  for (int i = 0; i < n; ++i) make(n);
}
"""
    rows = audit_native(source)
    assert {row["rule_id"] for row in rows} == {
        "loop-callee-host-allocation",
        "loop-callee-host-reallocation",
        "loop-callee-host-release",
    }
    assert all(row["details"]["count_kind"] == "static-path-not-runtime-count" for row in rows)
    assert all(row["details"]["call_path"] for row in rows)
