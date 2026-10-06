"""Independent ownership/interleaving model, plus the actual portable host header.

Model tests are not CUDA numerical/synchronization or performance qualification.
The native header test explicitly skips without the repository-required ccache.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EMPTY = 0xFFFFFFFF
CLASSES = 55
BUCKETS = CLASSES + 1
CAPACITY = 256


def publish(classes: list[int], order: list[int]) -> tuple[list[int], list[int]]:
    """Independent model of a completed publication phase."""
    heads = [EMPTY] * BUCKETS
    links = [EMPTY] * CAPACITY
    for slot in order:
        assert 0 <= slot < len(classes) <= CAPACITY
        bucket = min(classes[slot], CLASSES)
        links[slot], heads[bucket] = heads[bucket], slot
    return heads, links


def interleaved_drain(classes: list[int], workers: int, seed: int) -> Counter[int]:
    """Interleave loads and compare-exchanges, including stale observations.

    A claim's CAS is the linearization point. links[] never changes in this
    phase. No synthetic GPU timing or floating-point result is generated.
    """
    rng = random.Random(seed)
    order = list(range(len(classes)))
    rng.shuffle(order)
    heads, links = publish(classes, order)
    # Each worker retains a class after success, exactly as a warp leader does.
    states = [
        {"bucket": i % BUCKETS, "phase": "load", "observed": EMPTY, "visited": 0}
        for i in range(workers)
    ]
    visits: Counter[int] = Counter()
    active = list(range(workers))
    steps = 0
    while active:
        worker = rng.choice(active)
        state = states[worker]
        bucket = state["bucket"]
        if state["phase"] == "load":
            state["observed"] = heads[bucket]
            state["phase"] = "check"
        elif state["phase"] == "check":
            old = state["observed"]
            if old == EMPTY:
                state["visited"] += 1
                if state["visited"] == BUCKETS:
                    active.remove(worker)
                else:
                    state["bucket"] = (bucket + 1) % BUCKETS
                    state["phase"] = "load"
            else:
                assert 0 <= old < len(classes)
                state["desired"] = links[old]
                state["phase"] = "cas"
        else:
            observed = heads[bucket]
            if observed == state["observed"]:
                heads[bucket] = state["desired"]
                assert min(classes[observed], CLASSES) == bucket
                visits[observed] += 1
                assert visits[observed] == 1
                state["visited"] = 0
                state["phase"] = "load"
            else:
                state["observed"] = observed
                state["phase"] = "check"
        steps += 1
        assert steps < 200000, "finite pop-only queue failed to terminate"
    assert heads == [EMPTY] * BUCKETS
    return visits


@pytest.mark.parametrize("count", range(CAPACITY + 1))
def test_every_packet_tail_is_owned_once(count: int) -> None:
    classes = [(i * 17 + count) % CLASSES for i in range(count)]
    visits = interleaved_drain(classes, 8, count)
    assert visits == Counter(range(count))


@pytest.mark.parametrize("workers", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("pattern", ["one-class", "all-classes", "fallback", "random"])
def test_dynamic_worker_count_and_class_skew(workers: int, pattern: str) -> None:
    rng = random.Random(37 + workers)
    if pattern == "one-class":
        classes = [20] * CAPACITY
    elif pattern == "all-classes":
        classes = [i % CLASSES for i in range(CAPACITY)]
    elif pattern == "fallback":
        classes = [55 + i % 7 for i in range(CAPACITY)]
    else:
        classes = [rng.randrange(61) for _ in range(CAPACITY)]
    assert interleaved_drain(classes, workers, 912 + workers) == Counter(
        range(CAPACITY)
    )


@pytest.mark.parametrize("seed", range(32))
def test_different_claim_interleavings(seed: int) -> None:
    rng = random.Random(seed)
    classes = [rng.choice([0, 5, 20, 21, 54, 55, 100]) for _ in range(CAPACITY)]
    assert interleaved_drain(classes, 8, seed) == Counter(range(CAPACITY))


def test_stale_compare_exchange_cannot_duplicate_a_descriptor() -> None:
    heads, links = publish([20, 20], [0, 1])
    first_read = second_read = heads[20]
    assert first_read == 1
    heads[20] = links[first_read]  # first worker wins CAS(1,0)
    assert heads[20] != second_read  # second worker must retry, not claim slot 1
    second_read = heads[20]
    heads[20] = links[second_read]
    assert {first_read, second_read} == {0, 1}
    assert heads[20] == EMPTY


def test_cuda_wrapper_has_warp_publication_and_no_domain_admission() -> None:
    wrapper = (ROOT / "src/scf/cuda/direct_warp_queue.cuh").read_text()
    assert "__syncwarp(mask)" in wrapper
    assert "__shfl_sync(mask, slot, 0)" in wrapper
    assert "atomicCAS(address, expected, desired)" in wrapper
    assert "atomicAdd(address, 0U)" in wrapper
    assert "__syncthreads()" not in wrapper
    assert "global_cursor" not in wrapper
    assert "direct_shell_quartet_survives_screening" not in wrapper


def test_portable_storage_bound_and_phase_contract() -> None:
    header = (ROOT / "src/scf/direct_warp_queue.hpp").read_text()
    assert "heads[bucket_count]" in header
    assert "next[Capacity]" in header
    assert "No floating-point operation" in header
    assert "heads only advance" in header
    assert "LIFO" in header
    assert "while (slot != empty)" in header
    assert (BUCKETS + CAPACITY) * 4 == 1248


def test_actual_cpp_queue_with_concurrent_host_threads(tmp_path: Path) -> None:
    ccache = shutil.which("ccache")
    if ccache is None:
        pytest.skip(
            "ccache unavailable; no uncached native build or CUDA qualification claimed"
        )
    cxx = shutil.which(os.environ.get("CXX", "c++"))
    if cxx is None:
        pytest.skip("host C++ compiler unavailable")
    subprocess.run([ccache, "--version"], check=True, capture_output=True, text=True)
    executable = tmp_path / "direct-warp-queue-host"
    subprocess.run(
        [
            ccache,
            cxx,
            "-std=c++17",
            "-O2",
            "-pthread",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "src"),
            str(ROOT / "tests/native/direct_warp_queue_host.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=120
    )
    assert "260 concurrent host batches passed" in result.stdout
