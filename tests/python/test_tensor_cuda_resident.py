"""CPU proofs of the resident ABI contract emitted on top of a verified plan.

These checks need no GPU and no NVCC: they assert the generated translation
unit keeps the ordinary host-staged entry points intact, pins exactly the
planned input/output offsets, and propagates a non-zero ordinary status
instead of swallowing it. Real-device numerical parity is a separate opt-in
suite.
"""

import ctypes
import json
import os
import random
import re
import time
import typing
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.cuda_runtime import _Metrics
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    Symmetry,
    TensorSpec,
    add,
    constant,
    input_tensor,
    multiply,
)
from generativeqc_compiler.tensor.cuda_plan import ALIGNMENT, plan_cuda
from generativeqc_compiler.tensor.cuda_resident import (
    DeviceTensor,
    _check_lease,
    _resident_dependency_logical_name,
)
from generativeqc_compiler.tensor.cuda_resident_emit import resident_source
from tensor_resident_host import build_resident_host, build_resident_validation_host

TARGET = cuda_target_info("sm_80")


def doubled_pair_program() -> typing.Any:
    """Two inputs, two outputs: one elementwise and one symmmetric double."""
    occupied = IndexSpace("occupied", "occupied", 2)
    virtual = IndexSpace("virtual", "virtual", 3)
    i, j = Index("i", occupied), Index("j", occupied)
    a, b = Index("a", virtual), Index("b", virtual)
    common = {
        "representation": "restricted_spatial",
        "role": "parameter",
        "differentiable": True,
    }
    singles = TensorSpec((i, a), **common)
    doubles = TensorSpec((i, j, a, b), symmetries=(Symmetry((1, 0, 3, 2)),), **common)
    t1 = input_tensor("t1", singles)
    t2 = input_tensor("t2", doubles)
    return Program({"squared": add(multiply(t1, t1)), "copy": add(t2)})


@pytest.mark.parametrize("embedded", [True, False])
@pytest.mark.parametrize("device", [0, 2])
def test_resident_artifact_keeps_the_ordinary_abi(
    tmp_path: Path,
    required_native_cxx: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    embedded: bool,
    device: int,
) -> None:
    """Resolve and execute both generated ABIs using the same owner and spans."""
    plan = plan_cuda(_transfer_program("two"), TARGET, max_bytes=1 << 26)
    lib = build_resident_host(
        plan,
        tmp_path,
        required_native_cxx,
        monkeypatch,
        ordinary=True,
        embed_static_data=embedded,
    )
    lib.tensor_plan_identity.restype = ctypes.c_char_p
    assert lib.tensor_plan_identity().decode() == plan.identity
    lib.tensor_create.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.tensor_destroy.argtypes = [ctypes.c_void_p]
    lib.tensor_destroy.restype = None
    lib.tensor_run.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.tensor_probe.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    error = ctypes.create_string_buffer(256)
    metrics = _Metrics()
    owner = ctypes.c_void_p()
    assert lib.tensor_create(device, ctypes.byref(owner), error, len(error)) == 0
    assert lib.host_selected_device() == 0
    lib.host_clear_work_devices()
    try:
        inputs = [
            (ctypes.c_int64 * plan.steps[i].node.spec.size)(
                *(100 * slot + j for j in range(plan.steps[i].node.spec.size))
            )
            for slot, i in enumerate(plan.inputs)
        ]
        outputs = [
            (ctypes.c_int64 * plan.steps[i].node.spec.size)() for _, i in plan.outputs
        ]
        pointers = lambda arrays: (ctypes.c_void_p * len(arrays))(
            *(ctypes.addressof(array) for array in arrays)
        )

        def run() -> int:
            return lib.tensor_run(
                owner,
                pointers(inputs),
                pointers(outputs),
                0,
                ctypes.byref(metrics),
                error,
                len(error),
            )

        if embedded:
            with pytest.raises(AttributeError):
                _ = lib.tensor_static_initialize
        else:
            lib.tensor_static_bytes.restype = ctypes.c_size_t
            lib.tensor_static_initialize.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            assert lib.tensor_static_bytes() == 0  # This control plan has no payload.
            assert run() != 0 and b"not initialized" in error.value
            error.value = b""
            assert (
                lib.resident_run(owner, 0, ctypes.byref(metrics), error, len(error))
                != 0
            )
            assert b"not initialized" in error.value
            assert lib.tensor_static_initialize(owner, None, 1, error, len(error)) != 0
            assert lib.tensor_static_initialize(owner, None, 0, error, len(error)) == 0
            assert lib.tensor_static_initialize(owner, None, 0, error, len(error)) != 0
        assert run() == 0, error.value
        assert metrics.owned_device_bytes == plan.allocation_bytes
        for slot, (_, step) in enumerate(plan.outputs):
            if step in plan.inputs:
                assert list(outputs[slot]) == list(inputs[plan.inputs.index(step)])
        # Host-staged and resident calls share the actual emitted slot mapping.
        for slot, array in enumerate(inputs):
            for j in range(len(array)):
                array[j] += 7
            assert (
                lib.resident_upload(
                    owner, slot, array, ctypes.sizeof(array), error, len(error)
                )
                == 0
            )
        assert lib.resident_run(owner, 0, ctypes.byref(metrics), error, len(error)) == 0
        for slot, (_, step) in enumerate(plan.outputs):
            array = outputs[slot]
            assert (
                lib.resident_download(
                    owner, slot, array, ctypes.sizeof(array), error, len(error)
                )
                == 0
            )
            if step in plan.inputs:
                assert list(array) == list(inputs[plan.inputs.index(step)])
        assert lib.host_pending_work() == 0
        assert lib.host_selected_device() == 0
        assert lib.host_work_on_device(device)
    finally:
        lib.tensor_destroy(owner)
    assert lib.tensor_probe(0, error, len(error)) == 0
    assert json.loads(error.value)["architecture"] == "sm_80"  # Fixture response.


def _transfer_program(kind: str) -> typing.Any:
    # Control tensors have no floating validation or computation launches. Use
    # their real lowering rather than patching private emitter dependencies.
    counts = (6, 36) if kind == "two" else (0 if kind == "empty" else 3, 5, 2)
    inputs = []
    for name, count in zip(("a", "b", "c"), counts, strict=False):
        index = Index(name, IndexSpace(name, "occupied", count))
        inputs.append(
            input_tensor(
                name,
                TensorSpec(
                    (index,),
                    dtype="int64",
                    representation="general",
                    role="parameter",
                    differentiable=False,
                ),
            )
        )
    outputs = {f"out_{i}": value for i, value in enumerate(reversed(inputs))}
    # CUDA plan identity requires a floating dtype. An empty literal admits the
    # real plan without floating input validation, initialization or arithmetic.
    empty = Index("empty", IndexSpace("empty", "occupied", 0))
    outputs["empty_literal"] = constant(
        (), TensorSpec((empty,), role="constant", differentiable=False)
    )
    return Program(outputs)


@pytest.fixture(params=["two", "reordered", "empty"])
def resident_host(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    native_cxx: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
) -> typing.Any:
    """A real generated resident ABI, with only CUDA/owner emission stubbed."""
    plan = plan_cuda(_transfer_program(request.param), TARGET, max_bytes=1 << 26)
    lib = build_resident_host(plan, tmp_path, native_cxx, monkeypatch)
    return plan, lib


def _assert_resident_abi(plan: typing.Any, lib: ctypes.CDLL) -> None:
    assert lib.resident_plan_identity().decode() == plan.identity + ":resident-abi-v1"
    assert lib.resident_abi() == 1
    error = ctypes.create_string_buffer(256)
    metrics = _Metrics()
    result = ctypes.byref(metrics)
    capacity = plan.allocation_bytes
    owner = lib.host_owner(capacity, capacity - ALIGNMENT)
    lib.host_owner_device(owner, 2)
    lib.host_clear_work_devices()
    try:
        _assert_resident_busy(
            lib,
            owner,
            capacity,
            lambda: lib.resident_run(owner, 0, result, error, len(error)),
            error,
        )
        # Calls resolve every required public export with its actual C signature.
        lib.host_clear_copy_kinds()
        before = lib.host_synchronizations()
        assert lib.resident_run(owner, 0, result, error, len(error)) == 0, error.value
        assert lib.host_synchronizations() > before
        assert lib.host_pending_work() == 0
        assert lib.host_copy_matches(0)
        assert lib.host_copy_covers(ctypes.sizeof(ctypes.c_int))
        for prepare, live in ((211, 307), (503, 401)):
            lib.host_metrics(owner, 109, prepare, live)
            assert lib.resident_run(owner, 0, result, error, len(error)) == 0
            assert metrics.owned_device_bytes == capacity
            assert metrics.provider_retained_bytes == 109
            assert metrics.prepare_device_delta == prepare
            assert metrics.observed_device_delta == max(prepare, live)
        for marker in (-3, 5):
            lib.host_status_marker(marker)
            error.value = b""
            before = lib.host_synchronizations()
            try:
                assert lib.resident_run(owner, 0, result, error, len(error)) != 0
                assert error.value
                assert lib.host_synchronizations() > before
                assert lib.host_pending_work() == 0
            finally:
                lib.host_status_marker(0)
            error.value = b""
            assert lib.resident_run(owner, 0, result, error, len(error)) == 0
        for failure in (1, 2, 3):
            lib.host_failure(failure)
            error.value = b""
            before = lib.host_synchronizations()
            try:
                assert lib.resident_run(owner, 0, result, error, len(error)) != 0
                assert b"injected" in error.value
                assert lib.host_synchronizations() > before
                assert lib.host_pending_work() == 0
            finally:
                lib.host_failure(0)
            assert lib.host_selected_device() == 0
        for pointer, metrics in ((None, result), (owner, None)):
            error.value = b""
            assert lib.resident_run(pointer, 0, metrics, error, len(error)) != 0
            assert error.value
        lib.host_static_ready(owner, 0)
        error.value = b""
        before = lib.host_synchronizations()
        assert lib.resident_run(owner, 0, result, error, len(error)) != 0
        assert b"not initialized" in error.value
        assert lib.host_synchronizations() > before
        assert lib.host_work_on_device(2), "error drains must use the owner's device"
        assert lib.host_selected_device() == 0
    finally:
        lib.host_destroy(owner)


def test_resident_source_declares_the_resident_abi(resident_host: typing.Any) -> None:
    _assert_resident_abi(*resident_host)


def test_resident_selection_failure_does_not_touch_owner(
    resident_host: typing.Any,
) -> None:
    """Selection failure admits no transfers, launches or stream cleanup."""
    plan, lib = resident_host
    owner = lib.host_owner(plan.allocation_bytes, plan.allocation_bytes - ALIGNMENT)
    lib.host_owner_device(owner, 2)
    error, metrics = ctypes.create_string_buffer(256), _Metrics()
    before = lib.host_synchronizations(), lib.host_work_sequence()
    lib.host_fail_device(2)
    try:
        assert lib.resident_run(owner, 0, ctypes.byref(metrics), error, len(error)) != 0
        assert b"selection" in error.value
        assert (lib.host_synchronizations(), lib.host_work_sequence()) == before
        assert lib.host_selected_device() == 0
    finally:
        lib.host_fail_device(-1)
        lib.host_destroy(owner)


def _assert_resident_busy(
    lib: ctypes.CDLL,
    owner: typing.Any,
    capacity: int,
    operation: typing.Callable[[], int],
    error: typing.Any,
    host: typing.Any = None,
) -> None:
    """Reject without admitted work while another real host thread owns the mutex."""
    arena = lib.host_arena(owner)
    before = ctypes.string_at(arena, capacity)
    host_before = None if host is None else host.raw
    sequence, pending = lib.host_work_sequence(), lib.host_pending_work()
    holder = lib.host_hold_mutex(owner)
    assert holder, "finite owner-mutex handshake failed"
    try:
        assert lib.host_mutex_held(holder)
        error.value = b""
        start = time.monotonic()
        status = operation()
        elapsed = time.monotonic() - start
        assert elapsed < 3.0, "busy admission blocked instead of returning promptly"
        assert status != 0
        assert b"busy" in error.value
        assert ctypes.string_at(arena, capacity) == before
        assert host is None or host.raw == host_before
        assert lib.host_work_sequence() == sequence
        assert lib.host_pending_work() == pending
    finally:
        lib.host_release_mutex(holder)


def _assert_resident_spans(plan: typing.Any, lib: ctypes.CDLL) -> None:
    assert all(plan.steps[i].last_use == len(plan.steps) for i in plan.inputs)
    slots = (
        [plan.steps[i] for i in plan.inputs],
        [plan.steps[i] for _, i in plan.outputs],
    )
    capacity = plan.allocation_bytes
    error_offset = capacity - ALIGNMENT
    owner = lib.host_owner(capacity, error_offset)
    lib.host_owner_device(owner, 2)
    lib.host_clear_work_devices()
    error = ctypes.create_string_buffer(256)
    arena = lib.host_arena(owner)
    try:
        expected = bytearray([0xA5] * capacity)
        expected[error_offset : error_offset + ctypes.sizeof(ctypes.c_int)] = bytes(
            ctypes.c_int(0x01010101)
        )
        for slot, step in enumerate(slots[0]):
            size = step.node.spec.size * step.node.spec.itemsize
            payload = bytes((j * 17 + slot * 43) % 256 for j in range(size))
            host = ctypes.create_string_buffer(payload) if size else None
            if slot == 0:
                _assert_resident_busy(
                    lib,
                    owner,
                    capacity,
                    lambda slot=slot, host=host, size=size: lib.resident_upload(
                        owner, slot, host, size, error, len(error)
                    ),
                    error,
                    host,
                )
            lib.host_clear_copy_kinds()
            before = lib.host_synchronizations()
            assert (
                lib.resident_upload(owner, slot, host, size, error, len(error)) == 0
            ), error.value
            assert lib.host_synchronizations() > before
            assert lib.host_pending_work() == 0
            if size:
                assert lib.host_copy_matches(1)
            expected[step.offset : step.offset + size] = payload
            assert ctypes.string_at(arena, capacity) == expected
            if not size:
                guard = ctypes.create_string_buffer(b"\xcc", 1)
                assert (
                    lib.resident_upload(owner, slot, guard, 0, error, len(error)) == 0
                )
                assert guard.raw == b"\xcc"
                assert ctypes.string_at(arena, capacity) == expected

        # Only the reserved status int may change during this kernel-free run.
        metrics = _Metrics()
        assert (
            lib.resident_run(owner, 0, ctypes.byref(metrics), error, len(error)) == 0
        ), error.value
        expected[error_offset : error_offset + ctypes.sizeof(ctypes.c_int)] = bytes(
            ctypes.c_int(0)
        )
        assert ctypes.string_at(arena, capacity) == expected
        assert lib.host_pending_work() == 0

        # Seed the whole arena independently: downloads cannot accidentally pass
        # by undoing an equally incorrect upload mapping.
        raw = random.Random(2296).randbytes(capacity)
        # Require distinguishable windows for every nonempty output length.
        lengths = {
            s.node.spec.size * s.node.spec.itemsize
            for s in slots[1]
            if s.node.spec.size
        }
        for length in lengths:
            assert (
                len({raw[j : j + length] for j in range(capacity - length + 1)})
                == capacity - length + 1
            )
        ctypes.memmove(arena, raw, capacity)
        for slot, step in enumerate(slots[1]):
            size = step.node.spec.size * step.node.spec.itemsize
            host = ctypes.create_string_buffer(size) if size else None
            if slot == 0:
                _assert_resident_busy(
                    lib,
                    owner,
                    capacity,
                    lambda slot=slot, host=host, size=size: lib.resident_download(
                        owner, slot, host, size, error, len(error)
                    ),
                    error,
                    host,
                )
            lib.host_clear_copy_kinds()
            before = lib.host_synchronizations()
            assert (
                lib.resident_download(owner, slot, host, size, error, len(error)) == 0
            ), error.value
            assert lib.host_synchronizations() > before
            assert lib.host_pending_work() == 0
            if size:
                assert lib.host_copy_matches(0)
                assert host.raw == raw[step.offset : step.offset + size]
            assert ctypes.string_at(arena, capacity) == raw
            if not size:
                guard = ctypes.create_string_buffer(b"\xcc", 1)
                assert (
                    lib.resident_download(owner, slot, guard, 0, error, len(error)) == 0
                )
                assert guard.raw == b"\xcc"
                assert ctypes.string_at(arena, capacity) == raw

        for transfer, steps in zip(
            (lib.resident_upload, lib.resident_download), slots, strict=True
        ):
            step = max((s for s in steps if s.node.spec.size), key=lambda s: s.offset)
            slot = steps.index(step)
            size = step.node.spec.size * step.node.spec.itemsize
            host = ctypes.create_string_buffer(bytes([0xCC] * (size + 1)))
            cases = (
                (None, slot, host, size),
                (owner, len(steps), host, size),
                (owner, slot, host, size + 1),
                (owner, slot, host, size - 1),
                (owner, slot, None, size),
                (owner, slot, None, 0),
                (owner, len(steps), None, 0),
                (None, slot, None, 0),
                (
                    owner,
                    slot + (1 << (ctypes.sizeof(ctypes.c_size_t) * 8 - 1)),
                    host,
                    size,
                ),
                (
                    owner,
                    slot,
                    host,
                    size + (1 << (ctypes.sizeof(ctypes.c_size_t) * 8 - 1)),
                ),
            )
            for args in cases:
                error.value = b""
                assert transfer(*args, error, len(error)) != 0
                assert error.value
                assert ctypes.string_at(arena, capacity) == raw
                assert host.raw == bytes([0xCC] * (size + 1)) + b"\0"
            # Both bounds checks: offset outside arena; extent beyond arena.
            for limit in (max(0, step.offset - 1), step.offset + size - 1):
                lib.host_capacity(owner, limit)
                error.value = b""
                before = lib.host_synchronizations()
                assert transfer(owner, slot, host, size, error, len(error)) != 0
                assert error.value
                assert lib.host_synchronizations() > before
                assert ctypes.string_at(arena, capacity) == raw
            lib.host_capacity(owner, capacity)
            for failure in (1, 2, 3):
                lib.host_failure(failure)
                error.value = b""
                before = lib.host_synchronizations()
                try:
                    assert transfer(owner, slot, host, size, error, len(error)) != 0
                    assert b"injected" in error.value
                    assert lib.host_synchronizations() > before
                    assert ctypes.string_at(arena, capacity) == raw
                    assert host.raw == bytes([0xCC] * (size + 1)) + b"\0"
                finally:
                    lib.host_failure(0)
                assert lib.host_selected_device() == 0
        assert lib.host_work_on_device(2)
    finally:
        lib.host_destroy(owner)


def test_resident_copy_selects_owner_and_rejects_cross_device(
    tmp_path: Path, native_cxx: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same-device copies restore the caller; cross-device copies admit no work."""
    plan = plan_cuda(_transfer_program("two"), TARGET)
    lib = build_resident_host(
        plan,
        tmp_path,
        native_cxx,
        monkeypatch,
        extension=r"""
extern "C" int test_copy(void* src, void* dst, size_t bytes, char* error, size_t size) {
    using namespace generativeqc_resident;
    return copy(src, resident_inputs, 2, 0, dst, resident_inputs, 2, 0, bytes, error, size);
}
""",
    )
    lib.test_copy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    src = lib.host_owner(plan.allocation_bytes, plan.allocation_bytes - ALIGNMENT)
    dst = lib.host_owner(plan.allocation_bytes, plan.allocation_bytes - ALIGNMENT)
    size = plan.steps[plan.inputs[0]].node.spec.size * 8
    payload = ctypes.create_string_buffer(bytes(i % 256 for i in range(size)))
    error = ctypes.create_string_buffer(256)
    try:
        for owner in (src, dst):
            lib.host_owner_device(owner, 2)
        lib.host_clear_work_devices()
        assert lib.resident_upload(src, 0, payload, size, error, len(error)) == 0
        assert lib.test_copy(src, dst, size, error, len(error)) == 0, error.value
        offset = plan.steps[plan.inputs[0]].offset
        assert (
            ctypes.string_at(lib.host_arena(dst) + offset, size) == payload.raw[:size]
        )
        assert lib.host_work_on_device(2) and lib.host_selected_device() == 0
        for bad_device in (0, 1):
            lib.host_owner_device(src, bad_device)
            before = lib.host_work_sequence(), lib.host_synchronizations()
            assert lib.test_copy(src, dst, size, error, len(error)) != 0
            assert b"device mismatch" in error.value
            assert (lib.host_work_sequence(), lib.host_synchronizations()) == before
            assert lib.host_selected_device() == 0
    finally:
        lib.host_destroy(src)
        lib.host_destroy(dst)


def test_resident_spans_match_the_pinned_plan_offsets() -> None:
    """Retain floating/computed span coverage until a matching executable gate.

    The host control-plan transfer oracle cannot establish the storage contract
    of computed FP64 outputs. These planner/span assertions remain scoped to
    that lifetime contract, independently of the int64 ABI tests.
    """
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    assert all(plan.steps[i].last_use == len(plan.steps) for i in plan.inputs)
    source = resident_source(plan)
    for i in (*plan.inputs, *(i for _, i in plan.outputs)):
        step = plan.steps[i]
        assert f"{{{step.offset}ULL,{step.node.spec.size * 8}ULL}}" in source


def test_resident_control_span_transfers(
    resident_host: typing.Any,
) -> None:
    """Execute generated int64 control span mappings against raw host memory."""
    _assert_resident_spans(*resident_host)


@pytest.mark.skipif(
    os.environ.get("VIBEQC_RESIDENT_HOST_CONTROLS") != "1",
    reason="opt-in generated-source metamorphic/mutation controls",
)
@pytest.mark.parametrize(
    "mutation",
    [
        "format",
        "split_status",
        "arena_status",
        "wrong_status_offset",
        "ignored_arithmetic_status",
        "wrong_status_host",
        "provider_metric",
        "prepare_metric",
        "observed_prepare_only",
        "observed_live_only",
        "abi",
        "diagnostic",
        "run_sync",
        "run_kind",
        "run_status",
        "zero_status_bytes",
        "short_status_bytes",
        "begin_record",
        "status_reset",
        "late_reset",
        "narrow_slot",
        "narrow_bytes",
        "output_offset",
        "run_busy",
        "transfer_busy",
        "blocking_busy",
        "status_target",
        "run_endpoints",
        "event_position",
        "early_sync",
        "kind",
        "offset",
        "length",
        "status",
    ],
)
def test_resident_host_oracle_controls(
    mutation: str,
    tmp_path: Path,
    native_cxx: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Harmless spelling survives; actual ABI/transfer damage is detected."""
    plan = plan_cuda(_transfer_program("reordered"), TARGET, max_bytes=1 << 26)

    def transform(source: str) -> str:
        if mutation in ("arena_status", "wrong_status_offset"):
            offset = plan.allocation_bytes - ALIGNMENT
            if mutation == "wrong_status_offset":
                offset += ctypes.sizeof(ctypes.c_int)
            return source.replace(
                "ctx.error", f"reinterpret_cast<int*>(p + {offset}ULL)"
            )
        if mutation == "ignored_arithmetic_status":
            return source.replace("if (arithmetic_error)", "if (false)")
        if mutation == "wrong_status_host":
            return source.replace(
                "int arithmetic_error = 0;", "int arithmetic_error = 0, unrelated = 0;"
            ).replace(
                "cudaMemcpyAsync(&arithmetic_error,", "cudaMemcpyAsync(&unrelated,"
            )
        if mutation in ("provider_metric", "prepare_metric"):
            field = (
                "provider_retained_bytes"
                if mutation == "provider_metric"
                else "prepare_device_delta"
            )
            return source.replace(
                f"metrics.{field} = ctx.metrics.{field};", f"metrics.{field} = 0;"
            )
        if mutation in ("observed_prepare_only", "observed_live_only"):
            value = (
                "ctx.metrics.prepare_device_delta"
                if mutation == "observed_prepare_only"
                else "ctx.device_delta()"
            )
            return re.sub(
                r"std::max\(ctx.metrics.prepare_device_delta,\s*ctx.device_delta\(\)\)",
                value,
                source,
            )
        if mutation == "format":
            return (
                source.replace("resident_inputs", "input_spans")
                .replace("resident_outputs", "output_spans")
                .replace("ULL,", "ULL,\n  ")
            )
        if mutation == "split_status":
            return source.replace(
                "cuda_check(cudaMemcpyAsync(&arithmetic_error, ctx.error, sizeof(int),\n"
                "                                   cudaMemcpyDeviceToHost, ctx.stream));",
                "cuda_check(cudaMemcpyAsync(&arithmetic_error, ctx.error, sizeof(int)/2,"
                "cudaMemcpyDeviceToHost,ctx.stream));"
                "cuda_check(cudaMemcpyAsync(reinterpret_cast<unsigned char*>(&arithmetic_error)+sizeof(int)/2,"
                "reinterpret_cast<unsigned char*>(ctx.error)+sizeof(int)/2,sizeof(int)-sizeof(int)/2,"
                "cudaMemcpyDeviceToHost,ctx.stream));",
            )
        if mutation == "status":
            return source.replace(
                "return generativeqc_resident::transfer(",
                "return 0 * generativeqc_resident::transfer(",
            )
        if mutation == "abi":
            return source.replace(
                'extern "C" int resident_abi() { return 1; }',
                'extern "C" int resident_abi() { return 2; }',
            )
        if mutation == "diagnostic":
            return source.replace(
                "generativeqc_tensor::error_text(error, size, e.what()); return 1;",
                "return 1;",
            )
        if mutation == "run_sync":
            return source.replace("cuda_check(cudaEventSynchronize(ctx.end));", "")
        if mutation == "run_kind":
            return source.replace(
                "cudaMemcpyDeviceToHost, ctx.stream",
                "cudaMemcpyHostToDevice, ctx.stream",
            )
        if mutation == "run_status":
            return source.replace(
                "cuda_check(cudaMemcpyAsync(&arithmetic_error, ctx.error, sizeof(int),\n"
                "                                   cudaMemcpyDeviceToHost, ctx.stream));",
                "cudaMemcpyAsync(&arithmetic_error, ctx.error, sizeof(int),\n"
                "                                   cudaMemcpyDeviceToHost, ctx.stream);",
            )
        if mutation == "status_target":
            return source.replace(
                "cudaMemsetAsync(ctx.error, 0, sizeof(int), ctx.stream)",
                "cudaMemsetAsync(p, 0, sizeof(int), ctx.stream)",
            ).replace(
                "cudaMemcpyAsync(&arithmetic_error, ctx.error, sizeof(int),",
                "cudaMemcpyAsync(&arithmetic_error, p, sizeof(int),",
            )
        if mutation in ("zero_status_bytes", "short_status_bytes"):
            count = "0" if mutation == "zero_status_bytes" else "sizeof(int) - 1"
            return source.replace(
                "cudaMemcpyAsync(&arithmetic_error, ctx.error, sizeof(int),",
                f"cudaMemcpyAsync(&arithmetic_error, ctx.error, {count},",
            )
        if mutation == "begin_record":
            return source.replace(
                "cuda_check(cudaEventRecord(ctx.begin, ctx.stream));", ""
            )
        if mutation == "status_reset":
            return source.replace(
                "cuda_check(cudaMemsetAsync(ctx.error, 0, sizeof(int), ctx.stream));",
                "",
            )
        if mutation == "late_reset":
            return source.replace(
                "cuda_check(cudaMemsetAsync(ctx.error, 0, sizeof(int), ctx.stream));",
                "",
            ).replace(
                "cuda_check(cudaEventRecord(ctx.end, ctx.stream));",
                "cuda_check(cudaMemsetAsync(ctx.error, 0, sizeof(int), ctx.stream));"
                "cuda_check(cudaEventRecord(ctx.end, ctx.stream));",
            )
        if mutation == "narrow_slot":
            return source.replace("size_t slot", "unsigned int slot")
        if mutation == "narrow_bytes":
            return source.replace("size_t bytes,", "unsigned int bytes,")
        if mutation == "output_offset":
            steps = [plan.steps[i] for i in plan.inputs]
            outputs = [plan.steps[i] for _, i in plan.outputs]
            capacity = (
                max(
                    s.offset + s.node.spec.size * s.node.spec.itemsize
                    for s in steps + outputs
                )
                + 64
            )
            target = next(
                s
                for s in outputs
                if s.node.spec.size
                and s.offset + 256 + s.node.spec.size * s.node.spec.itemsize <= capacity
            )
            size = target.node.spec.size * target.node.spec.itemsize
            before, marker, after = source.partition(
                "static const Span resident_outputs[] ="
            )
            return (
                before
                + marker
                + after.replace(
                    f"{{{target.offset}ULL,{size}ULL}}",
                    f"{{{target.offset + 256}ULL,{size}ULL}}",
                    1,
                )
            )
        if mutation == "run_endpoints":
            return source.replace(
                "cudaMemcpyAsync(&arithmetic_error, ctx.error, sizeof(int),",
                "cudaMemcpyAsync(ctx.error, &arithmetic_error, sizeof(int),",
            )
        if mutation == "event_position":
            return source.replace(
                "cuda_check(cudaEventRecord(ctx.end, ctx.stream));", ""
            ).replace(
                "int arithmetic_error = 0;",
                "cuda_check(cudaEventRecord(ctx.end, ctx.stream)); int arithmetic_error = 0;",
            )
        if mutation == "early_sync":
            return source.replace(
                "cuda_check(cudaEventSynchronize(ctx.end));", ""
            ).replace(
                "int arithmetic_error = 0;",
                "cuda_check(cudaEventSynchronize(ctx.end)); int arithmetic_error = 0;",
            )
        if mutation == "kind":
            return source
        if mutation == "run_busy":
            return source.replace("if (!lock.owns_lock()) {", "if (false) {")
        if mutation == "blocking_busy":
            return source.replace("ctx.mutex, std::try_to_lock", "ctx.mutex")
        if mutation == "transfer_busy":
            return source
        field = 0 if mutation == "offset" else 1

        def corrupt(match: re.Match[str]) -> str:
            values = [int(match[1]), int(match[2])]
            values[field] += 1
            return f"{{{values[0]}ULL,{values[1]}ULL}}"

        return re.sub(r"\{(\d+)ULL,(\d+)ULL\}", corrupt, source, count=1)

    def wrong_kind(header: str) -> str:
        if mutation == "transfer_busy":
            return header.replace("if (!lock.owns_lock()) {", "if (false) {")
        return header.replace(
            "upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost",
            "upload ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice",
        )

    lib = build_resident_host(
        plan,
        tmp_path,
        native_cxx,
        monkeypatch,
        transform,
        header_transform=wrong_kind if mutation in ("kind", "transfer_busy") else None,
    )
    if mutation in (
        "abi",
        "diagnostic",
        "run_sync",
        "run_kind",
        "run_status",
        "zero_status_bytes",
        "short_status_bytes",
        "begin_record",
        "status_reset",
        "late_reset",
        "run_endpoints",
        "event_position",
        "early_sync",
        "run_busy",
        "blocking_busy",
        "status_target",
        "wrong_status_offset",
        "ignored_arithmetic_status",
        "wrong_status_host",
        "provider_metric",
        "prepare_metric",
        "observed_prepare_only",
        "observed_live_only",
    ):
        with pytest.raises(AssertionError):
            _assert_resident_abi(plan, lib)
        return
    _assert_resident_abi(plan, lib)
    if mutation in ("format", "split_status", "arena_status"):
        _assert_resident_spans(plan, lib)
    else:
        with pytest.raises(AssertionError):
            _assert_resident_spans(plan, lib)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_resident_validation_covers_every_input_symmetry(
    tmp_path: Path,
    required_native_cxx: typing.Any,
    dtype: str,
) -> None:
    """Independent array permutations check every input, sign and tolerance."""
    # Include a non-square shape, multiple symmetries on one input and a
    # negative sign, so checking only one slot/permutation cannot pass.
    a = Index("a", IndexSpace("a", "occupied", 2))
    b = Index("b", IndexSpace("b", "virtual", 3))
    c, d = Index("c", a.space), Index("d", b.space)
    specs = (
        TensorSpec((a, b), dtype=dtype, role="input"),
        TensorSpec(
            (a, b, c, d),
            dtype=dtype,
            role="input",
            symmetries=(Symmetry((2, 1, 0, 3)), Symmetry((0, 3, 2, 1))),
        ),
        TensorSpec(
            (b, d), dtype=dtype, role="input", symmetries=(Symmetry((1, 0), -1),)
        ),
    )
    program = Program(
        {f"out_{i}": input_tensor(f"input_{i}", spec) for i, spec in enumerate(specs)}
    )
    plan = plan_cuda(program, TARGET, max_bytes=1 << 26)
    lib = build_resident_validation_host(plan, tmp_path, required_native_cxx)
    arena = ctypes.create_string_buffer(plan.allocation_bytes)
    inputs = []
    for i in plan.inputs:
        step = plan.steps[i]
        values = np.arange(1, step.node.spec.size + 1, dtype=dtype).reshape(
            step.node.spec.shape
        )
        for symmetry in step.node.spec.symmetries:
            values = (
                values + symmetry.sign * values.transpose(symmetry.permutation)
            ) / 2
        inputs.append(values)

    def validate(arrays: list[np.ndarray]) -> int:
        for i, values in zip(plan.inputs, arrays, strict=True):
            ctypes.memmove(
                ctypes.addressof(arena) + plan.steps[i].offset,
                values.ctypes.data,
                values.nbytes,
            )
        return lib.validate(arena)

    assert validate(inputs) == 0
    for slot, i in enumerate(plan.inputs):
        for invalid in (np.nan, np.inf, -np.inf):
            damaged = [values.copy() for values in inputs]
            damaged[slot].flat[-1] = invalid
            assert validate(damaged) != 0, (slot, invalid)
        spec = plan.steps[i].node.spec
        if not spec.symmetries:
            continue
        # Test absolute and relative acceptance independently of the emitter's
        # tolerance helper. These are the public FP32/FP64 validation contracts.
        atol, rtol = (1e-6, 1e-5) if dtype == "float32" else (1e-11, 1e-10)
        for symmetry in spec.symmetries:
            for scale in (0.0, 100.0):
                base = [values * scale for values in inputs]
                # Violate only the selected permutation: a second constraint
                # must not conceal an omitted check for the first one.
                index = next(
                    index
                    for index in np.ndindex(spec.shape)
                    if tuple(index[axis] for axis in symmetry.permutation) != index
                    and all(
                        tuple(index[axis] for axis in other.permutation) == index
                        for other in spec.symmetries
                        if other != symmetry
                    )
                )
                partner = tuple(index[axis] for axis in symmetry.permutation)
                tolerance = atol + rtol * abs(base[slot][partner])
                for factor, accepted in ((0.25, True), (4.0, False)):
                    damaged = [values.copy() for values in base]
                    damaged[slot][index] += factor * tolerance
                    assert (validate(damaged) == 0) == accepted, (
                        slot,
                        symmetry,
                        scale,
                        factor,
                    )
        assert validate(inputs) == 0  # Rejection does not poison the next validation.


def test_resident_requires_pinned_materialized_outputs() -> None:
    """Resident spans are only valid while the planner pins outputs.

    This is the invariant the whole design rests on: every output step must
    be materialized and live to the end of the program, so its arena offset
    is stable and can be re-used by a later run. ``resident_source`` rejects
    a plan that violates it instead of emitting a stale span.
    """
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    for _, i in plan.outputs:
        step = plan.steps[i]
        assert not step.virtual
        assert step.last_use == len(plan.steps)
    # The emitter agrees with the planner rather than re-deriving it.
    assert resident_source(plan)


def test_resident_source_rejects_an_unpinned_output() -> None:
    """The emitter must fail closed when an output is not pinned for the
    full plan lifetime — this is the emitter's own check, not the planner's.
    A future planner change that recycles an output's arena offset must be
    caught here, before a compiled resident binary reads overwritten storage."""
    from types import SimpleNamespace

    import pytest as _pytest

    class _MockStep:
        virtual = False
        last_use = 0  # not pinned

    plan = SimpleNamespace(
        steps=[_MockStep],
        inputs=[],
        outputs=[("unpinned", 0)],
    )
    with _pytest.raises(ValueError, match="pinned"):
        resident_source(plan)


def test_resident_extension_is_part_of_the_generated_source() -> None:
    """Any plan-specific post-run action must be visible in the identity."""
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    clean = resident_source(plan)
    extended = resident_source(plan, extension="// plan-specific tables\n")
    assert clean != extended
    assert "// plan-specific tables" in extended


@pytest.mark.parametrize("declared", [False, True])
def test_resident_extension_post_run_contract(
    tmp_path: Path,
    required_native_cxx: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    declared: bool,
) -> None:
    """Only admitted hooks run, after success, and their failures reach the caller."""
    plan = plan_cuda(_transfer_program("two"), TARGET, max_bytes=1 << 26)
    extension = r"""
extern "C" {
int hook_calls = 0, hook_status = 0, hook_profile = -1;
void* hook_owner = nullptr;
Metrics* hook_metrics = nullptr;
}
int generativeqc_resident_post_run(void* owner, int profile, Metrics* metrics, char* error, size_t size) {
  ++hook_calls;
  hook_profile = profile;
  hook_owner = owner;
  hook_metrics = metrics;
  if (hook_status == 19) throw std::runtime_error("hook exception");
  if (hook_status) generativeqc_tensor::error_text(error, size, "hook status");
  return hook_status;
}
"""
    if declared:
        extension = "// __GENERATIVEQC_RESIDENT_POST_RUN_DECL__\n" + extension
    lib = build_resident_host(
        plan, tmp_path, required_native_cxx, monkeypatch, extension=extension
    )
    calls = ctypes.c_int.in_dll(lib, "hook_calls")
    status = ctypes.c_int.in_dll(lib, "hook_status")
    profile = ctypes.c_int.in_dll(lib, "hook_profile")
    owner = lib.host_owner(plan.allocation_bytes, plan.allocation_bytes - ALIGNMENT)
    error = ctypes.create_string_buffer(256)
    metrics = _Metrics()
    try:
        for code in (0, 37, -5, 19, 0):
            status.value = code
            before = calls.value
            error.value = b""
            result = lib.resident_run(
                owner, 1, ctypes.byref(metrics), error, len(error)
            )
            assert result == ((1 if code == 19 else code) if declared else 0)
            assert calls.value - before == int(declared)
            if declared:
                assert profile.value == 1
                assert ctypes.c_void_p.in_dll(lib, "hook_owner").value == owner
                assert ctypes.c_void_p.in_dll(
                    lib, "hook_metrics"
                ).value == ctypes.addressof(metrics)
                if code:
                    assert b"hook" in error.value
            assert lib.host_pending_work() == 0
        for marker in (-3, 5):
            before = calls.value
            error.value = b""
            lib.host_status_marker(marker)
            assert (
                lib.resident_run(owner, 0, ctypes.byref(metrics), error, len(error))
                != 0
            )
            assert calls.value == before
            assert error.value
            assert lib.host_pending_work() == 0
            lib.host_status_marker(0)
    finally:
        lib.host_status_marker(0)
        lib.host_destroy(owner)


# ── lease invalidation and identity normalisation regressions ──


def test_lease_is_invalidated_by_generation_or_readiness() -> None:
    """A DeviceTensor lease must be rejected after its owner's generation
    advances or readiness is cleared — otherwise a stale download could
    read arena data that was already overwritten by a later run()."""
    import pytest as _pytest

    class _FakeOwner:
        _pointer = object()
        _ready = True
        _generation = 0

    owner = _FakeOwner()
    owner.plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    # generation mismatch

    lease = DeviceTensor(owner, "squared", 99)
    with _pytest.raises(RuntimeError, match="stale"):
        lease._step()
    # readiness cleared
    owner._generation = 1
    lease2 = DeviceTensor(owner, "squared", 1)
    lease2._step()  # ok
    owner._ready = False
    with _pytest.raises(RuntimeError, match="stale|closed"):
        lease2._step()


def test_download_before_run_or_after_invalidation_is_rejected() -> None:
    """``download(None, name=...)`` and stale-lease downloads must both be
    rejected — the arena is not safe to read before ``run()`` succeeds or
    after a later ``upload``/failed run invalidated ``_ready``."""
    import pytest as _pytest

    class _FakeOwner:
        _pointer = object()
        _ready = False
        _generation = 0
        _lock = None  # unused by the readiness path we test

    owner = _FakeOwner()
    owner.plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)

    # download(None, name=...) before run: _ready check in download must fire
    # before _lock is acquired — verify at _check_lease level
    with _pytest.raises(RuntimeError, match="stale"):
        _check_lease(owner, DeviceTensor(owner, "squared", 99))  # generation mismatch
    owner._ready = True
    _check_lease(owner, DeviceTensor(owner, "squared", 0))  # ok
    owner._ready = False
    with _pytest.raises(RuntimeError, match="stale|closed"):
        _check_lease(owner, DeviceTensor(owner, "squared", 0))
    # download(None, ...) rejects None via TypeError
    with _pytest.raises(TypeError, match="DeviceTensor"):
        _check_lease(owner, None)


def test_resident_dependency_keys_agree_for_checkout_and_installed_wheel(
    tmp_path: Path,
) -> None:
    """Actual owned paths get the same cache key regardless of install prefix."""
    checkout = tmp_path / "checkout"
    package = tmp_path / "wheel" / "generativeqc_compiler"
    for checkout_name, installed_name, expected in (
        (
            "python/generativeqc_compiler/tensor/cuda_resident_emit.py",
            "tensor/cuda_resident_emit.py",
            "python/generativeqc_compiler/tensor/cuda_resident_emit.py",
        ),
        (
            "src/tensor/cuda_resident.cuh",
            "assets/src/tensor/cuda_resident.cuh",
            "src/tensor/cuda_resident.cuh",
        ),
    ):
        original = checkout / checkout_name
        installed = package / installed_name
        for path in (original, installed):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("owned dependency")
        assert (
            _resident_dependency_logical_name(
                original, package_root=package, checkout_root=checkout
            )
            == expected
        )
        assert (
            _resident_dependency_logical_name(
                installed, package_root=package, checkout_root=checkout
            )
            == expected
        )


def test_resident_dependency_names_match_real_checkout_or_wheel_assets() -> None:
    """The identity test runs on a CPU host; no NVCC or GPU is needed."""
    from generativeqc_compiler.common.paths import PACKAGE, asset_path

    assert (
        _resident_dependency_logical_name(PACKAGE / "tensor/cuda_resident.py")
        == "python/generativeqc_compiler/tensor/cuda_resident.py"
    )
    assert (
        _resident_dependency_logical_name(asset_path("src/tensor/cuda_resident.cuh"))
        == "src/tensor/cuda_resident.cuh"
    )


def test_compile_resident_source_identity_rejects_external_dependencies(
    tmp_path: Path,
) -> None:
    """Refuse paths outside either owner, including prefixes and symlinks.

    Previously this test only inspected the spelling of compile_resident.
    Invoke its actual key-building policy so accidental weakening is detected.
    """
    checkout = tmp_path / "checkout"
    package = tmp_path / "wheel" / "generativeqc_compiler"
    checkout.mkdir()
    package.mkdir(parents=True)
    external = tmp_path / "checkout-sibling" / "unowned.hpp"
    external.parent.mkdir()
    external.write_text("external")
    kwargs = {"package_root": package, "checkout_root": checkout}
    for path in (
        external,
        checkout / ".." / "checkout-sibling" / "unowned.hpp",
    ):
        with pytest.raises(ValueError, match="outside both"):
            _resident_dependency_logical_name(path, **kwargs)

    escape = checkout / "escape.hpp"
    escape.symlink_to(external)
    with pytest.raises(ValueError, match="outside both"):
        _resident_dependency_logical_name(escape, **kwargs)


def test_installed_resident_rejects_unowned_dependency_without_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.common import paths

    package = tmp_path / "wheel" / "generativeqc_compiler"
    package.mkdir(parents=True)
    external = tmp_path / "external.hpp"
    external.write_text("external")

    def unavailable() -> Path:
        raise ValueError("no source checkout")

    monkeypatch.setattr(paths, "source_root", unavailable)
    with pytest.raises(ValueError, match="outside both"):
        _resident_dependency_logical_name(external, package_root=package)
