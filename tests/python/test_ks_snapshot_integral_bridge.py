"""Exercise native integral-source ABI conversions without a CUDA device."""

import ctypes as ct
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc._ks_snapshot import NativeKsSnapshot


@pytest.mark.parametrize("layout", ("separate", "range", "combined"))
def test_integral_bridge_converts_the_complete_signature(layout: str) -> None:
    """Real ctypes conversion must accept the v2 selector before its output pointer.

    Mutating an already assigned argtypes list does not update ctypes' argument
    converters. A plain Python mock cannot catch that mismatch at the ABI boundary.
    """
    combined = layout == "combined"
    sources = {"separate": 4, "range": 5, "combined": 3}[layout]
    calls = []
    signature = ct.CFUNCTYPE(
        ct.c_int,
        ct.c_void_p,
        ct.c_void_p,
        *((ct.c_int,) if combined else ()),
        ct.POINTER(ct.c_double),
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(ct.c_uint64),
        ct.c_size_t,
    )

    @signature
    def evaluate(*args: Any) -> int:
        prefix = args[:-5]
        output, count, maximum, work, work_count = args[-5:]
        calls.append((prefix, count, maximum, work_count))
        for index in range(count):
            output[index] = index + 1
        for index in range(work_count):
            work[index] = index + 10
        return 0

    bridge = (
        "generativeqc_ks_snapshot_cuda_integral_gradient_v2"
        if combined
        else "generativeqc_ks_snapshot_cuda_integral_gradient_v1"
    )
    snapshot = SimpleNamespace(
        backend="cuda",
        check_current=lambda: None,
        _library=SimpleNamespace(**{bridge: evaluate}),
        _batch=SimpleNamespace(_batch=ct.c_void_p(11), _context=None),
        _handle=ct.c_void_p(22),
    )
    output, work = NativeKsSnapshot.cuda_integral_derivatives(
        snapshot,
        2,
        4096,
        range_exchange=layout == "range",
        combined_two_electron=combined,
    )
    assert calls == [((11, 22, 1) if combined else (11, 22), sources * 6, 4096, 9)]
    np.testing.assert_array_equal(
        output, np.arange(1, sources * 6 + 1).reshape(sources, 2, 3)
    )
    assert not output.flags.writeable
    assert tuple(work.values()) == tuple(range(10, 19))


def test_combined_bridge_is_optional_for_an_older_native_library() -> None:
    snapshot = SimpleNamespace(
        backend="cuda", check_current=lambda: None, _library=SimpleNamespace()
    )
    assert (
        NativeKsSnapshot.cuda_integral_derivatives(
            snapshot, 2, 4096, range_exchange=False, combined_two_electron=True
        )
        is None
    )
