"""Optional VV10 work receipts preserve unavailable and measured-zero states."""

from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import _native
from generativeqc.ks_diagnostics import _nonlocal_work


def test_nonlocal_work_keeps_exact_reuse_zeros() -> None:
    def query(handle: int, index: int, values: Any, count: int) -> int:
        assert (handle, index, count) == (12, 3, 6)
        values[:] = (1, 0, 1, 1, 1, 0)
        return _native.STATUS_SUCCESS

    library = SimpleNamespace(generativeqc_internal_batch_get_nonlocal_work_v1=query)
    assert _nonlocal_work(library, 12, 3) == {
        "provisional_focks": 0,
        "target_focks": 1,
        "pair_cache_probes": 1,
        "pair_cache_hits": 1,
        "pair_sweeps": 0,
    }


def test_nonlocal_work_does_not_invent_old_library_evidence() -> None:
    assert _nonlocal_work(SimpleNamespace(), 12, 3) is None
    library = SimpleNamespace(
        generativeqc_internal_batch_get_nonlocal_work_v1=lambda *_: (
            _native.STATUS_NOT_IMPLEMENTED
        )
    )
    assert _nonlocal_work(library, 12, 3) is None


def test_nonlocal_work_rejects_malformed_availability() -> None:
    library = SimpleNamespace(
        generativeqc_internal_batch_get_nonlocal_work_v1=lambda *_: (
            _native.STATUS_SUCCESS
        )
    )
    with pytest.raises(ValueError, match="availability"):
        _nonlocal_work(library, 12, 3)
