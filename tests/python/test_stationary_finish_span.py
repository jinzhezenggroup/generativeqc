"""Contiguous stationary source publication avoids unused D2H traffic."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from generativeqc._stationary_cuda import _CudaSources

ROOT = Path(__file__).resolve().parents[2]


class FakeOwner:
    source_names = (
        "one_electron",
        "coulomb",
        "xc_ao",
        "xc_grid",
        "xc_weight",
        "overlap_pulay",
        "nuclear",
    )
    natom = 2
    handle = SimpleNamespace(value=17)

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.flushes = 0

    def flush(self) -> None:
        self.flushes += 1

    def _call(self, name: str, *args: object) -> None:
        self.calls.append((name, *args))


def test_finish_span_binds_one_contiguous_runtime_interval() -> None:
    owner = FakeOwner()
    result = _CudaSources.finish_span(owner, ("xc_ao", "xc_grid", "xc_weight"))
    assert tuple(result) == ("xc_ao", "xc_grid", "xc_weight")
    assert owner.flushes == 1
    assert len(owner.calls) == 1
    name, _handle, start, count, _pointer, elements = owner.calls[0]
    assert name == "stationary_finish_span"
    assert (start, count, elements) == (2, 3, 18)


@pytest.mark.parametrize(
    "names,match",
    [
        ((), "must not be empty"),
        (("xc_ao", "xc_ao"), "duplicate"),
        (("unknown",), "unknown source"),
        (("xc_ao", "xc_weight"), "contiguous"),
        (("xc_grid", "xc_ao"), "runtime order"),
    ],
)
def test_finish_span_rejects_invalid_source_sets(
    names: tuple[str, ...], match: str
) -> None:
    owner = FakeOwner()
    with pytest.raises(ValueError, match=match):
        _CudaSources.finish_span(owner, names)
    assert owner.calls == []


def test_native_finish_span_copies_only_the_requested_source_interval() -> None:
    source = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    section = source.split("int stationary_finish_span(", 1)[1].split(
        "int stationary_finish_reduced(", 1
    )[0]
    assert "source_count > stationary_source_count - source_begin" in section
    assert "count != 3 * source_count * p->atoms" in section
    assert "const auto offset = 3 * source_begin * p->atoms;" in section
    assert "p->sources + offset" in section
    assert "p->downloads += count * 8;" in section
    assert "source_reduce<<<" not in section
