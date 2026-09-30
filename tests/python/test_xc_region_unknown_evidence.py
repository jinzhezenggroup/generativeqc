"""Unknown stage resources must not become known whole-region maxima."""

from dataclasses import replace

import pytest
from generativeqc_compiler.common.gpu_profitability import GpuProfitability
from generativeqc_compiler.dft.xc_schedule import (
    GRID_XC_COMPILED_REGION_STAGES,
    aggregate_grid_xc_compiled_evidence,
)


def _complete() -> GpuProfitability:
    return GpuProfitability(
        compiled_registers_per_thread=32,
        spill_store_bytes=0,
        spill_load_bytes=0,
        local_bytes=0,
        shared_bytes=0,
        compiled_occupancy_upper_bound=0.5,
    )


@pytest.mark.parametrize("stage_name", GRID_XC_COMPILED_REGION_STAGES)
@pytest.mark.parametrize(
    "field",
    (
        "compiled_registers_per_thread",
        "spill_store_bytes",
        "spill_load_bytes",
        "local_bytes",
        "shared_bytes",
        "compiled_occupancy_upper_bound",
    ),
)
def test_unknown_stage_field_remains_unknown(stage_name: str, field: str) -> None:
    stages = {name: _complete() for name in GRID_XC_COMPILED_REGION_STAGES}
    stages[stage_name] = replace(stages[stage_name], **{field: None})
    result = aggregate_grid_xc_compiled_evidence(stages)
    assert getattr(result, field) is None
    if field in ("spill_store_bytes", "spill_load_bytes"):
        assert result.spill_bytes is None


@pytest.mark.parametrize("invalid", (None, {}, 0))
def test_region_rejects_invalid_stage_records(invalid: object) -> None:
    stages = {name: _complete() for name in GRID_XC_COMPILED_REGION_STAGES}
    stages[GRID_XC_COMPILED_REGION_STAGES[0]] = invalid
    with pytest.raises(TypeError, match="GpuProfitability"):
        aggregate_grid_xc_compiled_evidence(stages)


def test_complete_stage_records_still_publish_known_pressure() -> None:
    stages = {name: _complete() for name in GRID_XC_COMPILED_REGION_STAGES}
    stages[GRID_XC_COMPILED_REGION_STAGES[-1]] = replace(
        _complete(), compiled_registers_per_thread=96, spill_store_bytes=8
    )
    result = aggregate_grid_xc_compiled_evidence(stages)
    assert result.compiled_registers_per_thread == 96
    assert result.spill_store_bytes == 8
    assert result.spill_load_bytes == 0
    assert result.spill_bytes == 8
    assert result.compiled_occupancy_upper_bound == 0.5
