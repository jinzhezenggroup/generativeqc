"""NCU execution-efficiency calibration stays mechanism-specific."""

import pytest
from generativeqc_compiler.common.ncu_efficiency import (
    NcuExecutionEvidence,
    assess_ncu_execution,
)


def _evidence(**kwargs: object) -> NcuExecutionEvidence:
    return NcuExecutionEvidence(
        architecture="sm_120",
        source_revision="c5ab37ec9e7b3d38d2e06729319f9eef66510e5c",
        kernel_identity="retained-test-kernel",
        report_sha256="0" * 64,
        device="retained-test-device",
        **kwargs,
    )


def test_direct_force_capture_separates_resource_bound_from_execution_mechanism() -> (
    None
):
    assessment = assess_ncu_execution(
        _evidence(
            theoretical_occupancy_fraction=1 / 6,
            achieved_occupancy_fraction=1 / 6,
            executed_threads_per_warp_instruction=17.91,
            issue_active_fraction=0.0643,
            eligible_warps_per_scheduler=0.07,
            warp_cycles_per_issued_instruction=31.10,
            barrier_cycles_per_issued_instruction=13.67,
            wait_cycles_per_issued_instruction=8.23,
            short_scoreboard_cycles_per_issued_instruction=5.50,
            long_scoreboard_cycles_per_issued_instruction=1.49,
            local_load_requests=58_849_833_769,
            local_store_requests=27_175_631_760,
            dram_busy_fraction=0.1693,
        )
    )

    assert assessment.confidence == "high"
    assert "resource-limited-occupancy-confirmed" in assessment.mechanisms
    assert "issue-starved" in assessment.mechanisms
    assert "lane-underutilized" in assessment.mechanisms
    assert "barrier-tail" in assessment.mechanisms
    assert "barrier-divergence-local-state" in assessment.mechanisms
    assert "dram-not-saturated" in assessment.mechanisms
    assert "serial-or-underexposed-reduction" not in assessment.mechanisms
    assert assessment.derived_metrics["occupancy_realization"] == pytest.approx(1.0)
    assert assessment.derived_metrics["barrier_stall_fraction"] == pytest.approx(
        13.67 / 31.10
    )
    assert any("homogeneous class/queue/CTA" in row for row in assessment.guidance)


def test_scalar_cc_reduction_is_not_misclassified_as_register_or_spill_problem() -> (
    None
):
    assessment = assess_ncu_execution(
        _evidence(
            theoretical_occupancy_fraction=1 / 48,
            achieved_occupancy_fraction=0.02083339,
            executed_threads_per_warp_instruction=1.0,
            issue_active_fraction=0.060826,
            eligible_warps_per_scheduler=0.050393,
            warp_cycles_per_issued_instruction=16.440484,
            barrier_cycles_per_issued_instruction=0.0,
            wait_cycles_per_issued_instruction=9.072588,
            short_scoreboard_cycles_per_issued_instruction=0.757119,
            long_scoreboard_cycles_per_issued_instruction=5.551705,
            local_load_requests=0,
            local_store_requests=0,
        )
    )

    assert "serial-or-underexposed-reduction" in assessment.mechanisms
    assert "barrier-tail" not in assessment.mechanisms
    assert "local-state-traffic-present" not in assessment.mechanisms
    assert any("cooperative/tiled reduction" in row for row in assessment.guidance)


def test_exact_jk_capture_distinguishes_scoreboard_local_state_from_barrier_tail() -> (
    None
):
    assessment = assess_ncu_execution(
        _evidence(
            achieved_occupancy_fraction=0.15178221,
            executed_threads_per_warp_instruction=9.79,
            issue_active_fraction=0.13698544,
            eligible_warps_per_scheduler=0.145114,
            warp_cycles_per_issued_instruction=13.400332,
            barrier_cycles_per_issued_instruction=0.0,
            wait_cycles_per_issued_instruction=3.591015,
            short_scoreboard_cycles_per_issued_instruction=1.063381,
            long_scoreboard_cycles_per_issued_instruction=6.555522,
            local_load_requests=20_785_345_629,
            local_store_requests=20_874_479_225,
        )
    )

    assert assessment.confidence == "high"
    assert "long-scoreboard-latency" in assessment.mechanisms
    assert "scoreboard-local-state" in assessment.mechanisms
    assert "barrier-tail" not in assessment.mechanisms


def test_payload_round_trip_requires_explicit_fraction_units() -> None:
    payload = _evidence(
        theoretical_occupancy_fraction=0.25,
        achieved_occupancy_fraction=0.20,
    ).to_payload()

    restored = NcuExecutionEvidence.from_payload(payload)

    assert restored.theoretical_occupancy_fraction == 0.25
    assert restored.achieved_occupancy_fraction == 0.20
    with pytest.raises(ValueError):
        _evidence(achieved_occupancy_fraction=20.0)


def test_unknown_payload_fields_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown NCU execution evidence fields"):
        NcuExecutionEvidence.from_payload(
            {
                "schema": "generativeqc.compiler.ncu-execution-evidence.v1",
                "evidence": {"magic_speedup": 2.0},
            }
        )
