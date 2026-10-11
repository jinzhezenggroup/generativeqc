"""Hardware-free regression tests for the #308 response timeline gates."""

import gzip
import json
from pathlib import Path

import pytest

from benchmarks.issue308_response_timeline import (
    retained_force_reference_path,
    validate_metric_response_work,
    validate_three_center_response_work,
)


@pytest.mark.parametrize(
    ("packed_pairs", "pair_storage", "pair_stride"),
    [(0, "dense", 4 * 4), (1, "packed", 4 * 5 // 2)],
)
def test_timeline_accepts_exact_dense_and_packed_response_work(
    packed_pairs: int, pair_storage: str, pair_stride: int
) -> None:
    """Both complete pair representations are valid response routes."""

    expected = 5 * pair_stride
    result = validate_three_center_response_work(
        {
            "response_packed_pairs": packed_pairs,
            "three_center_derivative_weights": expected,
            "three_center_derivative_weight_bytes": expected * 8,
        },
        nbf=4,
        naux=5,
    )

    assert result == {
        "pair_storage": pair_storage,
        "pair_stride": pair_stride,
        "expected_three_center_derivative_weights": expected,
        "observed_three_center_derivative_weights": expected,
    }


def test_timeline_rejects_dense_count_for_packed_route() -> None:
    """A packed route must not pass by reporting the larger dense count."""

    with pytest.raises(RuntimeError, match="packed pairs"):
        validate_three_center_response_work(
            {
                "response_packed_pairs": 1,
                "three_center_derivative_weights": 5 * 4 * 4,
            },
            nbf=4,
            naux=5,
        )


def test_timeline_rejects_weight_byte_mismatch() -> None:
    with pytest.raises(RuntimeError, match="weight bytes"):
        validate_three_center_response_work(
            {
                "response_packed_pairs": 1,
                "three_center_derivative_weights": 5 * (4 * 5 // 2),
                "three_center_derivative_weight_bytes": 1,
            },
            nbf=4,
            naux=5,
        )


def test_timeline_recovers_host_fallback_work_from_exact_fp64_bytes() -> None:
    """Legacy host-weight traces may publish bytes without an element count."""

    expected = 5 * 4 * 4
    result = validate_three_center_response_work(
        {
            "response_packed_pairs": 0,
            "three_center_derivative_weight_bytes": expected * 8,
        },
        nbf=4,
        naux=5,
    )

    assert result["observed_three_center_derivative_weights"] == expected


def test_metric_work_accepts_exact_host_fallback_bytes() -> None:
    assert (
        validate_metric_response_work({"metric_derivative_weight_bytes": 5 * 5 * 8}, 5)
        == 25
    )


@pytest.mark.parametrize("compressed", [False, True])
def test_timeline_retained_force_reference_storage(
    tmp_path: Path, compressed: bool
) -> None:
    from tools.generativeqc_validation.record import load_record

    logical = tmp_path / "forces.json"
    stored = logical.with_suffix(".json.gz") if compressed else logical
    payload = {"forces": [[1.0, -2.0, 3.0]], "seconds": [0.2, 0.3]}
    data = json.dumps(payload).encode()
    stored.write_bytes(gzip.compress(data, mtime=0) if compressed else data)
    actual = retained_force_reference_path(logical)
    assert actual == stored
    assert load_record(actual) == payload


def test_timeline_missing_reference_does_not_invent_a_companion(tmp_path: Path) -> None:
    logical = tmp_path / "missing.json"
    assert retained_force_reference_path(logical) == logical
