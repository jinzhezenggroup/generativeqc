"""Re-audit actual scheduled scientific journals without executing CUDA.

Raw source/API/activity joins, independent fixture energies and separately pinned
boundary budgets are the inputs. Saved diagnostic verdicts are never trusted.
"""

from __future__ import annotations

import copy
import json
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tools.cupti_source_capture import summarize_sources
from tools.cupti_source_ratchet import reaudit_capture
from tools.generativeqc_posthf.fixtures import load_fixture

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[2]
RETAINED = ROOT / "tests/reference_data/cupti/n4-1511"
POLICY = (
    ROOT / "manifests/residency_work_ratchets/hf_prepared_direct_fp64_tight.v3.json"
)


@pytest.fixture
def metric_archive(name: str) -> Iterator[tarfile.TarFile]:
    with tarfile.open(RETAINED / (name + ".tar.xz"), "r:xz") as captured:
        yield captured


@pytest.fixture
def prepared_archive(method: str) -> Iterator[tarfile.TarFile]:
    with tarfile.open(RETAINED / (method + ".tar.xz"), "r:xz") as captured:
        yield captured


def read(captured: tarfile.TarFile, name: str) -> Any:
    """Decode an actual regular journal without extracting archive paths."""
    member = captured.getmember(name)
    assert member.isfile()
    stream = captured.extractfile(member)
    assert stream is not None
    with stream:
        return json.load(stream)


@pytest.mark.parametrize("name", ["h2", "lih", "water"])
def test_real_scientific_metric_transform_lineage(
    metric_archive: tarfile.TarFile,
    name: str,
) -> None:
    archive = metric_archive
    prefix = f"df-metric/{name}/"
    source = read(archive, prefix + "source-boundaries.json")
    activity = read(archive, prefix + "activity.json")
    regions = {
        int(key): value for key, value in read(archive, prefix + "regions.json").items()
    }
    result = summarize_sources(source, activity, regions)
    assert result["source_annotations_intact"], result["issues"]
    assert result["qualified_host_round_trips"]
    assert {row["owner"] for row in result["qualified_host_round_trips"]} == {
        "df_metric_setup"
    }
    numerical = read(archive, prefix + "numerical.json")
    meta, _ = load_fixture(name)
    fixture = meta["records"]["df"]
    assert numerical["fixture_array_hash"] == meta["array_hash"]
    for energy in (
        fixture["hf_energy"] + fixture["correlation_energy"],
        numerical["ordinary_energy"],
        numerical["cpu_energy"],
    ):
        assert abs(numerical["energy"] - energy) < 1e-9
    broken = copy.deepcopy(source)
    for row in broken["records"]:
        if row["dependency"]:
            row["dependency"] = (1 << 63) - 1
    rejected = summarize_sources(broken, activity, regions)
    assert not rejected["source_annotations_intact"]
    assert not rejected["qualified_host_round_trips"]
    assert rejected["zero_round_trip_assertion"] is None


@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_real_prepared_resident_boundary_contract(
    prepared_archive: tarfile.TarFile,
    method: str,
) -> None:
    archive = prepared_archive
    expected = read(archive, f"{method}-expected.json")
    source = read(archive, f"{method}-capture/source-boundaries.json")
    activity = read(archive, f"{method}-capture/activity.json")
    result = reaudit_capture(expected, POLICY, source, activity)
    assert result["status"] == "WITHIN_OBSERVED_RATCHET", result["issues"]
    assert (
        result["declared_resident_scopes"][
            "zero_unexpected_post_prepare_host_round_trips"
        ]
        is True
    )
    assert result["declared_resident_scopes"]["owners"] == ["hf_bucket"]
    assert result["residency_status"] == "INCOMPLETE"
    assert result["zero_round_trip_assertion"] is None
    numerical = read(archive, f"{method}-capture/numerical.json")
    for kind in ("numerical", "independent_numerical"):
        for sample in numerical[kind]:
            assert sample["energy_error"] < expected["workload"]["energy_gate"]
            if sample["force_error"] is not None:
                assert sample["force_error"] < expected["workload"]["force_gate"]
    # An equal-byte scientific-payload substitution must not inherit the
    # publication allowance merely because aggregate traffic is unchanged.
    altered = copy.deepcopy(source)
    altered["catalog"]["payload"]["4"] = "unexpected-host-density-transform"
    rejected = reaudit_capture(expected, POLICY, altered, activity)
    assert rejected["counts"] == result["counts"]
    assert rejected["status"] == "VIOLATION"
    assert (
        rejected["declared_resident_scopes"][
            "zero_unexpected_post_prepare_host_round_trips"
        ]
        is False
    )
