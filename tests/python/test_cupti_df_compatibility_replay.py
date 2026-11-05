"""CPU replay of retained CUDA DF observations, never fresh GPU qualification."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from tools.cupti_source_capture import summarize_sources
from tools.generativeqc_posthf.fixtures import load_fixture

if TYPE_CHECKING:
    from collections.abc import Iterator

RETAINED = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/residency-n4-20261106/production-captures.tar.xz"
)


@pytest.fixture(scope="module")
def capture_archive() -> Iterator[tarfile.TarFile]:
    """Read named files in memory without extracting archive paths."""
    with tarfile.open(RETAINED, "r:xz") as archive:
        yield archive


def archived_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    """Require an actual retained file, not an optional or regenerated summary."""
    stream = archive.extractfile(name)
    assert stream is not None
    with stream:
        return stream.read()


def archived_json(archive: tarfile.TarFile, name: str) -> Any:
    """Decode exact journal bytes from the historical source-matched capture."""
    return json.loads(archived_bytes(archive, name))


@pytest.mark.parametrize("name", ["h2", "water", "lih", "f_heh"])
def test_real_df_compatibility_journals_and_independent_fixture(
    capture_archive: tarfile.TarFile, name: str
) -> None:
    """Recompute real D2H joins and numerical parity, without inventing H2D links."""
    prefix = f"df-compatibility/{name}"
    source = archived_json(capture_archive, f"{prefix}/source-boundaries.json")
    activity = archived_json(capture_archive, f"{prefix}/activity.json")
    regions = archived_json(capture_archive, f"{prefix}/regions.json")
    result = summarize_sources(
        source, activity, {int(key): value for key, value in regions.items()}
    )
    observed = archived_json(capture_archive, f"{prefix}/numerical-and-work.json")
    expected, arrays = load_fixture(name)
    assert observed["fixture_array_hash"] == expected["array_hash"]
    with np.load(
        io.BytesIO(archived_bytes(capture_archive, f"{prefix}/numerical.npz")),
        allow_pickle=False,
    ) as numerical:
        assert (
            abs(
                float(numerical["correlation_energy"])
                - expected["records"]["df"]["correlation_energy"]
            )
            < 1e-9
        )
        np.testing.assert_allclose(
            numerical["amplitudes"], arrays["df_t2"], atol=1e-11, rtol=1e-10
        )
    assert result["source_annotations_intact"], result["issues"]
    compatibility = result["by_role"]["compatibility"]
    assert compatibility["source_transfer_calls"] > 0
    assert compatibility["source_event_fences"] > 0
    assert compatibility["d2h_bytes"] == observed["metrics"]["d2h_bytes"] > 0
    assert result["status"] == "INCOMPLETE"
    assert observed["metrics"]["subsequent_h2d_bytes"] is None
    assert observed["zero_round_trip_assertion"] is None
    assert all(row["dependency_identity"] is None for row in result["operations"])
