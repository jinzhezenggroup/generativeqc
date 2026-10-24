"""Recompute numerical and timing claims from authenticated endpoint samples."""

from __future__ import annotations

import hashlib
import itertools
import lzma
import statistics
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.evidence import block_error, validate_evidence

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_json, load_publication_record

DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/wb97mv-hessian-direction-reuse-20261016"
)
ENGINES = ("baseline", "candidate", "gpu4pyscf")


def test_optimized_hessian_publication_is_authenticated() -> None:
    """Retain source identity and raw scientific evidence, not only a speed ratio."""
    manifest = load_json(DIRECTORY / "publication.json")
    files = {
        entry["path"]: (DIRECTORY / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    evidence = load_publication_record(DIRECTORY)
    validate_evidence(evidence)
    samples = load_publication_record(DIRECTORY, role="samples")
    assert samples["status"] == "completed"
    assert (
        samples["driver_sha256"]
        == hashlib.sha256(
            lzma.decompress((DIRECTORY / "compare-native-ab.py.xz").read_bytes())
        ).hexdigest()
    )
    assert samples["provenance"]["candidate"]["source_revision"] == evidence["revision"]
    assert manifest["decision"]["scope"] == "numerical"
    assert manifest["decision"]["status"] == "accepted"
    parity = evidence["qualification"]["resident_streamed_full_matrices"]
    assert len(parity) == 4
    for owners in parity.values():
        assert owners["resident"]["resident_integral_bytes"] > 0
        assert owners["streamed"]["resident_integral_bytes"] == 0
        for field in ("energy", "hvp", "hessian", "response_residual"):
            assert block_error(
                np.atleast_1d(owners["resident"][field]),
                np.atleast_1d(owners["streamed"][field]),
                atol=3e-12,
                rtol=3e-12,
            )["passed"]
    for engine in ENGINES[:2]:
        build = evidence["qualification"]["clean_build"]["builds"][engine]
        provenance = samples["provenance"][engine]
        assert build["binary_sha256"]["generativeqc"] == provenance["cli_sha256"]
        assert (
            build["binary_sha256"]["libgenerativeqc.so.0.1.0"]
            == provenance["library_sha256"]
        )
        assert build["inventory_sha256"] == provenance["inventory_sha256"]
        for case in samples["cases"].values():
            assert all(
                row["result"]["native_source_identity"] == build["source_identity"]
                for row in case["samples"]
                if row["engine"] == engine
            )


@pytest.mark.parametrize("case", ("h2", "water"))
def test_optimized_raw_matrices_pass_independent_and_common_gates(case: str) -> None:
    """Check raw entries against retained independent values-only energy stencils."""
    samples = load_publication_record(DIRECTORY, role="samples")["cases"][case]
    evidence = load_publication_record(DIRECTORY)
    oracle = evidence["qualification"]["independent_full_matrices"][case]
    expected = np.asarray(oracle["expected"])
    stencils = np.asarray(oracle["energy_stencil_matrices"])
    coordinates = 3 * len(samples["atoms_bohr"])
    assert expected.shape == (coordinates, coordinates)
    assert stencils.shape == (2, coordinates, coordinates)
    np.testing.assert_array_equal(expected, (4 * stencils[1] - stencils[0]) / 3)
    assert block_error(oracle["actual"], expected, atol=8e-7, rtol=1e-7)["passed"]
    groups = {
        engine: [row for row in samples["samples"] if row["engine"] == engine]
        for engine in ENGINES
    }
    errors = {}
    for first, second in itertools.combinations(ENGINES, 2):
        errors[first + "/" + second] = max(
            block_error(
                left["result"]["matrix"],
                right["result"]["matrix"],
                atol=3e-12 if second == "candidate" else 3e-6,
                rtol=0,
            )["max_absolute_error"]
            for left, right in itertools.product(groups[first], groups[second])
        )
    assert errors == samples["summary"]["max_matrix_errors"]
    assert errors["baseline/candidate"] <= 3e-12
    assert max(errors.values()) <= 3e-6
    for baseline, candidate in itertools.product(
        groups["baseline"], groups["candidate"]
    ):
        previous = baseline["result"]["diagnostics"]
        current = candidate["result"]["diagnostics"]
        for field in (
            "completed_directions",
            "scf_iterations",
            "induced_fock_actions",
            "one_electron_primitive_components",
            "grid_mixed_contractions",
            "vv10_reference_pairs",
        ):
            assert previous[field] == current[field]
        assert (
            current["exact_primitive_components"]
            < previous["exact_primitive_components"]
        )
        assert current["exact_integral_source_builds"] == coordinates
        assert current["exact_resident_integral_hits"] == coordinates
    for row in groups["candidate"]:
        assert np.asarray(row["result"]["matrix"]).shape == expected.shape
        assert block_error(row["result"]["matrix"], expected, atol=8e-7, rtol=1e-7)[
            "passed"
        ]
        diagnostics = row["result"]["diagnostics"]
        assert diagnostics["runtime_python"] is False
        assert diagnostics["runtime_compilation"] is False
        assert diagnostics["posthoc_symmetrization"] is False
        assert diagnostics["intermediate_panel_downloads"] == 0
        assert diagnostics["numerical_host_fallbacks"] == 0
        assert diagnostics["displaced_scf_evaluations"] == 0
        assert diagnostics["derivative_authority"] == "compiler-ad"
        assert diagnostics["execution_residency"] == "device-resident"
        assert all(
            diagnostics[field] == "cuda"
            for field in (
                "first_integral_backend",
                "second_integral_backend",
                "grid_ad_backend",
                "response_backend",
            )
        )
    for row in groups["gpu4pyscf"]:
        assert row["result"]["response_grid_policy"] == "same-as-primary"
        assert (
            row["result"]["response_grid_points"]
            == row["result"]["semilocal_padded_grid_points"]
        )


@pytest.mark.parametrize("case", ("h2", "water"))
def test_optimized_ratios_use_all_primed_complete_endpoints(case: str) -> None:
    """A noisy future run need not be faster, but every reported ratio must agree."""
    report = load_publication_record(DIRECTORY, role="samples")
    samples = report["cases"][case]
    repeats = report["workload"]["primed_repeats"]
    medians = {}
    for engine in ENGINES:
        rows = [row for row in samples["samples"] if row["engine"] == engine]
        assert {row["repeat"] for row in rows} == set(range(repeats + 1))
        assert len(rows) == repeats + 1
        assert all(row["wall_seconds"] > 0 for row in rows)
        medians[engine] = statistics.median(
            row["wall_seconds"] for row in rows if row["repeat"] > 0
        )
    assert medians == samples["summary"]["primed_median_seconds"]
    assert samples["summary"]["candidate_over_baseline"] == (
        medians["candidate"] / medians["baseline"]
    )
    assert samples["summary"]["candidate_over_gpu4pyscf"] == (
        medians["candidate"] / medians["gpu4pyscf"]
    )
