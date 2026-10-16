"""Recompute the accepted molecular gates from the retained raw observations."""

from __future__ import annotations

import hashlib
import itertools
import lzma
import statistics
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.evidence import block_error, validate_evidence

from benchmarks.compare_wb97mv_hessian_gpu4pyscf import _summarize
from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_json, load_publication_record

DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/wb97mv-hessian-matched-cphf-20261016"
)


def test_matched_comparison_preserves_authenticated_observations() -> None:
    """Storage validity alone cannot substitute for numerical qualification."""
    manifest = load_json(DIRECTORY / "publication.json")
    files = {
        entry["path"]: (DIRECTORY / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    record = load_publication_record(DIRECTORY)
    validate_evidence(record)
    samples = load_publication_record(DIRECTORY, role="samples")
    assert samples["status"] == "completed"
    assert samples["provenance"]["source_commit"] == record["revision"]
    assert manifest["decision"]["scope"] == "numerical"
    assert manifest["decision"]["status"] == "accepted"
    assert record["performance"]["status"] != "pass"


@pytest.mark.parametrize("case", ("h2", "water"))
def test_raw_matrices_pass_common_and_independent_energy_gates(case: str) -> None:
    """Check every entry without symmetrization, including both energy stencils."""
    record = load_publication_record(DIRECTORY)
    samples = load_publication_record(DIRECTORY, role="samples")["cases"][case]
    oracle = record["qualification"]["independent_full_matrices"][case]
    coordinates = 3 * len(samples["atoms_bohr"])
    expected = np.asarray(oracle["expected"])
    stencils = np.asarray(oracle["energy_stencil_matrices"])
    assert expected.shape == (coordinates, coordinates)
    assert stencils.shape == (2, coordinates, coordinates)
    np.testing.assert_array_equal(expected, (4 * stencils[1] - stencils[0]) / 3)
    observed = block_error(oracle["actual"], expected, atol=8e-7, rtol=1e-7)
    assert observed["passed"]
    assert observed["max_absolute_error"] == pytest.approx(
        oracle["error"]["max_absolute_error"], rel=1e-12, abs=1e-18
    )
    groups = {
        engine: [row for row in samples["samples"] if row["engine"] == engine]
        for engine in ("generativeqc", "gpu4pyscf")
    }
    for row in samples["samples"]:
        matrix = np.asarray(row["result"]["matrix"])
        assert matrix.shape == expected.shape and np.isfinite(matrix).all()
        np.testing.assert_allclose(matrix, matrix.T, atol=2e-7, rtol=0)
    for native in groups["generativeqc"]:
        assert block_error(native["result"]["matrix"], expected, atol=8e-7, rtol=1e-7)[
            "passed"
        ]
    for native, reference in itertools.product(
        groups["generativeqc"], groups["gpu4pyscf"]
    ):
        assert block_error(
            native["result"]["matrix"], reference["result"]["matrix"], atol=3e-6, rtol=0
        )["passed"]


@pytest.mark.parametrize("case", ("h2", "water"))
def test_endpoint_ratios_exclude_first_observed_use(case: str) -> None:
    """Recompute timings from complete samples, not isolated Hessian kernels."""
    record = load_publication_record(DIRECTORY, role="samples")["cases"][case]
    observed = _summarize(record["samples"], 3e-6)
    assert observed["accepted"]
    assert observed == record["summary"]
    for engine in ("generativeqc", "gpu4pyscf"):
        rows = [row for row in record["samples"] if row["engine"] == engine]
        assert {row["repeat"] for row in rows} == {0, 1, 2}
        primed = [row["wall_seconds"] for row in rows if row["repeat"] > 0]
        assert all(seconds > 0 for seconds in primed)
        assert (
            statistics.median(primed)
            == observed["artifact_primed_fresh_process_median_seconds"][engine]
        )


@pytest.mark.parametrize("case", ("h2", "water"))
def test_reference_quadrature_and_native_runtime_contract(case: str) -> None:
    """Keep numerical ownership and response-grid work visible in each sample."""
    samples = load_publication_record(DIRECTORY, role="samples")
    assert samples["workload"]["response_grid_policy"] == "same-as-primary"
    for row in samples["cases"][case]["samples"]:
        result = row["result"]
        if row["engine"] == "gpu4pyscf":
            assert result["response_grid_policy"] == "same-as-primary"
            assert (
                result["response_grid_points"] == result["semilocal_padded_grid_points"]
            )
            assert result["vv10"] and result["grid_response"]
        else:
            diagnostics = result["diagnostics"]
            assert diagnostics["runtime_python"] is False
            assert diagnostics["runtime_compilation"] is False
            assert diagnostics["numerical_host_fallbacks"] == 0
            assert diagnostics["intermediate_panel_downloads"] == 0
            assert diagnostics["displaced_scf_evaluations"] == 0


def test_historical_control_driver_retains_exact_measured_bytes() -> None:
    """The compressed measured driver is provenance, not an implementation pin."""
    control = load_publication_record(DIRECTORY)["qualification"][
        "reference_grid_control"
    ]
    driver = lzma.decompress((DIRECTORY / "reference-grid-control.py.xz").read_bytes())
    assert hashlib.sha256(driver).hexdigest() == control["driver_sha256"]
