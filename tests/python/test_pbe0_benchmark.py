"""Host-only guards for PBE0's complete forces and independent evidence."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from generativeqc import GridSpec
from generativeqc._model_resolution import snapshot_basis

from benchmarks.readme_omol25 import protocol
from benchmarks.readme_pbe0 import PBE0, SCHEMA
from benchmarks.readme_wb97mv import configure_reference_full_fock
from tests.python.test_omol25_benchmark import run
from tools.render_omol25_benchmarks import collect, figure, validate


def test_pbe0_protocol_reuses_full_svp_and_has_no_vv10() -> None:
    """Retain polarization and moving-grid response; do not attach OMol25 masks."""
    full = snapshot_basis("def2-svp", "spherical")
    assert PBE0.native_method == "pbe0-rks"
    basis = replace(full, elements=tuple(full.by_element[number] for number in (1, 8)))
    for atoms in (3, 6, 12, 24, 48, 96):
        scientific = protocol(atoms, basis, GridSpec(), 5, benchmark=PBE0)
        assert scientific["method"] == "PBE0/RKS"
        assert scientific["basis"] == "def2-SVP"
        assert scientific["aos"] == atoms * 8
        assert scientific["basis_identity"] == basis.identity
        assert scientific["properties"] == ["energy", "forces"]
        assert scientific["force_return"] == "host_array_in_timed_endpoint"
        assert not scientific["density_fitting"]
        assert not any("nonlocal" in key for key in scientific)
        assert scientific["repeats"] == 5
        assert scientific["reference_fock_policy"] == "full-density-rebuild"
        assert max(shell.angular_momentum for shell in basis.by_element[8].shells) == 2


def test_reference_full_fock_discards_incremental_inputs_and_preserves_density() -> (
    None
):
    """The RKS backend ignores direct_scf alone; no stale potential may leak in."""
    calls = []
    potential = object()

    def original(
        mol: object = None,
        dm: object = None,
        dm_last: object = None,
        vhf_last: object = None,
        hermi: int = 1,
    ) -> object:
        calls.append((mol, dm, dm_last, vhf_last, hermi))
        return potential

    engine = SimpleNamespace(get_veff=original, direct_scf=True)
    configure_reference_full_fock(engine)
    molecule, density, stale_density, stale_potential = (object() for _ in range(4))
    assert (
        engine.get_veff(molecule, density, stale_density, stale_potential, 2)
        is potential
    )
    assert (
        engine.get_veff(dm=density, dm_last=stale_density, vhf_last=stale_potential)
        is potential
    )
    assert calls == [(molecule, density, None, None, 2), (None, density, None, None, 1)]
    assert engine.direct_scf is False


def test_pbe0_evidence_uses_own_schema_and_exact_oracle(tmp_path: Path) -> None:
    """Method-specific schema prevents accidentally plotting WB97M-V evidence."""
    reference = run()
    reference["schema"] = SCHEMA
    reference["protocol"].update(method="PBE0/RKS", basis="def2-SVP", aos=24)
    validate(reference, reference, schema=SCHEMA)
    with pytest.raises(ValueError, match="schema mismatch"):
        validate(reference, reference)
    directory = tmp_path / "3"
    directory.mkdir()
    oracle_path = directory / "reference.json"
    oracle_path.write_text(json.dumps(reference))
    native = copy.deepcopy(reference)
    native["reference_sha256"] = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    (directory / "native.json").write_text(json.dumps(native))
    for engine in ("reference", "native"):
        (directory / f"{engine}.outcome").write_text('{"exit_code":0}')
    point = collect(tmp_path, 3, schema=SCHEMA)
    assert point["engines"]["native"]["medians"]["warm"] == 1.0
    (directory / "native.outcome").write_text('{"exit_code":124}')
    point = collect(tmp_path, 3, schema=SCHEMA)
    assert point["engines"]["native"]["status"] == "timeout"
    assert "medians" not in point["engines"]["native"]


@pytest.mark.parametrize("atoms", (3, 6, 12))
def test_retained_pbe0_journals_recheck_every_endpoint_and_hash(atoms: int) -> None:
    """Public medians stay bound to raw host forces, including moved repeats."""
    directory = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/pbe0-def2-svp-20261001"
    )
    point = json.loads((directory / f"water{atoms}.json").read_text())
    raw_bytes = {
        engine: gzip.decompress(
            (directory / f"water{atoms}-{engine}.json.gz").read_bytes()
        )
        for engine in ("native", "reference")
    }
    reference = json.loads(raw_bytes["reference"])
    for engine, raw in raw_bytes.items():
        entry = point["engines"][engine]
        assert entry["status"] == "measured"
        assert entry["raw_sha256"] == hashlib.sha256(raw).hexdigest()
        validate(json.loads(raw), reference, schema=SCHEMA)
    assert (
        json.loads(raw_bytes["native"])["reference_sha256"]
        == hashlib.sha256(raw_bytes["reference"]).hexdigest()
    )
    assert point["protocol"]["method"] == "PBE0/RKS"
    assert point["protocol"]["aos"] == atoms * 8


@pytest.mark.parametrize("atoms", (24, 48, 96))
def test_retained_pbe0_larger_failures_are_not_timings(atoms: int) -> None:
    """Missing complete oracles cannot qualify native timings or curve bounds."""
    directory = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/pbe0-def2-svp-20261001"
    )
    point = json.loads((directory / f"water{atoms}.json").read_text())
    assert point["engines"]["native"]["status"] == "reference_unavailable"
    assert point["engines"]["reference"]["status"] == "failed"
    for entry in point["engines"].values():
        assert "medians" not in entry


@pytest.mark.parametrize("atoms", (3, 6, 12, 24, 48, 96))
def test_grid_reuse_campaign_rechecks_every_endpoint(
    atoms: int, tmp_path: Path
) -> None:
    """Rebuild each retained point from exact raw bytes, not summary gate flags."""
    directory = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/pbe0-grid-reuse-20261003"
    )
    provenance = json.loads((directory / "provenance.json").read_text())
    summaries = json.loads(
        gzip.decompress((directory / "summary.json.gz").read_bytes())
    )
    summary = next(point for point in summaries if point["atoms"] == atoms)
    restored = tmp_path / str(atoms)
    restored.mkdir()
    for engine, entry in summary["engines"].items():
        name = f"water{atoms}-{engine}.json.gz"
        raw = gzip.decompress((directory / name).read_bytes())
        identity = provenance["raw_files"][name]
        assert hashlib.sha256(raw).hexdigest() == identity["raw_sha256"]
        assert len(raw) == identity["raw_bytes"]
        (restored / f"{engine}.json").write_bytes(raw)
        (restored / f"{engine}.outcome").write_text(json.dumps(entry["outcome"]))
    checked = collect(tmp_path, atoms, schema=SCHEMA)
    assert checked["protocol"] == summary["protocol"]
    for engine, entry in checked["engines"].items():
        assert entry["status"] == "measured"
        assert len(entry["records"]) == 12
        assert entry["medians"] == summary["engines"][engine]["medians"]
        assert entry["raw_sha256"] == summary["engines"][engine]["raw_sha256"]
    native = checked["engines"]["native"]
    assert (
        native["native_build"]["library_sha256"] == provenance["native_library_sha256"]
    )
    for record in native["records"]:
        assert record["converged"] and record["gate"]
        if record["phase"] in ("warm", "moved-warm"):
            assert record["iterations"] == 1
            assert record["warm_start_used"]
            assert not record["warm_start_fallback"]


def test_pbe0_plot_has_its_own_method_label(tmp_path: Path) -> None:
    """Rendering alone needs matplotlib; timeout/evidence validation always runs."""
    pytest.importorskip("matplotlib")
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/pbe0-def2-svp-20261001/water3.json"
    )
    point = json.loads(path.read_text())
    figure([point], tmp_path, title="PBE0 / def2-SVP", filename="pbe0.svg")
    svg = (tmp_path / "pbe0.svg").read_text()
    assert "PBE0 / def2-SVP" in svg
    assert "OMol25" not in svg and "through-f" not in svg
