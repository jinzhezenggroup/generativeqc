"""CPU-only guards for the OMol25 benchmark's scientific and plotting contract."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import numpy as np
import pytest
from generativeqc import GridSpec, import_bse, load_basis

from benchmarks.readme_omol25 import (
    SCHEMA,
    check_record,
    native_endpoint_record,
    protocol,
    reference_energy_forces,
    require_force_endpoint,
    retain_endpoint_record,
    source_hashes,
)
from benchmarks.readme_wb97mv import reference_vv10_domain
from tools.render_omol25_benchmarks import collect, figure, validate


def sample(phase: str, geometry: int, repeat: int = 0) -> dict:
    """Make a complete shape-correct endpoint with explicit semantic work."""
    return {
        "phase": phase,
        "geometry": geometry,
        "repeat": repeat,
        "energy": -76.0,
        "forces": [[0.0, 0.0, 0.0]] * 3,
        "converged": True,
        "status": 0,
        "gate": True,
        "iterations": 1,
        "complete_seconds": 1.0,
    }


def run() -> dict:
    """Keep cold, moved and every requested replay in a deterministic fixture."""
    return {
        "schema": SCHEMA,
        "protocol": {
            "repeats": 5,
            "aos": 58,
            "force_return": "host_array_in_timed_endpoint",
        },
        "status": "measured",
        "stage": "complete",
        "records": [
            sample("cold", 0),
            *[sample("warm", 0, i) for i in range(5)],
            sample("moved", 1),
            *[sample("moved-warm", 1, i) for i in range(5)],
        ],
    }


def test_protocol_preserves_diffuse_basis_and_hf_geometries() -> None:
    source = Path(__file__).parents[1] / "data/external_basis/sto-3g-ho.json"
    basis = import_bse(
        source,
        source="test",
        source_version="1",
        license="BSD-3-Clause",
        representation="spherical",
    )
    original = protocol(3, basis, GridSpec(), 5)
    assert original["aos"] == 7
    assert original["basis_identity"] == basis.identity
    assert original["energy_tolerance"] == 1e-12
    assert (
        original["density_tolerance"]
        == original["reference_gradient_tolerance"]
        == 1e-10
    )
    assert original["properties"] == ["energy", "forces"]
    assert original["force_return"] == "host_array_in_timed_endpoint"
    assert original["nonlocal_density_threshold"] == 1e-8
    assert original["reference_nonlocal_weight_threshold"] == 1e-14
    first, second = original["geometries_bohr"]
    assert first[0] == second[0] and first[2] == second[2]
    assert second[1][1][2] - first[1][1][2] == pytest.approx(0.001)
    oxygen = basis.by_element[8]
    extra = replace(oxygen.shells[-1], exponents=("0.01",), coefficients=(("1",),))
    changed = replace(
        basis,
        elements=tuple(
            replace(element, shells=(*element.shells, extra))
            if element.atomic_number == 8
            else element
            for element in basis.elements
        ),
    )
    assert protocol(3, changed, GridSpec(), 5)["aos"] == 10
    assert changed.identity != basis.identity


def test_reference_force_export_precedes_endpoint_clock_stop() -> None:
    """HF includes GPU4PySCF's force D2H transfer, not just gradient kernels."""
    events = []
    gradient_values = np.arange(9).reshape(3, 3)
    seed_copy = object()

    def seed_copy_event() -> object:
        events.append("seed-copy")
        return seed_copy

    def scf(*, dm0: object) -> float:
        assert dm0 is seed_copy
        events.append("scf")
        return -76.0

    gradient = SimpleNamespace(grid_response=False)

    def gradient_kernel() -> np.ndarray:
        assert gradient.grid_response is True
        events.append("gradient")
        return gradient_values

    gradient.kernel = gradient_kernel
    engine = SimpleNamespace(
        kernel=scf, converged=True, nuc_grad_method=lambda: gradient
    )

    def export(values: np.ndarray) -> np.ndarray:
        events.append("export")
        np.testing.assert_array_equal(values, -gradient_values)
        return values.copy()

    def clock() -> float:
        events.append("clock")
        return 10.0 if events.count("clock") == 1 else 14.5

    energy, forces, seconds = reference_energy_forces(
        engine,
        SimpleNamespace(copy=seed_copy_event),
        export=export,
        synchronize=lambda: events.append("synchronize"),
        clock=clock,
    )
    assert events == [
        "synchronize",
        "clock",
        "seed-copy",
        "scf",
        "gradient",
        "export",
        "synchronize",
        "clock",
    ]
    assert energy == -76.0 and seconds == 4.5
    np.testing.assert_array_equal(forces, -gradient_values)


@pytest.mark.parametrize("failure", ["convergence", "gradient", "export"])
def test_reference_endpoint_propagates_failure(failure: str) -> None:
    events = []

    def kernel(*, dm0: Any) -> float:
        assert dm0 is None
        return -76.0

    def gradient_kernel() -> np.ndarray:
        events.append("gradient")
        if failure == "gradient":
            raise RuntimeError("gradient failed")
        return np.zeros((3, 3))

    def export(values: np.ndarray) -> NoReturn:
        events.append("export")
        raise RuntimeError("export failed")

    engine = SimpleNamespace(
        kernel=kernel,
        converged=failure != "convergence",
        nuc_grad_method=lambda: SimpleNamespace(kernel=gradient_kernel),
    )
    with pytest.raises(
        RuntimeError,
        match={
            "convergence": "SCF did not converge",
            "gradient": "gradient failed",
            "export": "export failed",
        }[failure],
    ):
        reference_energy_forces(
            engine,
            None,
            export=export,
            synchronize=lambda: events.append("synchronize"),
            clock=lambda: events.append("clock") or 10.0,
        )
    assert (
        events
        == ["synchronize", "clock"]
        + {
            "convergence": [],
            "gradient": ["gradient"],
            "export": ["gradient", "export"],
        }[failure]
    )


def test_source_identity_includes_public_and_geometry_consumers() -> None:
    """A native binary digest cannot identify an uncommitted Python force fix."""
    hashes = source_hashes()
    root = Path(__file__).resolve().parents[2]
    for path in (
        "python/generativeqc/calculator.py",
        "python/generativeqc/ks.py",
        "python/generativeqc/_stationary_cuda.py",
        "python/generativeqc/_stationary_composite_cuda.py",
        "src/scf/cuda/direct_jk.cpp",
        "src/scf/cuda/direct_jk_kernels.cu",
        "src/scf/cuda/direct_jk_kernels.hpp",
        "src/scf/cuda/direct_jk_plan.hpp",
    ):
        assert hashes[path] == hashlib.sha256((root / path).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "path",
    (
        "python/generativeqc_compiler/integral/direct_recurrence_cuda.py",
        "python/generativeqc_compiler/integral/direct_order2_shell_cuda.py",
    ),
)
def test_source_identity_tracks_direct_generated_header_owners(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """Dirty header-emitter bytes must remain identifiable beside the binary hash."""
    target = Path(__file__).resolve().parents[2] / path
    read_bytes = Path.read_bytes
    original = source_hashes()
    assert original[path] == hashlib.sha256(read_bytes(target)).hexdigest()
    changed_bytes = read_bytes(target) + b"\n# provenance regression fixture\n"
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda source: changed_bytes if source == target else read_bytes(source),
    )
    changed = source_hashes()
    assert changed[path] == hashlib.sha256(changed_bytes).hexdigest()
    assert {key for key in changed if changed[key] != original[key]} == {path}


@pytest.mark.parametrize("fail", [False, True])
def test_reference_vv10_screening_matches_force_module_and_restores_defaults(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """SCF-only configuration leaves the independently imported force mask stale."""
    scf = SimpleNamespace(NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD=1e-10)
    forces = SimpleNamespace(NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD=2e-10)
    monkeypatch.setitem(sys.modules, "gpu4pyscf.dft", SimpleNamespace(numint=scf))
    monkeypatch.setitem(sys.modules, "gpu4pyscf.grad", SimpleNamespace(rks=forces))
    try:
        with reference_vv10_domain(1e-8) as domain:
            assert domain == {"scf": 1e-8, "forces": 1e-8}
            assert scf.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD == 1e-8
            assert forces.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD == 1e-8
            if fail:
                raise RuntimeError("reference SCF failed")
    except RuntimeError:
        assert fail
    assert scf.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD == 1e-10
    assert forces.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD == 2e-10


@pytest.mark.parametrize(
    "mutation",
    [
        "force",
        "shape",
        "nan",
        "convergence",
        "timing",
        "duplicate",
        "missing",
        "old-schema",
        "no-host-export",
    ],
)
def test_validate_checks_every_repeat(mutation: str) -> None:
    reference = run()
    native = copy.deepcopy(reference)
    native["native_schedule_policy"] = "automatic-generated-SPD/canonical-through-f"
    row = native["records"][3]
    if mutation == "force":
        row["forces"][0][0] = 1e-4
    elif mutation == "shape":
        row["forces"] = [[0.0, 0.0, 0.0]]
    elif mutation == "nan":
        row["energy"] = float("nan")
    elif mutation == "convergence":
        row["converged"] = False
    elif mutation == "timing":
        row["complete_seconds"] = float("inf")
    elif mutation == "duplicate":
        row["repeat"] = 0
    elif mutation == "old-schema":
        native["schema"] = "generativeqc.readme-omol25.v2"
    elif mutation == "no-host-export":
        native["protocol"].pop("force_return")
    else:
        native["records"].pop()
    with pytest.raises(ValueError):
        validate(native, reference)


def test_collect_binds_independent_reference_and_preserves_timeouts(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "3"
    directory.mkdir()
    reference = run()
    reference_path = directory / "reference.json"
    reference_path.write_text(json.dumps(reference))
    native = copy.deepcopy(reference)
    native["native_schedule_policy"] = "automatic-generated-SPD/canonical-through-f"
    native["reference_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    (directory / "native.json").write_text(json.dumps(native))
    for engine in ("native", "reference"):
        (directory / f"{engine}.outcome").write_text(
            '{"exit_code":0,"time_limit_seconds":900}'
        )
    point = collect(tmp_path, 3)
    assert point["engines"]["native"]["medians"]["warm"] == 1.0
    assert (
        point["engines"]["native"]["native_schedule_policy"]
        == native["native_schedule_policy"]
    )
    assert "forces" not in point["engines"]["native"]["records"][0]
    assert "forces" in point["independent_references"][0]
    native["reference_sha256"] = "wrong"
    (directory / "native.json").write_text(json.dumps(native))
    with pytest.raises(ValueError, match="different independent"):
        collect(tmp_path, 3)
    (directory / "native.outcome").write_text(
        '{"exit_code":124,"time_limit_seconds":900}'
    )
    native.update(status="running", stage="cold/0")
    (directory / "native.json").write_text(json.dumps(native))
    point = collect(tmp_path, 3)
    assert point["engines"]["native"]["status"] == "timeout"
    assert "medians" not in point["engines"]["native"]
    (directory / "native.outcome").write_text(
        '{"exit_code":null,"time_limit_seconds":900,"reason":"reference_unavailable"}'
    )
    assert (
        collect(tmp_path, 3)["engines"]["native"]["status"] == "reference_unavailable"
    )
    native.update(
        status="unsupported",
        stage="native/capability",
        records=[],
        native_capabilities={
            "supported_properties": ["energy"],
            "maximum_basis_angular_momentum": 3,
        },
    )
    (directory / "native.json").write_text(json.dumps(native))
    (directory / "native.outcome").write_text(
        '{"exit_code":1,"time_limit_seconds":900}'
    )
    assert collect(tmp_path, 3)["engines"]["native"]["status"] == "unsupported"


@pytest.mark.parametrize("exit_code", [124, 137])
def test_collect_preserves_timeout_before_first_journal(
    tmp_path: Path, exit_code: int
) -> None:
    """Setup can exhaust the process limit before the first raw JSON exists."""
    directory = tmp_path / "3"
    directory.mkdir()
    (directory / "native.outcome").write_text(
        json.dumps({"exit_code": exit_code, "time_limit_seconds": 900})
    )
    point = collect(tmp_path, 3)
    assert point["engines"]["native"]["status"] == "timeout"
    assert "medians" not in point["engines"]["native"]
    assert point["engines"]["reference"]["status"] == "not_run"


def test_collect_retains_complete_seed_cost_and_experimental_policy(
    tmp_path: Path,
) -> None:
    """A compact publication must not lose costs or relabel opt-ins as defaults."""
    directory = tmp_path / "3"
    directory.mkdir()
    reference = run()
    reference_path = directory / "reference.json"
    reference_path.write_text(json.dumps(reference))
    native = copy.deepcopy(reference)
    native.update(
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        native_experiment={
            "preliminary_provider": "lda16",
            "force_active_ao_cutoff": 1e-16,
        },
        preliminary_density={
            "complete_source_seconds": 0.3,
            "source_fock_builds": None,
        },
        target_prepare_seconds=0.2,
        preliminary_wrapper_seconds=0.4,
    )
    (directory / "native.json").write_text(json.dumps(native))
    for engine in ("reference", "native"):
        (directory / f"{engine}.outcome").write_text('{"exit_code":0}')
    point = collect(tmp_path, 3)
    for key in (
        "native_experiment",
        "preliminary_density",
        "target_prepare_seconds",
        "preliminary_wrapper_seconds",
    ):
        assert point["engines"]["native"][key] == native[key]
    pytest.importorskip("matplotlib")
    figure([point], tmp_path, phase="cold", filename="cold.svg")
    svg = (tmp_path / "cold.svg").read_text()
    assert "experimental AO/seed policy" in svg
    assert "Complete cold SCF" in svg


@pytest.mark.parametrize(
    "experiment,label",
    [
        ({"force_max_device_bytes": 4 << 30}, "experimental force capacity"),
        ({"force_max_host_bytes": 4 << 30}, "experimental force capacity"),
        ({"force_max_device_bytes": 0}, "experimental force capacity"),
        ({}, "automatic through-f"),
        (
            {"force_max_device_bytes": None, "force_max_host_bytes": None},
            "automatic through-f",
        ),
        ({"force_active_ao_cutoff": 1e-16}, "experimental AO selection"),
        ({"preliminary_provider": "lda16"}, "experimental AO/seed policy"),
        (
            {"force_active_ao_cutoff": 1e-16, "force_max_host_bytes": 4 << 30},
            "explicit force capacity",
        ),
    ],
)
def test_plot_labels_explicit_force_capacity(
    tmp_path: Path, experiment: dict, label: str
) -> None:
    """A budget-only opt-in must not inherit an automatic/default policy label."""
    pytest.importorskip("matplotlib")
    rows = run()["records"]
    point = {
        "atoms": 3,
        "protocol": {"aos": 58},
        "engines": {
            "native": {
                "status": "measured",
                "records": rows,
                "native_schedule_policy": "automatic-generated-SPD/canonical-through-f",
                "native_experiment": experiment,
            },
            "reference": {"status": "measured", "records": rows},
        },
    }
    figure([point], tmp_path)
    svg = (tmp_path / "omol25.svg").read_text()
    assert label in svg
    if "capacity" in label:
        assert "GenerativeQC direct (automatic through-f)" not in svg


def test_plot_keeps_variable_iteration_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    original = Axes.errorbar
    observed = []

    def capture(self: Axes, values: list, times: list, **kwargs: object) -> object:
        observed.append((values, times, kwargs["yerr"]))
        return original(self, values, times, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", capture)
    rows = [
        dict(
            sample("warm", 0, index),
            complete_seconds=float(index + 1),
            iterations=index + 1,
        )
        for index in range(5)
    ]
    figure(
        [
            {
                "atoms": 3,
                "protocol": {"aos": 58},
                "engines": {
                    engine: {"status": "measured", "records": rows}
                    for engine in ("native", "reference")
                },
            }
        ],
        tmp_path,
    )
    assert observed == [([58], [3.0], [[2.0], [2.0]])] * 2
    assert (tmp_path / "omol25.svg").is_file()


def test_check_record_rejects_missing_force() -> None:
    row = sample("warm", 0)
    row["forces"] = None
    assert not check_record(row, sample("cold", 0))["gate"]


@pytest.mark.parametrize("properties", [["energy"], []])
def test_preflight_rejects_energy_only_and_allows_qualified_f_shells(
    properties: list[str],
) -> None:
    with pytest.raises(NotImplementedError, match="analytic forces"):
        require_force_endpoint(
            {
                "supported_properties": properties,
                "maximum_basis_angular_momentum": 3,
            }
        )
    require_force_endpoint(
        {
            "supported_properties": ["energy", "forces"],
            "maximum_basis_angular_momentum": 3,
        }
    )


def test_retained_omol25_basis_preserves_f_and_diffuse_shells() -> None:
    root = Path(__file__).parents[2] / "benchmarks/results/omol25-wb97mv-20261001"
    basis = load_basis(root / "def2-tzvpd-ho.json")
    restored = import_bse(
        root / "def2-tzvpd-ho.bse.json",
        source=basis.provenance.source,
        source_version=basis.provenance.version.partition("; basis-data=")[0],
        license=basis.provenance.license,
        representation="spherical",
    )
    assert restored.identity == basis.identity
    assert max(shell.angular_momentum for shell in basis.by_element[8].shells) == 3
    assert [
        protocol(atoms, basis, GridSpec(), 5)["aos"] for atoms in (3, 6, 12, 24, 48, 96)
    ] == [58, 116, 232, 464, 928, 1856]


def test_unsupported_native_plot_has_no_native_latency(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    figure(
        [
            {
                "atoms": 3,
                "protocol": {"aos": 58},
                "engines": {
                    "native": {"status": "unsupported", "records": []},
                    "reference": {
                        "status": "measured",
                        "records": [sample("warm", 0, i) for i in range(5)],
                    },
                },
            }
        ],
        tmp_path,
    )
    assert (
        "f-shell force API unavailable (no native timings)"
        in (tmp_path / "omol25.svg").read_text()
    )


@pytest.mark.parametrize("failure", ["SCF did not converge", "stationary force failed"])
@pytest.mark.parametrize("has_diagnostic", [False, True])
def test_native_failure_is_journaled_without_a_force_work_record(
    failure: str, has_diagnostic: bool
) -> None:
    """Execute the real row/retention code with a failed strict=False endpoint."""
    record = {"records": [], "stage": "cold/0"}
    saved = []
    # Preserve any available solver history even if the later force stage fails.
    # Missing diagnostics remain null instead of synthesizing an empty history.
    diagnostic = {"history": [{"iteration": 1, "energy_change": None}]}
    item = SimpleNamespace(
        energy=-76.0,
        forces=None,
        converged=failure != "SCF did not converge",
        status=5,
        status_message=failure,
        iterations=100,
        fock_builds=101,
        energy_change=0.1,
        density_rms=0.2,
        physical_residual_rms=0.3,
        ks_diagnostic=SimpleNamespace(to_payload=lambda: copy.deepcopy(diagnostic))
        if has_diagnostic
        else None,
        warm_start_used=False,
        warm_start_fallback=False,
    )
    row = native_endpoint_record(
        item,
        phase="cold",
        geometry_index=0,
        repeat=0,
        seconds=2.5,
        prepare=0.2,
        owner=None,
        force_work=None,
    )

    def save(stage: str) -> None:
        assert stage == "cold/0"
        assert record["records"] == [row]
        saved.append(copy.deepcopy(record))

    with pytest.raises(RuntimeError, match="independent energy/force gate failed"):
        retain_endpoint_record(record, row, sample("cold", 0), save)
    assert saved == [record]
    assert len(record["records"]) == 1
    failed = record["records"][0]
    assert failed["detail"] == failure and failed["status"] == 5
    assert failed["iterations"] == 100 and failed["seconds"] == 2.5
    assert failed["native_force_components"] is None
    assert failed["physical_residual_rms"] == 0.3
    assert failed["native_ks_diagnostic"] == (diagnostic if has_diagnostic else None)
    assert failed["gate"] is False
    assert failed["forces"] is None
    assert failed["native_scf_ao_work"] is None
    assert failed["complete_seconds"] == 2.7


@pytest.mark.parametrize(
    "mutation", [None, "energy", "force", "shape", "nan", "convergence", "status"]
)
def test_retention_uses_independent_gate_before_journaling(
    mutation: str | None,
) -> None:
    row = sample("warm", 0)
    oracle = sample("cold", 0)
    if mutation == "energy":
        row["energy"] += 2e-8
    elif mutation == "force":
        row["forces"] = [[2e-7, 0.0, 0.0]] * 3
    elif mutation == "shape":
        row["forces"] = [[0.0, 0.0, 0.0]]
    elif mutation == "nan":
        row["forces"] = [[float("nan"), 0.0, 0.0]] * 3
    elif mutation == "convergence":
        row["converged"] = False
    elif mutation == "status":
        row["status"] = 5
    record = {"stage": "warm/0", "records": []}
    saved = []

    def save(stage: str) -> None:
        assert stage == "warm/0"
        assert row["gate"] is (mutation is None)
        saved.append(copy.deepcopy(record))

    if mutation is None:
        retain_endpoint_record(record, row, oracle, save)
        assert row["energy_error"] == row["force_error"] == 0.0
    else:
        with pytest.raises(RuntimeError, match="independent energy/force gate failed"):
            retain_endpoint_record(record, row, oracle, save)
    assert saved == [record]
    assert record["records"] == [row]
