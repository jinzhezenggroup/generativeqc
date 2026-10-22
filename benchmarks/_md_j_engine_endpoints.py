"""Complete, matched PBE0 energy/force endpoints for the MD-J campaign.

This helper is intentionally not a second process runner. The benchmark driver
owns finite Slurm admission, fresh processes, timeouts and source/binary receipts.
It supplies frozen explicit shells; synthetic raw-J densities never enter SCF.
Instrumentation runs separately and is never accepted as performance timing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

SCHEMA = "md-j-energy-force-v1"
CASES = {"water32-sto-3g": ("sto-3g", 224), "water32-def2-svp": ("def2-svp", 768)}
ENERGY_GATE = 3e-9
FORCE_GATE = 3e-8
RESIDUAL_GATE = 1e-9


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def endpoint_settings() -> dict[str, Any]:
    """The predeclared scientific prescription, identical for both engines."""
    return {
        "method": "pbe0-rks",
        "precision": "fp64",
        "device": "cuda",
        "basis_representation": "spherical",
        "density_fitting": "none",
        "initial_guess": None,
        "max_iterations": 180,
        "energy_tolerance": 1e-11,
        "density_tolerance": 1e-9,
        "screening_tolerance": 1e-12,
        "diis_history": 8,
        "grid": {"radial_points": 24, "angular_polar": 8, "angular_azimuth": 16},
        "tile_points": 256,
        "xc_schedule": "device_fused",
        "properties": ["energy", "forces"],
        "warm_start": False,
        "coordinate_units": "bohr",
        "energy_units": "hartree",
        "force_units": "hartree/bohr",
        "moved_geometry": "all-coordinate 0.002*sin(1+3*atom+axis) bohr displacement",
    }


def _scientific_input(case: dict[str, Any]) -> dict[str, Any]:
    """Freeze only physical inputs; exclude the separate synthetic J density."""
    result = {
        key: case[key]
        for key in ("name", "atoms", "basis", "shells", "charge", "multiplicity")
    }
    result["nao"] = case.get("nao", case.get("expected_nao"))
    basis, nao = CASES[result["name"]]
    if (
        len(result["atoms"]) != 96
        or result["basis"] != basis
        or result["nao"] != nao
        or result["charge"] != 0
        or result["multiplicity"] != 1
    ):
        raise ValueError(
            "endpoint requires the fixed 96-atom closed-shell prescription"
        )
    if sum(2 * shell["angular_momentum"] + 1 for shell in result["shells"]) != nao:
        raise ValueError("explicit spherical shells do not match the required AO count")
    for symbol, xyz in result["atoms"]:
        if (
            symbol not in ("O", "H")
            or len(xyz) != 3
            or not all(map(math.isfinite, xyz))
        ):
            raise ValueError("invalid frozen molecular geometry")
    if [atom[0] for atom in result["atoms"]].count("O") != 32:
        raise ValueError("endpoint requires exactly 32 waters")
    # JSON roundtrip also rejects nonfinite shell coefficients/exponents.
    return json.loads(json.dumps(result, allow_nan=False))


def _geometries(inputs: dict[str, Any]) -> dict[str, Any]:
    base = [[float(x) for x in atom[1]] for atom in inputs["atoms"]]
    moved = [
        [
            value + 0.002 * math.sin(1 + 3 * atom + axis)
            for axis, value in enumerate(xyz)
        ]
        for atom, xyz in enumerate(base)
    ]
    return {
        label: {"coordinates": coordinates, "sha256": _hash(coordinates)}
        for label, coordinates in (("original", base), ("moved", moved))
    }


@contextmanager
def _capture_stderr() -> Iterator[Any]:
    """Capture native fprintf diagnostics and preserve them in worker stderr."""
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as stream:
        try:
            os.dup2(stream.fileno(), 2)
            yield stream
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            stream.seek(0)
            # Re-emit verbatim to the campaign's retained subprocess log.
            while data := stream.read(65536):
                os.write(2, data)


def _work_receipt(stderr: str, engine: str) -> dict[str, Any]:
    work, selected, calls = [], [], []
    for line in stderr.splitlines():
        if line.startswith(("MD_J_WORK ", "MD_J_ENGINE_WORK ")):
            fields = dict(part.split("=", 1) for part in line.split()[1:])
            row = {key: int(value) for key, value in fields.items()}
            (selected if line.startswith("MD_J_ENGINE_WORK ") else work).append(row)
        elif line.startswith("MD_J_NORMAL_COUNTS="):
            calls.append(int(line.split("=", 1)[1]))
    return {
        "requested_engine": engine,
        "work_rows": work,
        "selected_source_rows": selected,
        "destroyed_owner_md_calls": calls,
        "observation": "native executed source counters, not environment or preparation schedule",
    }


def _result_record(item: Any) -> dict[str, Any]:
    diagnostic = item.ks_diagnostic
    forces = None if item.forces is None else item.forces.tolist()
    return {
        "energy": item.energy,
        "forces": forces,
        "forces_sha256": _hash(forces),
        "converged": item.converged,
        "succeeded": item.succeeded,
        "status": item.status,
        "status_message": item.status_message,
        "iterations": item.iterations,
        "fock_builds": item.fock_builds,
        "backend": item.executed_backend,
        "energy_change": item.energy_change,
        "density_rms": item.density_rms,
        "physical_residual_max": None
        if diagnostic is None
        else diagnostic.physical_residual_max,
        "warm_start_used": item.warm_start_used,
        "initial_density_used": None
        if diagnostic is None
        else diagnostic.initial_density_used,
        "grid_points": None if diagnostic is None else diagnostic.grid_points,
        "basis_metadata": item.basis_metadata,
        "precision": item.precision,
        "ks_diagnostic": None if diagnostic is None else diagnostic.to_payload(),
    }


def _run_sequence(
    inputs: dict[str, Any],
    *,
    engine: str,
    instrumented: bool,
    warm_repeats: int,
    factory: Callable[[], Any],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Measure public synchronous calls; cold includes construction/preparation."""
    geometries = _geometries(inputs)
    phases: list[dict[str, Any]] = []
    calculator, owner = None, None
    teardown_seconds = None
    with _capture_stderr() as captured:
        try:
            for label in (
                "cold",
                *(f"warm-{index + 1}" for index in range(warm_repeats)),
                "moved",
            ):
                offset = captured.tell()
                start = clock()
                if label == "cold":
                    calculator = factory()
                    owner = calculator.prepare_batch(
                        [inputs["atoms"]],
                        charges=[inputs["charge"]],
                        multiplicities=[inputs["multiplicity"]],
                        warm_start=False,
                    )
                assert owner is not None
                # Geometry transfer, invalidation and rebuilding are inside the
                # moved timer. No density is imported, saved or restored here.
                result = owner.execute(
                    [geometries["moved"]["coordinates"]] if label == "moved" else None,
                    properties=("energy", "forces"),
                    strict=True,
                )
                seconds = clock() - start
                if len(result.items) != 1:
                    raise RuntimeError(
                        "endpoint must return exactly one complete molecular result"
                    )
                captured.flush()
                end = captured.tell()
                captured.seek(offset)
                stderr = captured.read(end - offset).decode(errors="replace")
                captured.seek(end)
                phases.append(
                    {
                        "phase": label,
                        "geometry_sha256": geometries[
                            "moved" if label == "moved" else "original"
                        ]["sha256"],
                        "seconds": None if instrumented else seconds,
                        "execution": _work_receipt(stderr, engine),
                        **_result_record(result.items[0]),
                    }
                )
        finally:
            start = clock()
            try:
                if owner is not None:
                    owner.close()
            finally:
                if calculator is not None:
                    calculator.clear_cache()
            teardown_seconds = None if instrumented else clock() - start
        captured.seek(0)
        stderr = captured.read().decode(errors="replace")
    settings = endpoint_settings()
    return {
        "schema": SCHEMA,
        "engine": engine,
        "instrumented": instrumented,
        "settings": settings,
        "settings_sha256": _hash(settings),
        "scientific_input": inputs,
        "scientific_input_sha256": _hash(inputs),
        "geometries": geometries,
        "natoms": len(inputs["atoms"]),
        "nao": inputs["nao"],
        "warm_repeats": warm_repeats,
        "process_cold": True,
        "context_primed": False,
        "supplied_density": False,
        "phases": phases,
        "execution": _work_receipt(stderr, engine),
        "teardown_seconds": teardown_seconds,
        "timing_scope": {
            "cold": "Calculator construction + explicit-basis preparation + full SCF energy and analytic forces; imports, result serialization and retained-owner teardown excluded",
            "warm": "same prepared owner, full energy and analytic forces, same geometry; converged-density reuse disabled",
            "moved": "same owner, deterministic geometry upload/invalidation/rebuild + full SCF energy and analytic forces; converged-density reuse disabled",
            "teardown": "owner.close + Calculator.clear_cache, separately timed after moved endpoint",
        },
    }


def run_endpoint_worker(
    case: dict[str, Any], *, engine: str, instrumented: bool, warm_repeats: int = 3
) -> dict[str, Any]:
    """Run only under the enclosing fresh-process runner's finite allocation."""
    from benchmarks.md_j_engine_acceptance import _finite_slurm

    _finite_slurm()
    if engine not in ("incumbent", "reference") or not 3 <= warm_repeats <= 100:
        raise ValueError("endpoint requires a native engine and 3..100 warm repeats")
    inputs = _scientific_input(case)
    os.environ["GENERATIVEQC_MD_J_ENGINE"] = engine
    os.environ.pop("GENERATIVEQC_DISABLE_MD_J", None)
    for key in ("GENERATIVEQC_MD_J_COUNTS", "GENERATIVEQC_MD_J_WORK_COUNTS"):
        if instrumented:
            os.environ[key] = "1"
        else:
            os.environ.pop(key, None)
    # Import modules only: no Calculator, context, device synchronization or
    # basis owner may precede the cold timer in this fresh process.
    from generativeqc import Calculator, GridSpec, KsOptions, Primitive, Shell

    def factory() -> Any:
        settings = endpoint_settings()
        options = {
            key: settings[key]
            for key in (
                "method",
                "precision",
                "device",
                "basis_representation",
                "density_fitting",
                "initial_guess",
                "max_iterations",
                "energy_tolerance",
                "density_tolerance",
                "screening_tolerance",
                "diis_history",
            )
        }
        return Calculator(
            basis=tuple(
                Shell(
                    row["atom_index"],
                    row["angular_momentum"],
                    tuple(Primitive(*primitive) for primitive in row["primitives"]),
                )
                for row in inputs["shells"]
            ),
            ks_options=KsOptions(
                grid=GridSpec(**settings["grid"]),
                tile_points=settings["tile_points"],
                xc_schedule=settings["xc_schedule"],
            ),
            **options,
        )

    return _run_sequence(
        inputs,
        engine=engine,
        instrumented=instrumented,
        warm_repeats=warm_repeats,
        factory=factory,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (float, int))
        and math.isfinite(value)
        and value > 0
    )


def _selection(record: dict[str, Any], engine: str) -> None:
    rows = record["work_rows"]
    selected = record["selected_source_rows"]
    fields = {
        "fock",
        "bra",
        "ket",
        "uniform_roots",
        "uniform_directions",
        "uniform_summands",
        "residual_candidates",
        "residual_tasks",
        "residual_roots",
    }
    selected_fields = {
        "fock",
        "bra",
        "ket",
        "source_tiles",
        "screened_tiles",
        "bra_active_primitives",
        "ket_active_primitives",
    }
    _require(
        bool(rows) and len(rows) % 25 == 0,
        "missing/incomplete executing MD work census",
    )
    for values, keys in ((rows, fields), (selected, selected_fields)):
        for row in values:
            _require(
                set(row) == keys
                and all(type(value) is int and value >= 0 for value in row.values()),
                "malformed executing source census",
            )
            _require(row["fock"] > 0, "invalid executing Fock identity")
    for start in range(0, len(rows), 25):
        group = rows[start : start + 25]
        _require(
            {(row["bra"], row["ket"]) for row in group}
            == {(a, b) for a in range(5) for b in range(5)},
            "incomplete angular work domain",
        )
        _require(
            len({row["fock"] for row in group}) == 1, "mixed executing Fock identities"
        )
    _require(
        sum(row["uniform_roots"] for row in rows) > 0,
        "no actual MD source roots executed",
    )
    if engine == "reference":
        _require(
            len(selected) * 25 == len(rows) * 15,
            "reference selection not proven for every observed MD call",
        )
        for start in range(0, len(selected), 15):
            group = selected[start : start + 15]
            _require(
                {(row["bra"], row["ket"]) for row in group}
                == {(a, b) for a in range(5) for b in range(a, 5)},
                "incomplete reference source domain",
            )
            _require(
                {row["fock"] for row in group} == {rows[start // 15 * 25]["fock"]},
                "reference source/Fock identity mismatch",
            )
        _require(
            sum(row["source_tiles"] for row in selected) > 0,
            "reference source did not execute tiles",
        )
    else:
        _require(not selected, "incumbent unexpectedly executed reference source")


def _validate_sample(sample: dict[str, Any]) -> None:
    _require(sample["schema"] == SCHEMA, "wrong endpoint schema")
    _require(sample["process_status"] == "passed", "worker failed or timed out")
    _require(sample["engine"] in ("incumbent", "reference"), "invalid engine")
    _require(type(sample["instrumented"]) is bool, "unknown instrumentation mode")
    _require(sample["settings"] == endpoint_settings(), "changed scientific settings")
    _require(
        sample["settings_sha256"] == _hash(sample["settings"]), "settings hash mismatch"
    )
    inputs = _scientific_input(sample["scientific_input"])
    _require(
        sample["scientific_input_sha256"] == _hash(inputs),
        "molecular input hash mismatch",
    )
    _require(
        sample["natoms"] == 96 and sample["nao"] == inputs["nao"],
        "wrong full-size endpoint",
    )
    _require(
        sample["geometries"] == _geometries(inputs), "changed or stale moved geometry"
    )
    _require(
        sample["process_cold"] is True
        and sample["context_primed"] is False
        and sample["supplied_density"] is False,
        "cold state was primed or supplied density",
    )
    count = sample["warm_repeats"]
    _require(type(count) is int and 3 <= count <= 100, "insufficient warm repeats")
    labels = ["cold", *(f"warm-{index + 1}" for index in range(count)), "moved"]
    _require(
        [row["phase"] for row in sample["phases"]] == labels,
        "missing, duplicated or reordered endpoint phases",
    )
    if sample["instrumented"]:
        _selection(sample["execution"], sample["engine"])
        calls = sample["execution"]["destroyed_owner_md_calls"]
        _require(
            bool(calls) and all(type(value) is int and value >= 0 for value in calls),
            "missing native MD execution count",
        )
        _require(
            sum(calls) == len(sample["execution"]["work_rows"]) // 25,
            "executed MD census and destroyed-owner count disagree",
        )
        _require(
            sample["teardown_seconds"] is None,
            "instrumented teardown was presented as timing",
        )
    else:
        _require(
            not any(
                sample["execution"][field]
                for field in (
                    "work_rows",
                    "selected_source_rows",
                    "destroyed_owner_md_calls",
                )
            ),
            "timing worker was instrumented",
        )
        _require(_positive(sample["teardown_seconds"]), "missing teardown timing")
    for row in sample["phases"]:
        _require(
            row["converged"] is True
            and row["succeeded"] is True
            and row["status"] == 0
            and row["backend"] == "cuda",
            "incomplete, unconverged or non-CUDA endpoint",
        )
        _require(
            row["warm_start_used"] is False and row["initial_density_used"] is False,
            "endpoint reused a converged or supplied density",
        )
        _require(
            type(row["iterations"]) is int
            and 0 < row["iterations"] <= 180
            and type(row["fock_builds"]) is int
            and row["fock_builds"] > 0,
            "missing SCF work counts",
        )
        _require(
            row["precision"]["requested_mode"] == "fp64"
            and row["precision"]["effective_bits"] == 64
            and row["precision"]["mixed_stage_fock_builds"] == 0,
            "observed execution precision differs from the fixed FP64 prescription",
        )
        _require(
            math.isfinite(row["energy"])
            and math.isfinite(row["energy_change"])
            and abs(row["energy_change"]) <= 1e-11,
            "invalid energy/convergence change",
        )
        _require(
            math.isfinite(row["density_rms"]) and 0 <= row["density_rms"] <= 1e-9,
            "density convergence gate failed",
        )
        _require(
            math.isfinite(row["physical_residual_max"])
            and 0 <= row["physical_residual_max"] <= RESIDUAL_GATE,
            "physical residual gate failed",
        )
        _require(
            row["grid_points"] == 294912, "native grid differs from fixed prescription"
        )
        forces = row["forces"]
        _require(
            len(forces) == 96
            and all(len(xyz) == 3 and all(map(math.isfinite, xyz)) for xyz in forces),
            "missing/nonfinite analytic forces",
        )
        _require(row["forces_sha256"] == _hash(forces), "force payload hash mismatch")
        geometry = "moved" if row["phase"] == "moved" else "original"
        _require(
            row["geometry_sha256"] == sample["geometries"][geometry]["sha256"],
            "phase used stale geometry",
        )
        _require(
            bool(row["basis_metadata"]["orbital"]["mathematical_identity"]),
            "missing resolved basis identity",
        )
        _require(bool(row["ks_diagnostic"]["scf_domain"]), "missing native SCF domain")
        if sample["instrumented"]:
            _require(
                row["seconds"] is None, "instrumented endpoint was presented as timing"
            )
            _selection(row["execution"], sample["engine"])
            _require(
                len(row["execution"]["work_rows"]) // 25 >= row["fock_builds"],
                "MD did not cover the complete reported SCF Fock work",
            )
        else:
            _require(_positive(row["seconds"]), "invalid complete endpoint time")
            _require(
                not any(
                    row["execution"][field]
                    for field in (
                        "work_rows",
                        "selected_source_rows",
                        "destroyed_owner_md_calls",
                    )
                ),
                "instrumented phase in timing worker",
            )


def validate_endpoint_receipts(
    samples: list[dict[str, Any]], *, minimum_samples: int = 3
) -> dict[str, Any]:
    """Fail closed; matched-engine parity is not independent production accuracy."""
    failures: list[str] = []
    comparisons: list[dict[str, Any]] = []
    timing: dict[str, Any] = {}
    speedups: dict[str, float] = {}
    try:
        _require(
            type(minimum_samples) is int and minimum_samples >= 3,
            "at least three fresh-process samples required",
        )
        _require(bool(samples), "no full energy-force receipts")
        for index, sample in enumerate(samples):
            try:
                _validate_sample(sample)
            except (
                ValueError,
                KeyError,
                TypeError,
                IndexError,
                OverflowError,
            ) as error:
                raise ValueError(f"sample {index}: {error}") from error
        for field in (
            "scientific_input_sha256",
            "settings_sha256",
            "library_sha256",
            "source_identity",
            "input_sha256",
            "run_id",
            "warm_repeats",
        ):
            _require(
                len({_hash(sample[field]) for sample in samples}) == 1
                and bool(samples[0][field]),
                f"unmatched or absent {field}",
            )
        _require(
            all(type(row["pid"]) is int and row["pid"] > 0 for row in samples)
            and len({row["pid"] for row in samples}) == len(samples),
            "workers did not run in distinct fresh processes",
        )
        baseline = next(
            sample
            for sample in samples
            if sample["engine"] == "incumbent" and not sample["instrumented"]
        )
        for engine in ("incumbent", "reference"):
            clean = [
                row
                for row in samples
                if row["engine"] == engine and not row["instrumented"]
            ]
            evidence = [
                row
                for row in samples
                if row["engine"] == engine and row["instrumented"]
            ]
            _require(
                len(clean) >= minimum_samples and bool(evidence),
                f"{engine}: missing clean repeats or separate executing-source evidence",
            )
            _require(
                len({row["repetition"] for row in clean}) == len(clean),
                "duplicate repetition identity",
            )
            timing[engine] = {}
            for phase in ("cold", "warm", "moved"):
                seconds = [
                    row["seconds"]
                    for sample in clean
                    for row in sample["phases"]
                    if row["phase"].split("-")[0] == phase
                ]
                timing[engine][phase] = {
                    "samples": seconds,
                    "median": statistics.median(seconds),
                    "minimum": min(seconds),
                    "maximum": max(seconds),
                    "population_stdev": statistics.pstdev(seconds),
                }
        # Every diagnostic and timed result must match its same-geometry clean
        # incumbent result. Thus counters cannot qualify an unrelated endpoint.
        for index, sample in enumerate(samples):
            for row, expected in zip(sample["phases"], baseline["phases"], strict=True):
                # All warm repeats have the original cold geometry and must
                # reproduce its endpoint as well as the matching engine phase.
                if row["phase"].startswith("warm-"):
                    cold = sample["phases"][0]
                    _require(
                        abs(row["energy"] - cold["energy"]) <= ENERGY_GATE
                        and max(
                            abs(a - b)
                            for first, second in zip(
                                row["forces"], cold["forces"], strict=True
                            )
                            for a, b in zip(first, second, strict=True)
                        )
                        <= FORCE_GATE,
                        "same-geometry warm endpoint differs from cold",
                    )
                energy_error = abs(row["energy"] - expected["energy"])
                deltas = [
                    a - b
                    for first, second in zip(
                        row["forces"], expected["forces"], strict=True
                    )
                    for a, b in zip(first, second, strict=True)
                ]
                maximum = max(map(abs, deltas))
                _require(
                    energy_error <= ENERGY_GATE and maximum <= FORCE_GATE,
                    f"{sample['engine']} {row['phase']}: energy/force parity gate failed",
                )
                _require(
                    row["basis_metadata"]["orbital"]["mathematical_identity"]
                    == expected["basis_metadata"]["orbital"]["mathematical_identity"],
                    "resolved basis/phase identity differs",
                )
                _require(
                    row["ks_diagnostic"]["scf_domain"]
                    == expected["ks_diagnostic"]["scf_domain"],
                    "native SCF domains differ",
                )
                comparisons.append(
                    {
                        "sample": index,
                        "phase": row["phase"],
                        "energy_absolute": energy_error,
                        "force_maximum": maximum,
                        "force_rms": math.sqrt(
                            sum(x * x for x in deltas) / len(deltas)
                        ),
                    }
                )
        speedups = {
            phase: timing["incumbent"][phase]["median"]
            / timing["reference"][phase]["median"]
            for phase in ("cold", "warm", "moved")
        }
    except (
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        StopIteration,
        OverflowError,
    ) as error:
        failures.append(str(error) or "missing complete incumbent endpoint baseline")
    return {
        "status": "failed" if failures else "passed",
        "failures": failures,
        "numerical_parity_passed": not failures,
        "comparisons": comparisons,
        "energy_gate_hartree": ENERGY_GATE,
        "force_gate_hartree_per_bohr": FORCE_GATE,
        "physical_residual_gate": RESIDUAL_GATE,
        "timing": timing if not failures else {},
        "incumbent_over_reference": speedups if not failures else {},
        "default_promotion_qualified": False,
        "accuracy_scope": "matched incumbent/reference regression parity; no independent full-size grid-converged energy/force reference or production ranking",
        "timing_policy": "clean samples only; instrumented source evidence independently compared to clean same-geometry endpoints",
    }
