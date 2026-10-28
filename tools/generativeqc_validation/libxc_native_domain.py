"""Qualify the current automatic worker against the existing boundary profile.

The historical bulk candidate and installed automatic work-domain worker are
separate recipes. Keep the established cases and tolerances, but execute the
current common native point emitter; never relabel a legacy candidate receipt.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

import numpy as np
from generativeqc_compiler.common.compiler_cache import (
    resolve_compiler_cache,
    run_cached_compiler,
)
from generativeqc_compiler.common.provenance import canonical_hash, file_hash
from generativeqc_compiler.xc.automatic_point import emit_automatic_point
from generativeqc_compiler.xc.libxc_bulk_capabilities import (
    STAGE_EVIDENCE_SCHEMA,
    functional_capability,
)
from generativeqc_compiler.xc.production_domain_cases import numerical_cases
from generativeqc_compiler.xc.production_domain_controls import run_control_case

SCHEMA = "generativeqc.libxc-native-domain.v1"
GATES = {"rtol": 2e-6, "atol": 1e-8}


def packed_cases(profile: Any, spin: str) -> tuple[list[str], np.ndarray]:
    """Project the existing physical fixtures to the native two-spin point ABI."""
    cases = numerical_cases(profile, spin=spin)
    rows = []
    for case in cases:
        if spin == "polarized":
            rho, gradient, tau = case.rho, case.gradient, case.tau
        else:
            rho = [case.rho[0] / 2] * 2
            gradient = [np.asarray(case.gradient[0]) / 2] * 2
            tau = [case.tau[0] / 2] * 2
        rows.append([*rho, *gradient[0], *gradient[1], *tau])
    return [case.case_id for case in cases], np.asarray(rows)


def same_fixture_inputs(recorded: Any, expected: np.ndarray) -> bool:
    """Allow at most two adjacent float64 values for fixture reconstruction.

    Python 3.12 changed builtin floating sum, moving polarized isoorbital tau
    by one ULP relative to 3.11. This is not a numerical-output tolerance:
    zero/sign/shape stay exact, and near-isoorbital cases remain distinct.
    """
    actual = np.asarray(recorded)
    if actual.dtype.kind not in "fiu":
        return False
    actual = actual.astype(np.float64, copy=False)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        return False
    if not np.isfinite(expected).all():
        return False
    if not np.array_equal(actual == 0, expected == 0):
        return False
    if not np.array_equal(np.signbit(actual), np.signbit(expected)):
        return False
    lower = np.nextafter(np.nextafter(expected, -np.inf), -np.inf)
    upper = np.nextafter(np.nextafter(expected, np.inf), np.inf)
    return bool(((actual >= lower) & (actual <= upper)).all())


def _point(name: str, family: str, spin: str) -> Any:
    # LDA RKS has an explicit U1 density worker. GGA/tau-MGGA RKS uses
    # the installed P2 differential on equal physical spin coordinates.
    layout: Literal["polarized", "unpolarized"] = (
        "unpolarized" if family == "lda" and spin == "unpolarized" else "polarized"
    )
    return emit_automatic_point(name, spin=layout)


def qualify_native_domain(name: str, root: Path, output: Path) -> dict[str, Any]:
    """Compile and execute both layouts with the pinned independent Libxc oracle."""
    from test_libxc_automatic_point_native import _evaluate, _native_source, _oracle

    capability = functional_capability(name)
    profile = capability.production_domain_profile
    cache = resolve_compiler_cache()
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("native domain requires a C++ compiler")
    version = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    records = []
    for spin in profile.spin_layouts:
        point = _point(name, capability.family, spin)
        directory = output / name / spin
        directory.mkdir(parents=True, exist_ok=True)
        source, obj, binary = (
            directory / p for p in ("point.cpp", "point.o", "point.so")
        )
        source.write_text(_native_source(point))
        for command in (
            [
                compiler,
                "-std=c++20",
                "-O2",
                "-fPIC",
                "-I",
                str(root / "src"),
                "-I",
                str(root / "include"),
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            [compiler, "-shared", str(obj), "-o", str(binary)],
        ):
            execution = run_cached_compiler(
                command, 180, label="native Libxc domain qualification"
            )
            if execution.returncode or execution.timed_out:
                raise RuntimeError(execution.stderr)
        library = ctypes.CDLL(str(binary.resolve()))
        pointer = ctypes.POINTER(ctypes.c_double)
        library.evaluate.argtypes = (pointer, pointer, ctypes.c_size_t)
        library.evaluate.restype = ctypes.c_int
        ids, inputs = packed_cases(profile, spin)
        actual = _evaluate(library, inputs)
        polarized = spin == "polarized"
        observed = (
            actual[:, :11]
            if polarized
            else np.column_stack(
                (
                    actual[:, 0],
                    actual[:, 1:3].mean(axis=1),
                    0.5 * (actual[:, 3:6] + actual[:, 6:9]),
                    actual[:, 9:11].mean(axis=1),
                )
            )
        )
        reference = _oracle(name, inputs, int(polarized))
        invalid = inputs[[0]].copy()
        invalid[:, 0] = np.nan
        invalid_rejected = bool(np.all(_evaluate(library, invalid)[:, 11] == 0))
        controls = [
            run_control_case(name, spin=spin, case_id=case)[0]
            for case in profile.case_ids_for_spin(spin)
            if case.startswith("control/")
        ]
        records.append(
            {
                "spin": spin,
                "case_ids": ids,
                "inputs": inputs.tolist(),
                "observed": observed.tolist(),
                "reference": reference.tolist(),
                "valid": actual[:, 11].tolist(),
                "controls": controls,
                "native_invalid_rejected": invalid_rejected,
                "point_source_identity": point.source_identity,
                "mathematics": point.mathematical_contract,
                "translation_unit_sha256": file_hash(source),
                "object_sha256": file_hash(obj),
                "binary_sha256": file_hash(binary),
            }
        )
    result = {
        "schema": SCHEMA,
        "subject_identity": capability.identity,
        "profile": profile.to_payload(),
        "gates": GATES,
        "oracle": "PySCF-2.14.0/Libxc-7.0.0",
        "compiler": {
            "sha256": file_hash(Path(compiler)),
            "version": version,
            "cache": cache.version,
        },
        "records": records,
    }
    result["identity"] = canonical_hash(result)
    return result


def native_domain_stage(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Recompute named case outcomes and exact current worker/source bindings."""
    capability = functional_capability(name)
    profile = capability.production_domain_profile
    if (
        result.get("schema") != SCHEMA
        or result.get("subject_identity") != capability.identity
        or result.get("profile") != profile.to_payload()
        or result.get("gates") != GATES
        or result.get("oracle") != "PySCF-2.14.0/Libxc-7.0.0"
        or result.get("identity")
        != canonical_hash({k: v for k, v in result.items() if k != "identity"})
    ):
        raise ValueError("native production-domain identity/profile mismatch")
    if [r["spin"] for r in result["records"]] != list(profile.spin_layouts):
        raise ValueError("native production-domain missing spin coverage")
    failures = []
    for row in result["records"]:
        spin = row["spin"]
        point = _point(name, capability.family, spin)
        # Normalize tuple/list differences introduced by JSON persistence.
        if row["point_source_identity"] != point.source_identity or canonical_hash(
            row["mathematics"]
        ) != canonical_hash(point.mathematical_contract):
            raise ValueError(
                "native production-domain worker/domain/derivative ABI changed"
            )
        ids, inputs = packed_cases(profile, spin)
        if row["case_ids"] != ids or not same_fixture_inputs(row["inputs"], inputs):
            raise ValueError("native production-domain fixture coverage mismatch")
        observed, reference = np.asarray(row["observed"]), np.asarray(row["reference"])
        expected_shape = (len(ids), 11 if spin == "polarized" else 6)
        if (
            observed.shape != expected_shape
            or reference.shape != expected_shape
            or len(row["valid"]) != len(ids)
        ):
            raise ValueError("native production-domain output ABI mismatch")
        for index, case in enumerate(ids):
            if (
                row["valid"][index] != 1
                or not np.isfinite(observed[index]).all()
                or not np.isfinite(reference[index]).all()
                or not np.allclose(
                    observed[index],
                    reference[index],
                    rtol=GATES["rtol"],
                    atol=GATES["atol"],
                )
            ):
                failures.append(
                    f"{spin}:{case}: native E/vxc differs from independent Libxc"
                )
        expected_controls = [
            case
            for case in profile.case_ids_for_spin(spin)
            if case.startswith("control/")
        ]
        if [r["case_id"] for r in row["controls"]] != expected_controls:
            raise ValueError("native production-domain missing controls")
        for control in row["controls"]:
            if control["spin"] != spin or control["status"] != "pass":
                failures.append(f"{spin}:{control['case_id']}: control failed")
        if row["native_invalid_rejected"] is not True:
            failures.append(
                f"{spin}:control/invalid-nonfinite: native worker accepted invalid input"
            )
        for field in ("translation_unit_sha256", "object_sha256", "binary_sha256"):
            value = row[field]
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"native production-domain invalid {field}")
    envelope = {
        "schema": STAGE_EVIDENCE_SCHEMA,
        "subject_identity": capability.identity,
        "stage": "production-domain",
        "status": "fail" if failures else "pass",
        "reason": "; ".join(failures) if failures else None,
        "evidence": f"native-domain:{name}#sha256={result['identity']}",
    }
    if not failures:
        envelope["qualification"] = profile.to_payload()
    return envelope
