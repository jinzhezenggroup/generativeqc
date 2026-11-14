"""Audit retained installed CPU endpoints without running an electronic engine.

The producer remains unaccepted: this separate receipt binds its raw outputs to
an already accepted cold independent-state/full-stencil qualification. No SCF,
finite difference, matrix symmetrization, or production import occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


class Inputs:
    """Record the exact bytes actually read, including native library identity."""

    def __init__(self) -> None:
        self.members: dict[str, dict[str, Any]] = {}

    def read(self, path: Path) -> bytes:
        data = path.read_bytes()
        self.members[str(path.resolve())] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        return data

    def json(self, path: Path) -> dict[str, Any]:
        return json.loads(self.read(path))

    def arrays(self, path: Path) -> dict[str, np.ndarray]:
        with np.load(io.BytesIO(self.read(path)), allow_pickle=False) as panels:
            result = {key: panels[key] for key in panels.files}
        for name, value in result.items():
            if value.dtype.kind not in "bifu" or not np.isfinite(value).all():
                raise ValueError(f"nonfinite or nonreal array: {path}:{name}")
        return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _error(left: np.ndarray, right: np.ndarray) -> float:
    _require(left.shape == right.shape and left.size > 0, "array shape mismatch")
    return float(np.max(np.abs(left - right)))


def validate(
    producer_path: Path,
    reference_directory: Path,
    manifest_path: Path,
    case: str,
    *,
    inputs: Inputs | None = None,
) -> dict[str, Any]:
    """Require complete installed calls and independently accepted matching roots.

    This deliberately consumes the canonical qualification manifest, rather than
    accepting producer-selected tolerances. The source reference may use an older
    binary, but its physical model, input provenance and frozen gates must match.
    """
    inputs = Inputs() if inputs is None else inputs
    manifest = inputs.json(manifest_path)
    specification, gates = manifest["cases"][case], manifest["gates"]
    producer = inputs.json(producer_path)
    reference = inputs.json(reference_directory / "receipt.json")
    _require(reference["accepted"] is True, "independent reference is not accepted")
    _require(reference["stage"] == "accepted", "independent reference is incomplete")
    _require(reference["case"] == case, "reference case mismatch")
    _require(
        reference["oracle_scf_cross_seeding"] is False
        and reference["production_displaced_scfs"] == 0,
        "reference independence contract changed",
    )
    _require(
        producer["native_endpoints_complete"] is True
        and producer["independent_validation_pending"] is True
        and producer["accepted"] is False
        and producer["stage"] == "independent-validation-pending",
        "installed producer has not completed its pending endpoints",
    )
    _require("failure" not in producer, "installed producer recorded a failure")
    _require(
        all(
            producer["oracle_modules_absent"].get(name) is True
            for name in ("pyscf", "dftd3", "cupy")
        ),
        "installed producer did not establish reference-engine isolation",
    )
    manifest_sha = inputs.members[str(manifest_path.resolve())]["sha256"]
    _require(
        producer["fixture_manifest_sha256"] == manifest_sha
        and reference["source_hashes"][
            "tests/data/global_hybrid_ecp/native_curvature_manifest.json"
        ]
        == manifest_sha,
        "qualification manifest identity mismatch",
    )
    for name in ("atoms_bohr", "charge", "multiplicity", "max_eri_primitives"):
        _require(
            producer[name] == reference[name] == specification[name],
            f"physical input mismatch: {name}",
        )
    for name in ("method_identity", "basis_manifest"):
        _require(producer[name] == reference[name], f"physical model mismatch: {name}")
    _require(producer["gates"] == reference["gates"] == gates, "frozen gates changed")
    _require(
        producer["scf_max_iterations"]
        == reference["native_scf_policy"]["max_iterations"]
        == specification["native_scf_max_iterations"],
        "SCF allowance mismatch",
    )
    library = Path(producer["library"])
    _require(
        hashlib.sha256(inputs.read(library)).hexdigest() == producer["library_sha256"],
        "installed native library identity mismatch",
    )
    _require(
        reference["minimum_occupied_root_overlap"]
        >= gates["minimum_occupied_root_overlap"],
        "independent displaced root gate failed",
    )
    for name, value in reference["central_errors"].items():
        _require(
            np.isfinite(value) and value < gates[name], f"reference central {name}"
        )

    prefix = producer_path.with_suffix("")
    central = inputs.arrays(Path(f"{prefix}-central.npz"))
    oracle = inputs.arrays(reference_directory / "independent-state.npz")
    density, occupations = central["density"], central["occupations"]
    _require(
        density.ndim == 3 and density.shape[1] == density.shape[2],
        "installed spin density layout mismatch",
    )
    spins, nao, _ = density.shape
    _require(
        spins == (1 if specification["multiplicity"] == 1 else 2),
        "installed spin channels mismatch",
    )
    _require(occupations.shape == (spins, nao), "installed occupation layout mismatch")
    expected_occupations = oracle["occupations"].reshape(spins, nao)
    _require(
        np.array_equal(occupations, expected_occupations),
        "installed occupations changed",
    )
    reconstructed = np.einsum(
        "sik,sk,sjk->sij", central["coefficients"], occupations, central["coefficients"]
    )
    errors = {
        "density": _error(density, oracle["density"].reshape(density.shape)),
        "density_reconstruction": _error(density, reconstructed),
        "energy": abs(
            float(central["energy"])
            - float(oracle["electronic_energy"] + oracle["d3_energy"])
        ),
        "electron_count": _error(
            np.einsum("sij,ji->s", density, oracle["overlap"]), occupations.sum(axis=1)
        ),
    }
    for name in ("density", "density_reconstruction", "energy", "electron_count"):
        gate = {
            "density_reconstruction": "density",
            "electron_count": "independent_electron_count",
        }.get(name, name)
        _require(
            np.isfinite(errors[name]) and errors[name] < gates[gate],
            f"installed central {name} gate failed",
        )

    dimension = 3 * len(specification["atoms_bohr"])
    matrix = inputs.arrays(Path(f"{prefix}-matrix.npz"))["matrix"]
    _require(
        matrix.shape == (dimension, dimension), "installed Hessian layout mismatch"
    )
    raw = inputs.arrays(reference_directory / "native-matrix.npz")["matrix"]
    errors["native_matrix_replay"] = _error(matrix, raw)
    errors["raw_symmetry"] = _error(matrix, matrix.T)
    translation = matrix @ np.tile(np.eye(3), (dimension // 3, 1))
    errors["translation"] = float(np.max(np.abs(translation)))
    _require(
        errors["native_matrix_replay"] < gates["hessian_hvp"],
        "native matrix replay failed",
    )
    for name in ("raw_symmetry", "translation"):
        _require(errors[name] < gates[name], f"installed {name} gate failed")
    references = []
    for step in manifest["difference_steps_bohr"]:
        target = inputs.arrays(reference_directory / f"independent-matrix-{step}.npz")[
            "matrix"
        ]
        error = _error(matrix, target)
        _require(
            error < gates["raw_hessian"], f"independent raw Hessian failed: {step}"
        )
        references.append(target)
        errors[f"independent_matrix_{step}"] = error
    actions = []
    for index in range(2):
        action = inputs.arrays(Path(f"{prefix}-hvp-{index}.npz"))
        direction, value = action["direction"], action["value"]
        _require(
            direction.shape == value.shape == (dimension // 3, 3), "HVP layout mismatch"
        )
        expected = (np.sin, np.cos)[index](np.arange(dimension) + 1).reshape(
            direction.shape
        ) * 0.1
        _require(
            np.array_equal(direction, expected), "qualification HVP direction changed"
        )
        _require(
            _error(matrix @ direction.ravel(), value.ravel()) < gates["hessian_hvp"],
            "installed Hessian/HVP parity failed",
        )
        for target in references:
            _require(
                _error(target @ direction.ravel(), value.ravel()) < gates["raw_hvp"],
                "independent raw HVP failed",
            )
        actions.append((direction.ravel(), value.ravel()))
    errors["bilinear"] = abs(
        float(actions[0][0] @ actions[1][1] - actions[1][0] @ actions[0][1])
    )
    _require(
        errors["bilinear"] < gates["bilinear"], "installed bilinear symmetry failed"
    )
    for name in ("matrix_diagnostics", "hvp_0_diagnostics", "hvp_1_diagnostics"):
        diagnostic = producer[name]
        _require(
            diagnostic["public_calculator_endpoint"] is True
            and diagnostic["numerical_runtime"] == "native-cpp-cpu"
            and diagnostic["runtime_compilation"] is False
            and diagnostic["displaced_scf_evaluations"] == 0
            and diagnostic["reference_engine_fallbacks"] == 0
            and diagnostic["posthoc_symmetrization"] is False
            and diagnostic["total_method_identity"] == producer["method_identity"]
            and diagnostic["d3_pair_visits"] > 0
            and np.isfinite(diagnostic["response_residual_norm"])
            and diagnostic["response_residual_norm"] < gates["response_residual"],
            f"installed native endpoint diagnostics failed: {name}",
        )
    return {
        "accepted": True,
        "case": case,
        "scope": "retained installed public H/two HVPs against accepted independent references",
        "new_independent_oracles": False,
        "producer_modified": False,
        "ts_qualified": False,
        "scale_qualified": False,
        "errors": errors,
        "gates": gates,
        "producer_seconds": producer["seconds"],
        "producer_maximum_rss_kib": producer["maximum_rss_kib"],
        "referenced_members": inputs.members,
    }


def main() -> int:
    """Write a new, exclusive acceptance or failure receipt; never edit inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    started, inputs = time.perf_counter(), Inputs()
    with arguments.output.open("x", encoding="utf-8") as stream:
        try:
            result = validate(
                arguments.producer,
                arguments.reference_directory,
                arguments.manifest,
                arguments.case,
                inputs=inputs,
            )
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as error:
            result = {
                "accepted": False,
                "failure_type": type(error).__name__,
                "failure": str(error),
                "referenced_members": inputs.members,
            }
        result["validation_seconds"] = time.perf_counter() - started
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
