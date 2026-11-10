"""Record complete installed CPU D3(BJ) curvature endpoints, without an oracle.

Run from outside the checkout with PYTHONPATH unset, a wheel installed in the
selected interpreter and a finite outer timeout. This receipt measures two
independent public calls (each performs its own SCF); it is not a TS, Ni, ECP,
default-grid or universal performance qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import generativeqc
import numpy as np
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.method import DispersionCorrectionPrimitive, resolve_method
from generativeqc_compiler.method._generated_parameters import d3_parameters
from generativeqc_compiler.method.dispersion import D3Spec


def digest(path: Path) -> str:
    """Identify the actual installed Python/native bytes and input manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Publish a receipt only after both complete raw endpoints pass invariants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    installed = Path(generativeqc.__file__).resolve()
    if installed.is_relative_to(checkout):
        raise RuntimeError(
            "qualification requires an installed package, not PYTHONPATH"
        )
    manifest = json.loads(args.manifest.read_text())
    atoms = manifest["atoms_bohr"]
    gates = manifest["gates"]
    direction = np.random.default_rng(2845).normal(size=(len(atoms), 3)) * 0.03
    started = time.perf_counter()
    electronic = resolve_method("B3LYP")
    method = replace(
        electronic,
        identifier="B3LYP-D3(BJ)",
        primitives=(
            *electronic.primitives,
            DispersionCorrectionPrimitive(D3Spec(**d3_parameters("B3LYP-D3(BJ)"))),
        ),
    )
    calculator = Calculator(
        method=method,
        device="cpu",
        basis_representation="spherical",
        ks_options=KsOptions(grid=GridSpec(**manifest["grid"])),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
    )
    try:
        options = {
            "cache": args.cache,
            "solver_options": GMRESOptions(atol=1e-12, rtol=1e-11),
        }
        matrix = calculator.hessian(atoms, **options)
        action = calculator.hessian_vector_product(atoms, direction, **options)
        library = Path(calculator._library._name).resolve()
        parity = float(
            np.max(
                abs(matrix.matrix @ direction.reshape(-1) - action.value.reshape(-1))
            )
        )
        symmetry = float(np.max(abs(matrix.matrix - matrix.matrix.T)))
        translation = float(
            np.max(abs(matrix.matrix @ np.tile(np.eye(3), (len(atoms), 1))))
        )
        if max(parity, symmetry, translation) > gates["raw_invariants_abs"]:
            raise ArithmeticError("installed raw Hessian/HVP invariants failed")
        receipt = {
            "schema": "generativeqc.installed-d3-second-order/v1",
            "method_manifest_identity": method.manifest_identity,
            "manifest_sha256": digest(args.manifest),
            "installed_package": str(installed),
            "installed_library": str(library),
            "installed_library_sha256": digest(library),
            "installed_python_sha256": {
                name: digest(installed.parent / name)
                for name in (
                    "calculator.py",
                    "batch.py",
                    "composite_rks_hessian.py",
                    "d3_hessian.py",
                )
            },
            "runtime_library_search_path": os.environ.get("LD_LIBRARY_PATH", ""),
            "oracle_modules_loaded": sorted(
                name for name in ("pyscf", "dftd3", "geometric") if name in sys.modules
            ),
            "matrix_identity": matrix.identity,
            "hvp_identity": action.identity,
            "matrix_diagnostics": matrix.diagnostics,
            "hvp_diagnostics": action.diagnostics,
            "h_at_v_max_abs": parity,
            "raw_symmetry_max_abs": symmetry,
            "translation_max_abs": translation,
            "public_scf_evaluations": 2,
            "scope": manifest["scope"],
        }
        if receipt["oracle_modules_loaded"]:
            raise RuntimeError(
                "installed production endpoint imported a reference provider"
            )
    finally:
        calculator.clear_cache()
    receipt.update(
        {
            "preparation_through_cleanup_seconds": time.perf_counter() - started,
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "maximum_child_rss_kib": resource.getrusage(
                resource.RUSAGE_CHILDREN
            ).ru_maxrss,
            "memory_observer": "getrusage high-water marks, not simultaneous process-tree sum",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
