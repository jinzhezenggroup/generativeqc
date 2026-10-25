"""Replay the retained #2276 empty-beta inputs without a native library or GPU.

The current generated CPU point and pinned Libxc see identical rounded inputs.
Historical CUDA secants remain separately attributed. New secants use linear AO
moments added to the captured features, so they are an independent host reduction,
not a rerun of the historical native AO contraction or a complete KS endpoint.
No extrapolated short stencil is asserted to be the empty-spin derivative.
"""

from __future__ import annotations

import argparse
import ctypes
import gzip
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from generativeqc_compiler.common.compiler_cache import run_cached_compiler
from generativeqc_compiler.common.compiler_process import run_compiler
from generativeqc_compiler.common.provenance import file_hash
from generativeqc_compiler.xc.automatic_point import emit_automatic_point
from generativeqc_compiler.xc.libxc_work import automatic_work_policy

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/xc/physical-kinetic-predicate/empty-beta-h2plus.json.gz"


def compile_point(
    directory: Path,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, str]]:
    """Compile only the canonical CPU point, through a verified compiler cache."""
    point = emit_automatic_point("MGGA_C_TPSS", backend="cpu")
    source = directory / "point.cpp"
    source.write_text(
        '#include <cmath>\n#include <stdexcept>\n#include "dft/xc.hpp"\n'
        "using generativeqc::dft::SemilocalPointValue;\n"
        + point.source
        + f"""
extern "C" int evaluate(const double* input, double* output, std::size_t count) {{
  try {{
    for (std::size_t p=0; p<count; ++p) {{
      const auto* x=input+10*p;
      double rho[2]{{x[0],x[1]}};
      double gradient[2][3]{{{{x[2],x[3],x[4]}},{{x[5],x[6],x[7]}}}};
      double tau[2]{{x[8],x[9]}};
      const auto v={point.function_name}(rho,gradient,tau);
      auto* y=output+11*p;
      y[0]=v.energy;
      for(unsigned s=0;s<2;++s) {{
        y[1+s]=v.rho[s]; y[9+s]=v.kinetic[s];
        for(unsigned a=0;a<3;++a)y[3+3*s+a]=v.gradient[s][a];
      }}
    }}
    return 0;
  }} catch (...) {{ return 1; }}
}}
"""
    )
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("a C++ compiler is required for the canonical point replay")
    obj, library = directory / "point.o", directory / "point.so"
    commands = (
        (
            run_cached_compiler,
            [
                compiler,
                "-std=c++20",
                "-O2",
                "-fPIC",
                "-I",
                str(ROOT / "src"),
                "-I",
                str(ROOT / "include"),
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            180,
        ),
        (run_compiler, [compiler, "-shared", str(obj), "-o", str(library)], 60),
    )
    for index, (runner, command, timeout) in enumerate(commands):
        result = runner(command, timeout, label="TPSS boundary replay")
        (directory / f"compile-{index}.log").write_text(result.stdout + result.stderr)
        if result.returncode or result.timed_out:
            raise RuntimeError(result.stderr)
    native = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_double)
    native.evaluate.argtypes = [pointer, pointer, ctypes.c_size_t]
    native.evaluate.restype = ctypes.c_int

    def evaluate(features: np.ndarray) -> np.ndarray:
        packed = np.ascontiguousarray(
            np.column_stack(
                (
                    features[:, :, 0],
                    features[:, :, 1:4].reshape(-1, 6),
                    features[:, :, 4],
                )
            )
        )
        output = np.empty((len(features), 11))
        status = native.evaluate(
            packed.ctypes.data_as(pointer),
            output.ctypes.data_as(pointer),
            len(features),
        )
        if status or not np.isfinite(output).all():
            raise RuntimeError("canonical point replay rejected the captured input")
        return output

    return evaluate, {
        "expression_identity": point.expression_identity,
        "source_sha256": file_hash(source),
        "library_sha256": file_hash(library),
    }


def work_counts(features: np.ndarray) -> dict[str, int]:
    """Count the same floor/ferromagnetic branches as the historical diagnostic."""
    policy = automatic_work_policy("MGGA_C_TPSS")
    rho = np.maximum(features[:, :, 0], policy.density_threshold)
    zeta = (rho[:, 0] - rho[:, 1]) / rho.sum(axis=1)
    return {
        "beta_raw_rho_at_or_below_floor": int(
            np.sum(features[:, 1, 0] <= policy.density_threshold)
        ),
        "beta_tau_at_or_below_floor": int(
            np.sum(features[:, 1, 4] <= policy.tau_floor)
        ),
        "ferromagnetic_C0_points": int(np.sum(1 - np.abs(zeta) <= 1e-12)),
    }


def replay(fixture: Path, directory: Path) -> dict:
    """Retain all beta-birth stencils and the separately defined sub-floor control."""
    import pyscf
    from pyscf.dft import libxc

    if (pyscf.__version__, libxc.__version__) != ("2.14.0", "7.0.0"):
        raise RuntimeError(
            "this retained diagnostic requires PySCF 2.14.0 / Libxc 7.0.0"
        )
    data = json.loads(gzip.decompress(fixture.read_bytes()))
    count, nao = data["point_count"], data["nao"]
    ao = np.asarray(data["ao"]).reshape(4, count, nao)
    base = np.asarray(data["features"]).reshape(2, 5, count).transpose(2, 0, 1)
    weights = np.asarray(data["weights"])
    orbitals = np.asarray(data["orbital_coefficients"])[1]
    evaluate, identity = compile_point(directory)

    def reference(features: np.ndarray) -> np.ndarray:
        raw = np.zeros((2, 6, len(features)))
        raw[:, :4, :] = features[:, :, :4].transpose(1, 2, 0)
        raw[:, 5, :] = features[:, :, 4].T
        return libxc.eval_xc("MGGA_C_TPSS", raw, spin=1, deriv=0)[0] * features[
            :, :, 0
        ].sum(axis=1)

    baseline, oracle_baseline = evaluate(base), reference(base)
    rows = []
    vectors = (
        orbitals[:, 0],
        orbitals[:, 1],
        (orbitals[:, 0] + orbitals[:, 1]) / np.sqrt(2),
    )
    for label, vector in zip(
        ("orbital-0-added", "orbital-1-added", "mixed-added"), vectors, strict=True
    ):
        jet = ao @ vector
        increment = np.column_stack(
            (
                jet[0] ** 2,
                (2 * jet[0] * jet[1:4]).T,
                0.5 * np.sum(jet[1:4] ** 2, axis=0),
            )
        )
        # The exported kinetic coefficient is half-vtau, while features store tau.
        response = baseline[:, 2] * increment[:, 0]
        response += np.sum(baseline[:, 6:9] * increment[:, 1:4], axis=1)
        response += 2 * baseline[:, 10] * increment[:, 4]
        stencils = []
        for step in (1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12, 1e-14, 1e-16):
            shifted = base.copy()
            shifted[:, 1] += step * increment
            stencils.append(
                {
                    "step": step,
                    "native_energy_secant": float(
                        weights @ (evaluate(shifted)[:, 0] - baseline[:, 0]) / step
                    ),
                    "libxc_energy_secant": float(
                        weights @ (reference(shifted) - oracle_baseline) / step
                    ),
                    "work_counts": work_counts(shifted),
                }
            )
        rows.append(
            {
                "direction": label,
                "raw_work_contraction": float(weights @ response),
                "stencils": stencils,
                "physical_birth_qualified": False,
            }
        )

    # This legal density-only beta birth stays strictly below the work floor.
    # All gradient/tau work inputs are fixed, and only the original-density
    # energy prefactor changes. Its exact composite derivative is E(0)/rho_a;
    # interior vrho_beta is a different mathematical quantity by construction.
    control = np.zeros((1, 2, 5))
    control[0, 0, 0] = 1e-10
    origin = evaluate(control)[0]
    step = automatic_work_policy("MGGA_C_TPSS").density_threshold / 8
    control[0, 1, 0] = step
    shifted = evaluate(control)[0]
    composite = origin[0] / control[0, 0, 0]
    secant = (shifted[0] - origin[0]) / step
    if abs(secant - composite) > 1e-10:
        raise RuntimeError(
            "density-only subfloor control failed its analytic energy gate"
        )
    return {
        "schema": "generativeqc.tpss-empty-spin-host-replay.v1",
        "claim": "point/AO diagnostic; no physical-birth or complete endpoint clearance",
        "fixture_sha256": file_hash(fixture),
        "current_point": identity,
        "historical_source_tree": data["source_tree"],
        "origin_work_counts": work_counts(base),
        "rows": rows,
        "historical_rows": data["historical_rows"],
        "subfloor_density_only_control": {
            "rho_alpha": float(control[0, 0, 0]),
            "step": step,
            "composite_derivative": float(composite),
            "energy_secant": float(secant),
            "raw_work_vrho_beta": float(origin[2]),
            "secant_error": float(abs(secant - composite)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    result = replay(args.fixture, directory)
    (directory / "replay.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result["subfloor_density_only_control"], indent=2))


if __name__ == "__main__":
    main()
