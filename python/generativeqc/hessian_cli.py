"""Private, versioned compiler-AD worker for the explicit C++ Hessian commands.

Normal native CLI commands never import this module. The caller selects an
interpreter explicitly and pins its already-linked native library; this worker
does not choose a different backend, oracle, geometry, or scientific method.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_PROTOCOL_BYTES = 64 << 20


def _json_value(value: Any) -> Any:
    """Detach immutable diagnostic containers without changing numerical data."""
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"unsupported compiler Hessian diagnostic type: {type(value).__name__}"
    )


def execute_request(request: dict, *, library: Path) -> dict:
    """Run exactly one native SCF plus the existing analytic Hessian/HVP endpoint.

    Units are fixed to Bohr on input and Hartree/Bohr squared on publication.
    All admission, response, memory and quadrature gates remain in their original
    owners. No finite difference or alternate host derivative path is added.
    """
    if request.get("schema") != "generativeqc.compiler-hessian-request/v1":
        raise ValueError("unsupported compiler Hessian request schema")
    operation = request["operation"]
    if operation not in ("hessian", "hvp"):
        raise ValueError("unsupported compiler Hessian operation")
    library = library.resolve(strict=True)
    os.environ["GENERATIVEQC_LIBRARY"] = str(library)

    from . import Calculator, GridSpec, KsOptions, _native
    from .response_solver import GMRESOptions

    native_identity = _native.load_library().generativeqc_get_source_identity().decode()
    if native_identity != request["native_source_identity"]:
        raise ValueError("compiler worker native source identity differs from C++ CLI")
    options = dict(request["calculator"])
    if options["method"] not in ("wb97m-v", "wb97m-v-rks", "wb97m-v-uks"):
        raise ValueError("compiler Hessian CLI requires canonical WB97M-V RKS/UKS")
    options["ks_options"] = KsOptions(grid=GridSpec(**request["grid"]))
    calculator = Calculator(**options)
    controls = {
        **request["controls"],
        "charge": request["charge"],
        "multiplicity": request["multiplicity"],
        "solver_options": GMRESOptions(**request["solver"]),
    }
    try:
        if operation == "hvp":
            result = calculator.hessian_vector_product(
                request["atoms_bohr"], request["direction"], **controls
            )
            panel = {"value": result.value.tolist()}
        else:
            result = calculator.hessian(request["atoms_bohr"], **controls)
            panel = {"matrix": result.matrix.tolist()}
    finally:
        calculator.clear_cache()
    return {
        "schema": "generativeqc.compiler-hessian-result/v1",
        "operation": operation,
        "method": options["method"],
        "backend": options["device"],
        "charge": request["charge"],
        "multiplicity": request["multiplicity"],
        "atom_count": len(request["atoms_bohr"]),
        "units": "hartree/bohr^2",
        "native_source_identity": native_identity,
        "native_library_path": str(library),
        "identity": result.identity,
        "diagnostics": _json_value(result.diagnostics),
        **panel,
    }


def main(argv: list[str] | None = None) -> int:
    """Keep stdout exclusively for one complete bounded JSON publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        encoded = sys.stdin.buffer.read(_PROTOCOL_BYTES + 1)
        if len(encoded) > _PROTOCOL_BYTES:
            raise ValueError(
                "compiler Hessian request exceeds the 64 MiB protocol limit"
            )
        request = json.loads(encoded)
        with contextlib.redirect_stdout(sys.stderr):
            result = execute_request(request, library=arguments.library)
        output = json.dumps(result, allow_nan=False, separators=(",", ":"))
        if len(output.encode("utf-8")) > _PROTOCOL_BYTES:
            raise ValueError(
                "compiler Hessian result exceeds the 64 MiB protocol limit"
            )
        sys.stdout.write(output + "\n")
        return 0
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        ImportError,
        ArithmeticError,
        MemoryError,
        LookupError,
    ) as error:
        print(f"generativeqc compiler Hessian: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
