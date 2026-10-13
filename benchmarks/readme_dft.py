"""Matched cold/warm DFT energy-plus-force endpoints for the README.

Use the shared moving-grid, independent-reference and fixed-density protocol.
r2SCAN-3c includes its method-defined basis and both geometry corrections.
"""

from __future__ import annotations

import argparse

from benchmarks.readme_omol25 import EndpointSpec
from benchmarks.readme_omol25 import main as run_endpoint

METHODS = ("hf", "pbe", "pbe0", "b3lyp", "r2scan", "r2scan-3c", "m06-2x", "wb97m-v")
SCHEMA = "generativeqc.readme-dft.v1"


def specification(method: str) -> EndpointSpec:
    """Use full-density reference potentials consistently across functionals."""
    if method not in METHODS:
        raise ValueError(f"unknown README functional: {method}")
    basis = "def2-mTZVPP" if method == "r2scan-3c" else "def2-SVP"
    return EndpointSpec(method, basis, SCHEMA, reference_full_fock=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--method", choices=METHODS, required=True)
    args, remaining = parser.parse_known_args()
    run_endpoint(specification(args.method), remaining)


if __name__ == "__main__":
    main()
