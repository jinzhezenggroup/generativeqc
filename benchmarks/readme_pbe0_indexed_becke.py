"""Qualify exact AO maps and generated Becke adjoints on complete PBE0 calls.

Reuse the independent README protocol unchanged. Select producers only for
already admitted force domains, preserving cutoff, tile shape and both budgets.
These controls do not register a production profile or authorize promotion.
"""

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from generativeqc import _force_active_ao
from generativeqc_compiler.common.provenance import file_hash

from benchmarks import readme_omol25
from benchmarks.readme_pbe0 import main as run_endpoint


def main() -> None:
    """Retain cold/warm/moved histories and independent E/F gates per arm."""
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "--force-producer",
        choices=("incumbent", "exact-jets-native-bitmask"),
        default="incumbent",
    )
    parser.add_argument(
        "--becke-primitive",
        choices=("off", "coefficients", "normalized-adjoints"),
        default="off",
    )
    controls, remaining = parser.parse_known_args()
    original = _force_active_ao.resolve_force_active_ao_policy
    original_hashes = readme_omol25.source_hashes

    def source_hashes() -> dict[str, str]:
        """Bind experimental producers/lowerings in addition to the base ledger."""
        root = Path(__file__).resolve().parents[1]
        paths = (
            "benchmarks/readme_pbe0_indexed_becke.py",
            "python/generativeqc/_resident_ao_maps.py",
            "python/generativeqc_compiler/dft/ao_map_plan.py",
            "python/generativeqc_compiler/dft/cuda.py",
            "python/generativeqc_compiler/dft/indexed_layout_native.py",
            "python/generativeqc_compiler/dft/envelope_cuda.py",
            "python/generativeqc_compiler/method/stationary_becke_phased.py",
            "python/generativeqc_compiler/xc/becke_normalized_adjoint.py",
            "python/generativeqc_compiler/xc/grid_phased.py",
            "src/dft/cuda_grid.cu",
            "src/dft/stationary_gradient_cuda.cuh",
        )
        return {**original_hashes(), **{path: file_hash(root / path) for path in paths}}

    def select(
        workload: _force_active_ao.ForceActiveAoWorkload,
    ) -> _force_active_ao.ForceActiveAoDecision:
        decision = original(workload)
        if controls.force_producer == "incumbent" or not decision.selected:
            return decision
        return replace(
            decision,
            producer=controls.force_producer,
            profile_id="qualification-exact-jets-native-bitmask",
            reason="explicit endpoint qualification; incumbent cutoff/budgets retained",
        )

    with (
        patch.object(_force_active_ao, "resolve_force_active_ao_policy", select),
        patch.object(readme_omol25, "source_hashes", source_hashes),
        patch.dict(
            os.environ,
            {"GENERATIVEQC_STATIONARY_BECKE_PRIMITIVE": controls.becke_primitive},
        ),
        patch.object(sys, "argv", [sys.argv[0], *remaining]),
    ):
        run_endpoint()


if __name__ == "__main__":
    main()
