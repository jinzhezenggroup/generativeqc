"""Build-only native CPU maps from the canonical coupled KS TensorIR programs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.method.ks_cpks import build_ks_cpks_maps
from generativeqc_compiler.tensor.host_affine import emit_affine_host

from tools.generate_df_kernels import write_if_changed


def generate() -> str:
    """Keep method algebra and occupation factors with the shared map builder."""
    programs = {
        f"{prefix}_{name}": program
        for prefix, occupation in (("rks", 2), ("uks", 1))
        for name, program in build_ks_cpks_maps(occupation=occupation).items()
    }
    return emit_affine_host(
        programs,
        namespace="generativeqc_host_ks_maps",
        extents={
            "cpks_ao": "nao",
            "cpks_orbital": "nao",
            "cpks_occupied": "nocc",
            "cpks_virtual": "nvir",
        },
    )


def main() -> None:
    """Generate a self-contained checked CPU header without a runtime import."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(arguments.output, generate())


if __name__ == "__main__":
    main()
