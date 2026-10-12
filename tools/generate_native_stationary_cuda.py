"""Generate statically composed native stationary wrappers and their registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.method.stationary_native_cuda import (
    NATIVE_STATIONARY_CUDA_PROFILES,
    emit_native_stationary_profile_cuda,
    emit_native_stationary_registry_cpp,
)

from tools.generate_df_kernels import write_if_changed


def main() -> None:
    """Build-time only; generation neither loads the runtime nor probes a GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--profile", choices=NATIVE_STATIONARY_CUDA_PROFILES)
    selection.add_argument(
        "--registry", nargs="*", choices=NATIVE_STATIONARY_CUDA_PROFILES
    )
    args = parser.parse_args()
    source = (
        emit_native_stationary_profile_cuda(args.profile)
        if args.profile
        else emit_native_stationary_registry_cpp(tuple(args.registry))
    )
    write_if_changed(args.output, source)


if __name__ == "__main__":
    main()
