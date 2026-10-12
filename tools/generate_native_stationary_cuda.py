"""Generate statically composed native stationary wrappers and their registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.method.stationary_cuda import (
    QUALIFIED_STATIONARY_AOT_PROFILE_NAMES,
)
from generativeqc_compiler.method.stationary_native_cuda import (
    NATIVE_STATIONARY_CUDA_PROFILES,
    emit_native_stationary_profile_cuda,
    emit_native_stationary_registry_cpp,
)

from tools.generate_df_kernels import write_if_changed


def main() -> None:
    """Build-time only; generation neither loads the runtime nor probes a GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--profiles", nargs="*", choices=NATIVE_STATIONARY_CUDA_PROFILES
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--profile", choices=NATIVE_STATIONARY_CUDA_PROFILES)
    selection.add_argument("--list-profiles", action="store_true")
    selection.add_argument("--list-aot-profiles", action="store_true")
    selection.add_argument("--all-profiles", action="store_true")
    selection.add_argument(
        "--registry", nargs="*", choices=NATIVE_STATIONARY_CUDA_PROFILES
    )
    args = parser.parse_args()
    if args.list_profiles or args.list_aot_profiles:
        if (
            args.output is not None
            or args.output_dir is not None
            or args.profiles is not None
        ):
            parser.error("profile discovery does not write an output file")
        names = (
            QUALIFIED_STATIONARY_AOT_PROFILE_NAMES
            if args.list_aot_profiles
            else NATIVE_STATIONARY_CUDA_PROFILES
        )
        print(";".join(names))
        return
    if args.all_profiles:
        if args.output is not None or args.output_dir is None:
            parser.error("--all-profiles requires --output-dir and forbids --output")
        profiles = (
            tuple(args.profiles)
            if args.profiles is not None
            else NATIVE_STATIONARY_CUDA_PROFILES
        )
        if not profiles or len(profiles) != len(set(profiles)):
            parser.error("--all-profiles requires unique qualified native graphs")
        for profile in profiles:
            write_if_changed(
                args.output_dir / f"native_{profile}.cu",
                emit_native_stationary_profile_cuda(profile),
            )
        return
    if args.output_dir is not None or args.profiles is not None:
        parser.error("--output-dir/--profiles apply only to --all-profiles")
    if args.output is None:
        parser.error("generation requires --output")
    source = (
        emit_native_stationary_profile_cuda(args.profile)
        if args.profile
        else emit_native_stationary_registry_cpp(tuple(args.registry))
    )
    write_if_changed(args.output, source)


if __name__ == "__main__":
    main()
