"""Generate statically composed native stationary wrappers and their registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.method.stationary_cuda import (
    AVAILABLE_STATIONARY_AOT_PROFILE_NAMES,
    QUALIFIED_STATIONARY_AOT_PROFILE_NAMES,
)
from generativeqc_compiler.method.stationary_native_cuda import (
    NATIVE_STATIONARY_CUDA_PROFILES,
    emit_native_stationary_profile_cuda,
    emit_native_stationary_registry_cpp,
)
from generativeqc_compiler.xc.automatic_semilocal import automatic_registry_entries

from tools.generate_df_kernels import write_if_changed


def main() -> None:
    """Build-time only; generation neither loads the runtime nor probes a GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--cmake-catalog",
        type=Path,
        help="write default/automatic selections during available-profile discovery",
    )
    parser.add_argument(
        "--cmake-dependencies",
        type=Path,
        help="write configure-time compiler inputs alongside profile discovery",
    )
    parser.add_argument(
        "--profiles", nargs="*", choices=NATIVE_STATIONARY_CUDA_PROFILES
    )
    parser.add_argument(
        "--functional",
        action="append",
        help="Bound automatic discovery to these structural registrations",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--profile", choices=NATIVE_STATIONARY_CUDA_PROFILES)
    selection.add_argument("--list-profiles", action="store_true")
    selection.add_argument("--list-aot-profiles", action="store_true")
    selection.add_argument("--list-available-aot-profiles", action="store_true")
    selection.add_argument("--list-automatic-aot-profiles", action="store_true")
    selection.add_argument("--all-profiles", action="store_true")
    selection.add_argument(
        "--registry", nargs="*", choices=NATIVE_STATIONARY_CUDA_PROFILES
    )
    args = parser.parse_args()
    if args.cmake_catalog is not None and not args.list_available_aot_profiles:
        parser.error("--cmake-catalog requires --list-available-aot-profiles")
    if args.functional and not (args.list_automatic_aot_profiles or args.cmake_catalog):
        parser.error("--functional applies only to automatic profile discovery")
    if args.list_automatic_aot_profiles or args.cmake_catalog:
        selected = (
            {name.upper() for name in args.functional} if args.functional else None
        )
        entries = automatic_registry_entries()
        if selected is not None and selected - {entry.name for entry in entries}:
            parser.error("unknown automatic semilocal functional selection")
        automatic = ";".join(
            f"libxc_{entry.code}_{spin}"
            for entry in entries
            if selected is None or entry.name in selected
            for spin in ("rks", "uks")
        )
        if args.list_automatic_aot_profiles:
            print(automatic)
            return
        # These compiler-owned identifiers contain only [a-z0-9_]. Publish
        # both selections in one host-only discovery process, without lowering
        # any automatic stationary graph or repeating compiler imports.
        write_if_changed(
            args.cmake_catalog,
            f'set(_generativeqc_stationary_default_catalog "{";".join(QUALIFIED_STATIONARY_AOT_PROFILE_NAMES)}")\n'
            f'set(_generativeqc_automatic_force_catalog "{automatic}")\n',
        )
    if args.list_profiles or args.list_aot_profiles or args.list_available_aot_profiles:
        if (
            args.output is not None
            or args.output_dir is not None
            or args.profiles is not None
        ):
            parser.error("profile discovery does not write an output file")
        names = (
            AVAILABLE_STATIONARY_AOT_PROFILE_NAMES
            if args.list_available_aot_profiles
            else QUALIFIED_STATIONARY_AOT_PROFILE_NAMES
            if args.list_aot_profiles
            else NATIVE_STATIONARY_CUDA_PROFILES
        )
        if args.cmake_dependencies is not None:
            from generativeqc_compiler.xc.libxc_bulk import CATALOG_PATH

            from tools.run_codegen import _local_python_dependencies

            # XC component admission parses this catalog during compiler
            # import; changes must revalidate discovery even without .py edits.
            dependencies = {
                *_local_python_dependencies(ROOT),
                Path(__file__).resolve(),
                CATALOG_PATH.resolve(),
            }
            write_if_changed(
                args.cmake_dependencies,
                "".join(f"{path.as_posix()}\n" for path in sorted(dependencies)),
            )
        print(";".join(names))
        return
    if args.cmake_dependencies is not None:
        parser.error("--cmake-dependencies applies only to profile discovery")
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
