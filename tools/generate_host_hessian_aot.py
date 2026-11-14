"""Generate explicitly selected native CPU Hessian leaves at build time only."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.method.host_hessian_aot import (
    emit_registry,
    parse_inventory,
    programs,
)

from tools.generate_df_kernels import write_if_changed


def main() -> None:
    """Generate pure C++ sources without importing a runtime or probing devices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true")
    selection.add_argument("--unit")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    inventory = parse_inventory(json.loads(args.inventory.read_text()))
    selected = programs(inventory)
    names = tuple(program.name + ".cpp" for program in selected) + ("registry.cpp",)
    if args.list:
        if args.output is not None or args.output_dir is not None:
            parser.error("--list does not accept output paths")
        print(";".join(names))
    elif args.all:
        if args.output_dir is None or args.output is not None:
            parser.error("--all requires --output-dir only")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for program in selected:
            write_if_changed(args.output_dir / (program.name + ".cpp"), program.emit())
        write_if_changed(args.output_dir / "registry.cpp", emit_registry(inventory))
    else:
        if args.output is None or args.output_dir is not None or args.unit not in names:
            parser.error("--unit requires a selected source name and --output only")
        source = (
            emit_registry(inventory)
            if args.unit == "registry.cpp"
            else next(
                program.emit()
                for program in selected
                if program.name + ".cpp" == args.unit
            )
        )
        write_if_changed(args.output, source)


if __name__ == "__main__":
    main()
