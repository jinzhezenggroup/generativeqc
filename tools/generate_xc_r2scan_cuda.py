"""Generate the native CUDA r2SCAN point ABI from shared semilocal lowering."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.xc.r2scan_stationary import emit_r2scan_stationary_point
from generativeqc_compiler.xc.semilocal_codegen import emit_r2scan_program


def emit_r2scan_device(*, variational_value: bool = False) -> str:
    """Preserve the native r2SCAN device ABI while sharing its scientific source."""

    body = emit_r2scan_program(
        value_type="R2scanDeviceValue",
        function_name="r2scan_device",
        identity_constant="kR2scanDeviceExpressionIdentity",
        qualifier="__device__ inline",
    )
    variational = ()
    if variational_value:
        variational = (
            "#define GENERATIVEQC_R2SCAN_VARIATIONAL_VALUE 1",
            emit_r2scan_stationary_point(
                with_action=False, qualifier="__device__ inline"
            ),
        )
    return "\n".join(
        [
            "// Generated from audited MPL-2.0 r2SCAN expressions.",
            "#pragma once",
            "#include <cmath>",
            *(['#include "r2scan_cuda_domain.hpp"'] if variational_value else []),
            "namespace generativeqc::dft::generated {",
            body.rstrip("\n"),
            *variational,
            "}  // namespace generativeqc::dft::generated",
            "",
        ]
    )


def write_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output", type=Path)
    outputs.add_argument("--stationary-output", type=Path)
    parser.add_argument("--variational-value", action="store_true")
    args = parser.parse_args()
    if args.output is not None:
        write_if_changed(
            args.output,
            emit_r2scan_device(variational_value=args.variational_value),
        )
    else:
        write_if_changed(
            args.stationary_output,
            "\n".join(
                (
                    "#pragma once",
                    "#include <cmath>",
                    "namespace generativeqc::dft::generated {",
                    *(
                        emit_r2scan_stationary_point(
                            with_action=action, qualifier="__device__ inline"
                        )
                        for action in (False, True)
                    ),
                    "}",
                    "",
                )
            ),
        )


if __name__ == "__main__":
    main()
