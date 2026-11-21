"""Generate native CUDA B3LYP semilocal E/vxc from the shared MethodIR graph."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.method.spec import (
    ExactExchangePrimitive,
    MethodIR,
    SemilocalXCPrimitive,
    resolve_method,
)
from generativeqc_compiler.xc.semilocal_codegen import (
    emit_polarized_semilocal,
    emit_polarized_semilocal_direction,
)
from generativeqc_compiler.xc.spec import FunctionalSpec


def _semilocal(method: MethodIR) -> FunctionalSpec:
    return next(
        primitive.functional
        for primitive in method.primitives
        if isinstance(primitive, SemilocalXCPrimitive)
    )


def emit_b3lyp_device() -> str:
    """Emit the production-tail B3LYP semilocal point ABI for native CUDA."""

    method = resolve_method("B3LYP", spin="polarized")
    exchange = [
        primitive
        for primitive in method.primitives
        if isinstance(primitive, ExactExchangePrimitive)
    ]
    if len(exchange) != 1 or exchange[0].operator != "full-range":
        raise RuntimeError(
            "B3LYP MethodIR lost its canonical full-range exchange primitive"
        )
    body = emit_polarized_semilocal(
        _semilocal(method),
        value_type="B3lypDeviceValue",
        function_name="b3lyp_device",
        identity_constant="kB3lypDeviceExpressionIdentity",
        production=True,
        function_qualifier="__device__ inline",
        declarations=(
            f'inline constexpr const char* kB3lypDeviceMethodIdentity = "{method.identity}";',
            f"inline constexpr double kB3lypDeviceExactExchange = {float(exchange[0].coefficient).hex()};",
        ),
    )
    return _header(body)


def emit_b3lyp_response(*, function_qualifier: str = "__device__ inline") -> str:
    """Emit the analytic action of the same production point model on CUDA."""
    method = resolve_method("B3LYP", spin="polarized")
    return _header(
        emit_polarized_semilocal_direction(
            _semilocal(method),
            value_type="B3lypDeviceDirection",
            function_name="b3lyp_device_direction",
            function_qualifier=function_qualifier,
            frozen_feature_sets=((0, 2, 3), (1, 3, 4)),
        )
    )


def _header(body: str) -> str:
    return "\n".join(
        [
            "// Generated from the canonical B3LYP MethodIR semilocal graph.",
            "#pragma once",
            "#include <cmath>",
            "namespace generativeqc::dft::generated {",
            body.rstrip("\n"),
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
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path)
    output.add_argument("--response-output", type=Path)
    args = parser.parse_args()
    if args.output:
        write_if_changed(args.output, emit_b3lyp_device())
    else:
        write_if_changed(args.response_output, emit_b3lyp_response())


if __name__ == "__main__":
    main()
