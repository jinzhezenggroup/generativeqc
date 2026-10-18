"""Emit inline compiler-AD ordered-quartet energy/weights/Fock source maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.method.wb97mv_nuclear_jk import (
    PARTS,
    SLOTS,
    quartet_operator,
    quartet_program,
)
from generativeqc_compiler.tensor.cuda_inline import lower_inline_cuda_output

from tools.generate_df_kernels import write_if_changed


def generate() -> str:
    """Specialize tiny quartet maps; no runtime, GPU or reference imports."""
    text = [
        "#pragma once",
        "#include <cmath>",
        "#include <cstddef>",
        "#include <string_view>",
        "namespace generativeqc_wb97mv_nuclear_jk_ad {",
        "enum class Kernel {" + ",".join(f"k_{part}" for part in PARTS) + "};",
        "struct Result { double energy{}, weight{}, delta_weight{}, fock[8]{}, motion[8]{}; };",
        "inline constexpr std::string_view radial[] = {"
        + ",".join(f'"{quartet_operator(part)[0]}"' for part in PARTS)
        + "};",
        "inline constexpr double omega[] = {"
        + ",".join(format(quartet_operator(part)[1], ".17g") for part in PARTS)
        + "};",
    ]
    positions = ((0, 1), (2, 3), (0, 2), (1, 3))
    for spins in (1, 2):
        bindings = {
            "integrals": ("value",),
            "d_integrals": ("motion",),
            "bar_energy": ("1.0",),
        }
        for name, (first, second) in zip(SLOTS, positions, strict=True):
            for prefix, pointer in (("", "density"), ("d_", "delta_density")):
                bindings[prefix + name] = tuple(
                    f"{pointer}[{spin} * nao * nao + ao[{first}] * nao + ao[{second}]]"
                    for spin in range(spins)
                )
        for part in PARTS:
            program = quartet_program(spins, part)
            text.extend(
                [
                    f"__device__ __forceinline__ bool {part}_{spins}(const double* density, const double* delta_density,",
                    " std::size_t nao, const std::size_t* ao, double value, double motion, Result& output) {",
                    " output = {};",
                ]
            )
            for name in program.outputs:
                expression = lower_inline_cuda_output(
                    program, output=name, bindings=bindings
                ).expression
                target = name
                if name.startswith(("fock_", "motion_")):
                    label, slot, spin = name.split("_")
                    target = f"{label}[{2 * int(slot) + int(spin)}]"
                text.append(f" output.{target} = {expression};")
            text.extend(
                [
                    " if (!isfinite(output.energy) || !isfinite(output.weight) || !isfinite(output.delta_weight)) return false;",
                    " for (unsigned index = 0; index < 8; ++index) if (!isfinite(output.fock[index]) || !isfinite(output.motion[index])) return false;",
                    " return true;",
                    "}",
                ]
            )
    text.extend(
        [
            "__device__ __forceinline__ bool evaluate(Kernel kernel, unsigned spins, const double* density,",
            " const double* delta_density, std::size_t nao, const std::size_t* ao, double value, double motion, Result& output) {",
            " if (spins != 1 && spins != 2) return false;",
            " switch (kernel) {",
        ]
    )
    for part in PARTS:
        text.append(
            f" case Kernel::k_{part}: return spins == 1 ? {part}_1(density,delta_density,nao,ao,value,motion,output) : {part}_2(density,delta_density,nao,ao,value,motion,output);"
        )
    return "\n".join([*text, " } return false;", "}", "}", ""])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(args.output, generate())


if __name__ == "__main__":
    main()
