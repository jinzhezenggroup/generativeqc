"""Generate compiler-owned Direct-HF pair/Hermite CUDA support."""

import argparse
import sys as _compiler_sys
from pathlib import Path

_compiler_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from generativeqc_compiler.integral.direct_pair_support_cuda import (
    emit_direct_pair_support_headers,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for name, source in emit_direct_pair_support_headers().items():
        output = args.output_directory / name
        if not output.exists() or output.read_text(encoding="utf-8") != source:
            output.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
