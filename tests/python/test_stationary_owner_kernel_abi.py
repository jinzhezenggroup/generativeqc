"""Compile the emitted owner expression and compare geometry ABI signatures."""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPILER = ROOT / "python/generativeqc_compiler/method/stationary_cuda.py"
HEADER = ROOT / "src/dft/stationary_gradient_cuda.cuh"


class StationaryOwnerKernelTests(unittest.TestCase):
    def test_geometry_kernel_declaration_matches_definition(self) -> None:
        prototype = re.search(
            r"__global__ void geometry_kernel\((.*?)\)\s*;",
            HEADER.read_text(),
            re.DOTALL,
        )
        definition = re.search(
            r"__global__ void geometry_kernel\((.*?)\)\s*\{",
            COMPILER.read_text(),
            re.DOTALL,
        )
        self.assertIsNotNone(prototype)
        self.assertIsNotNone(definition)

        def arguments(match: re.Match[str] | None) -> list[str]:
            assert match is not None
            return [" ".join(item.split()) for item in match.group(1).split(",")]

        self.assertEqual(arguments(prototype), arguments(definition))
        self.assertEqual(len(arguments(definition)), 18)

    def test_cooperative_kernel_declaration_matches_definition(self) -> None:
        signatures = []
        for source, ending in ((HEADER.read_text(), ";"), (COMPILER.read_text(), "{")):
            match = re.search(
                r"__global__ void geometry_cooperative_kernel\((.*?)\)\s*"
                + re.escape(ending),
                source,
                re.DOTALL,
            )
            self.assertIsNotNone(match)
            assert match is not None
            signatures.append(
                [" ".join(item.split()) for item in match.group(1).split(",")]
            )
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(len(signatures[0]), 18)

    def test_emitted_explicit_and_implicit_owner_expression(self) -> None:
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("host C++ compiler is required for the owner probe")
        match = re.search(
            r"const int64_t owner\s*=.*?;", COMPILER.read_text(), re.DOTALL
        )
        self.assertIsNotNone(match)
        assert match is not None
        program = r"""
#include <cstddef>
#include <cstdint>
int64_t select_owner(const int64_t* owners, size_t owner_offset,
                     size_t points_per_atom, size_t p) {
  OWNER_STATEMENT
  return owner;
}
int main() {
  const int64_t explicit_owners[]{2, 0, 1};
  for (size_t p = 0; p != 3; ++p) {
    if (select_owner(explicit_owners, 17, 4, p) != explicit_owners[p]) return 1;
  }
  for (size_t offset = 0; offset != 12; ++offset) {
    for (size_t p = 0; p != 7; ++p) {
      if (select_owner(nullptr, offset, 4, p) != int64_t((offset + p) / 4))
        return 2;
    }
  }
  if (select_owner(nullptr, 0, 0, 0) != -1) return 3;
  return 0;
}
""".replace("OWNER_STATEMENT", match.group(0))
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "owners.cpp"
            binary = Path(temporary) / "owners"
            source.write_text(program)
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror=uninitialized",
                    str(source),
                    "-o",
                    str(binary),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            subprocess.run([str(binary)], check=True, timeout=10)
