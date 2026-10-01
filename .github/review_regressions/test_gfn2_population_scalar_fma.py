"""Compare the emitted CUDA population scalar against its prior native FMA."""

from __future__ import annotations

import ctypes
import itertools
import math
import random
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("compiler_name", ["g++", "clang++"])
def test_generated_population_preserves_the_original_accumulator(
    tmp_path: Path, compiler_name: str
) -> None:
    compiler = shutil.which(compiler_name)
    if compiler is None:
        pytest.skip(f"requires {compiler_name}")
    header = tmp_path / "generated.cuh"
    subprocess.run(
        [sys.executable, str(ROOT / "tools/generate_gfn2_electronic_cuda.py"),
         "--output", str(header)], check=True, capture_output=True, text=True, timeout=60,
    )
    source = tmp_path / "fma.cpp"
    source.write_text(r'''
#define __device__
#include "generated.cuh"
extern "C" bool actual(double density, double integral, double accumulator, double* result) {
  return generativeqc::xtb::generated::gfn2_population_update_cuda_tensor(
      density,integral,accumulator,*result);
}
extern "C" double original(double density,double integral,double accumulator) {
  return std::fma(-density,integral,accumulator);
}
''')
    library = tmp_path / "fma.so"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
         "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    lib = ctypes.CDLL(str(library))
    lib.actual.argtypes = [ctypes.c_double] * 3 + [ctypes.POINTER(ctypes.c_double)]
    lib.actual.restype = ctypes.c_bool
    lib.original.argtypes = [ctypes.c_double] * 3
    lib.original.restype = ctypes.c_double
    rng = random.Random(1657)
    edge = (0.0, -0.0, 1.0, -1.0, 1e-300, 1e300, math.inf, -math.inf, math.nan)
    cases = list(itertools.product(edge, repeat=3))
    cases.extend(tuple(struct.unpack("d", struct.pack("Q", rng.getrandbits(64)))[0]
                       for _ in range(3)) for _ in range(10000))
    for case in cases:
        result = ctypes.c_double(123.0)
        expected = lib.original(*case)
        accepted = lib.actual(*case, ctypes.byref(result))
        assert accepted == (all(math.isfinite(x) for x in case) and math.isfinite(expected))
        if accepted:
            assert struct.pack("d", result.value) == struct.pack("d", expected), case
        else:
            assert result.value == 123.0
