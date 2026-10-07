"""Compile the actual capability export and check its fixed-width public ABI."""

from __future__ import annotations

import ctypes
import subprocess
import typing
from pathlib import Path

import pytest
from generativeqc import _native
from generativeqc.initial_guess import supports_automatic_minao

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def capability_object(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: typing.Any
) -> Path:
    folder = tmp_path_factory.mktemp("initial-guess-capabilities")
    output = folder / "capabilities.o"
    required_native_cxx.compile_object(
        ROOT / "src/api/c_api_calculation.cpp",
        output,
        args=(
            "-std=c++20",
            "-O0",
            "-fPIC",
            "-ffunction-sections",
            "-fdata-sections",
            "-DGENERATIVEQC_HAS_CUDA=0",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
    )
    return output


@pytest.mark.parametrize("short_enums", [False, True])
@pytest.mark.parametrize("language", ["c", "c++"])
def test_native_capability_export_has_fixed_width(
    capability_object: Path,
    required_native_cxx: typing.Any,
    tmp_path: Path,
    short_enums: bool,
    language: str,
) -> None:
    source = tmp_path / "client.c"
    source.write_text(r"""
#include <stdint.h>
#include <stdio.h>
#include "generativeqc/generativeqc.h"
#ifdef __cplusplus
#include <type_traits>
static_assert(std::is_same_v<decltype(generativeqc_initial_guess_capabilities_v1()),
                             uint32_t>);
#define ASSERT static_assert
#else
_Static_assert(_Generic(generativeqc_initial_guess_capabilities_v1(), uint32_t: 1,
                       default: 0), "capability query must return uint32_t");
#define ASSERT _Static_assert
#endif
ASSERT(GENERATIVEQC_INITIAL_GUESS_CAPABILITY_HF == 1U, "HF bit drifted");
ASSERT(GENERATIVEQC_INITIAL_GUESS_CAPABILITY_LDA == 2U, "LDA bit drifted");
ASSERT(GENERATIVEQC_INITIAL_GUESS_CAPABILITY_MINAO == 4U, "MINAO bit drifted");
int main(void) {
  if (generativeqc_initial_guess_options_version() != 1) return 1;
  const uint32_t capabilities = generativeqc_initial_guess_capabilities_v1();
  if (capabilities != (GENERATIVEQC_INITIAL_GUESS_CAPABILITY_HF |
                       GENERATIVEQC_INITIAL_GUESS_CAPABILITY_LDA |
                       GENERATIVEQC_INITIAL_GUESS_CAPABILITY_MINAO)) return 2;
  printf("%u", (unsigned) capabilities);
}
""")
    obj = tmp_path / "client.o"
    required_native_cxx.compile_object(
        source,
        obj,
        args=(
            "-x",
            language,
            "-std=c11" if language == "c" else "-std=c++20",
            *(("-fshort-enums",) if short_enums else ()),
            "-I",
            str(ROOT / "include"),
        ),
    )
    binary = tmp_path / "client"
    required_native_cxx.link(
        [obj, capability_object], binary, args=("-Wl,--gc-sections",)
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert int(result.stdout) == (
        _native.INITIAL_GUESS_CAPABILITY_HF
        | _native.INITIAL_GUESS_CAPABILITY_LDA
        | _native.INITIAL_GUESS_CAPABILITY_MINAO
    )


def test_capability_binding_uses_uint32() -> None:
    # A real ctypes function models the unsigned ABI, including future high bits.
    signature = ctypes.CFUNCTYPE(ctypes.c_uint32)

    class Library:
        generativeqc_initial_guess_options_version = signature(lambda: 1)
        generativeqc_initial_guess_capabilities_v1 = signature(
            lambda: (1 << 31) | _native.INITIAL_GUESS_CAPABILITY_MINAO
        )

    library = Library()
    assert supports_automatic_minao(library)
    assert library.generativeqc_initial_guess_capabilities_v1.restype is ctypes.c_uint32
    assert library.generativeqc_initial_guess_capabilities_v1.argtypes == []
