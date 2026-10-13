"""Compile the actual capability export and check its fixed-width public ABI."""

from __future__ import annotations

import ctypes
import subprocess
import typing
from pathlib import Path

import pytest
from generativeqc import _native
from generativeqc.initial_guess import (
    supports_automatic_huckel,
    supports_automatic_minao,
)

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
ASSERT(GENERATIVEQC_INITIAL_GUESS_CAPABILITY_AUTO_CPU == 8U, "AUTO_CPU bit drifted");
ASSERT(GENERATIVEQC_INITIAL_GUESS_AUTO_CPU == 4, "published AUTO_CPU kind drifted");
ASSERT(GENERATIVEQC_INITIAL_GUESS_HUCKEL == 5, "HUCKEL kind drifted");
ASSERT(GENERATIVEQC_INITIAL_GUESS_AUTO_HUCKEL_CPU == 6, "AUTO_HUCKEL_CPU kind drifted");
ASSERT(GENERATIVEQC_INITIAL_GUESS_CAPABILITY_HUCKEL == 16U, "HUCKEL bit drifted");
ASSERT(GENERATIVEQC_INITIAL_GUESS_CAPABILITY_HUCKEL_CUDA_RHF == 32U, "CUDA RHF bit drifted");
ASSERT(GENERATIVEQC_INITIAL_GUESS_CAPABILITY_HUCKEL_DF == 64U, "DF bit drifted");
ASSERT(GENERATIVEQC_INITIAL_GUESS_CAPABILITY_AUTO_HUCKEL_CPU == 128U, "AUTO_HUCKEL bit drifted");
int main(void) {
  if (generativeqc_initial_guess_options_version() != 1) return 1;
  const uint32_t capabilities = generativeqc_initial_guess_capabilities_v1();
  if (capabilities != (GENERATIVEQC_INITIAL_GUESS_CAPABILITY_HF |
                       GENERATIVEQC_INITIAL_GUESS_CAPABILITY_LDA |
                       GENERATIVEQC_INITIAL_GUESS_CAPABILITY_MINAO |
                       GENERATIVEQC_INITIAL_GUESS_CAPABILITY_AUTO_CPU |
                       GENERATIVEQC_INITIAL_GUESS_CAPABILITY_HUCKEL |
                       GENERATIVEQC_INITIAL_GUESS_CAPABILITY_HUCKEL_CUDA_RHF |
                       GENERATIVEQC_INITIAL_GUESS_CAPABILITY_HUCKEL_DF |
                       GENERATIVEQC_INITIAL_GUESS_CAPABILITY_AUTO_HUCKEL_CPU)) return 2;
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
        | _native.INITIAL_GUESS_CAPABILITY_AUTO_CPU
        | _native.INITIAL_GUESS_CAPABILITY_HUCKEL
        | _native.INITIAL_GUESS_CAPABILITY_HUCKEL_CUDA_RHF
        | _native.INITIAL_GUESS_CAPABILITY_HUCKEL_DF
        | _native.INITIAL_GUESS_CAPABILITY_AUTO_HUCKEL_CPU
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


@pytest.mark.parametrize("bits", [0, 8, 16, 32, 48, 64, 80, 96, 112, (1 << 31) | 112])
@pytest.mark.parametrize(
    "cuda_rhf,density_fitting",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_huckel_negotiates_target_domain_bits(
    bits: int, cuda_rhf: bool, density_fitting: bool
) -> None:
    signature = ctypes.CFUNCTYPE(ctypes.c_uint32)

    class Library:
        generativeqc_initial_guess_options_version = signature(lambda: 1)
        generativeqc_initial_guess_capabilities_v1 = signature(lambda: bits)

    required = 16 | (32 if cuda_rhf else 0) | (64 if density_fitting else 0)
    assert supports_automatic_huckel(
        Library(), cuda_rhf=cuda_rhf, density_fitting=density_fitting
    ) is (bits & required == required)


def test_huckel_missing_schema_and_incompatible_schema_keep_legacy_default() -> None:
    from types import SimpleNamespace

    assert not supports_automatic_huckel(SimpleNamespace())
    signature = ctypes.CFUNCTYPE(ctypes.c_uint32)
    assert not supports_automatic_huckel(
        SimpleNamespace(
            generativeqc_initial_guess_options_version=signature(lambda: 2),
            generativeqc_initial_guess_capabilities_v1=signature(lambda: 56),
        )
    )
