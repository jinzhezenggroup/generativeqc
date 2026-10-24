"""Host contracts for bounded native Becke representation admission."""

from __future__ import annotations

import ctypes as ct
import shutil
import subprocess
from itertools import product
from pathlib import Path

import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@pytest.fixture(scope="module")
def policy(tmp_path_factory: pytest.TempPathFactory) -> ct.CDLL:
    compiler = shutil.which("c++")
    cache = shutil.which("sccache") or shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("C++ compiler and verified compiler cache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("becke-scaled-native-policy")
    source, library = directory / "policy.cpp", directory / "policy.so"
    source.write_text(
        '#include "' + str(root / "src/dft/becke_scaled_policy.hpp") + '"\n'
        'extern "C" int request(const char* setting) {\n'
        "  try { return generativeqc::dft::binary_scaled_becke_requested(setting); }\n"
        "  catch (const std::invalid_argument&) { return -1; }\n"
        "}\n"
        'extern "C" bool eligible(size_t atoms, bool phased, bool centers, bool primitive) {\n'
        "  return generativeqc::dft::binary_scaled_becke_eligible(atoms, phased, centers, primitive);\n"
        "}\n"
        'extern "C" bool automatic(size_t atoms) {\n'
        "  return generativeqc::dft::binary_scaled_becke_automatic(atoms);\n"
        "}\n"
    )
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = ct.CDLL(str(library))
    result.request.argtypes = [ct.c_char_p]
    result.request.restype = ct.c_int
    result.eligible.argtypes = [ct.c_size_t, ct.c_bool, ct.c_bool, ct.c_bool]
    result.eligible.restype = ct.c_bool
    result.automatic.argtypes = [ct.c_size_t]
    result.automatic.restype = ct.c_bool
    return result


@pytest.mark.parametrize("setting,expected", [(None, 1), (b"0", 0), (b"1", 1)])
def test_binary_request_defaults_on_with_explicit_incumbent_override(
    policy: ct.CDLL, setting: bytes | None, expected: int
) -> None:
    assert policy.request(setting) == expected


@pytest.mark.parametrize("setting", [b"", b"on", b"off", b"true", b"2", b"-1", b" 1"])
def test_binary_request_rejects_ambiguous_selectors(
    policy: ct.CDLL, setting: bytes
) -> None:
    assert policy.request(setting) == -1


@pytest.mark.parametrize("atoms", [0, 1, 24, 32, 33, 48, 96, 128, 129, 1 << 40])
def test_binary_admission_cannot_acquire_storage_or_alias_primitive(
    policy: ct.CDLL, atoms: int
) -> None:
    for phased, centers, primitive in product((False, True), repeat=3):
        expected = 0 < atoms <= 128 and phased and centers and not primitive
        assert policy.eligible(atoms, phased, centers, primitive) == expected


@pytest.mark.parametrize("atoms", [0, 1, 48, 95, 96, 97, 128, 129, 1 << 40])
def test_automatic_crossover_keeps_unqualified_small_owners_on_incumbent(
    policy: ct.CDLL, atoms: int
) -> None:
    assert policy.automatic(atoms) == (96 <= atoms <= 128)


def test_scaled_native_metric_delta_keeps_admission_and_counts_actual_calls() -> None:
    from generativeqc._stationary_cuda import _metric_delta

    base = dict.fromkeys(
        (
            "h2d_bytes",
            "d2h_bytes",
            "launches",
            "primitive_records",
            "xc_points",
            "grid_pair_visits",
            "task_descriptors",
            "task_batches",
        ),
        0,
    )
    before = {
        **base,
        "becke_binary_scaled_batches": 3,
        "becke_binary_scaled_points": 17,
        "becke_binary_scaled_selected": 1,
    }
    after = {
        **before,
        "becke_binary_scaled_batches": 5,
        "becke_binary_scaled_points": 41,
    }
    actual = _metric_delta(after, before)
    assert actual["becke_binary_scaled_batches"] == 2
    assert actual["becke_binary_scaled_points"] == 24
    assert actual["becke_binary_scaled_selected"] == 1


def test_scaled_policy_is_an_authenticated_checkout_and_wheel_input() -> None:
    from generativeqc_compiler.method.stationary_cuda import STATIONARY_AOT_ASSETS

    root = Path(__file__).resolve().parents[2]
    assert "src/dft/becke_scaled_policy.hpp" in STATIONARY_AOT_ASSETS
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    asset = "src/dft/becke_scaled_policy.hpp"
    assert metadata["tool"]["scikit-build"]["wheel"]["force-include"][asset] == (
        "generativeqc_compiler/assets/" + asset
    )
    assert (root / asset).is_file()


def test_scaled_resource_bound_does_not_extend_the_stationary_atom_domain() -> None:
    from generativeqc_compiler.common.cuda_target import cuda_target_info
    from generativeqc_compiler.method.stationary_resources import (
        plan_stationary_cuda_resources,
    )

    shape = {
        "atoms": 128,
        "aos": 4,
        "primitives": 4,
        "points": 256,
        "tasks": 1,
        "spins": 1,
        "sources": 8,
        "target": cuda_target_info("sm_120"),
        "budget_bytes": 256 << 20,
        "phased_becke": True,
    }
    assert plan_stationary_cuda_resources(**shape).phased_becke_bytes > 0
    with pytest.raises(ValueError, match="atoms exceeds resource caps"):
        plan_stationary_cuda_resources(**{**shape, "atoms": 129})
