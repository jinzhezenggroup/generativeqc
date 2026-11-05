"""Serial execution of native pair admission, canonical mapping, and strict retry.

This uses the actual task header with a host-only CUDA shim. It is a scientific
and publication gate, not a substitute for physical CUDA/race/resource tests.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from generativeqc_compiler.integral.lowering.fock_accumulation import (
    emit_direct_fock_accumulation_header,
)
from generativeqc_compiler.integral.weighted_eri_cuda import (
    emit_low_order_weighted_header,
    emit_order4_weighted_header,
    emit_order5_weighted_header,
)
from test_weighted_eri_separable import CENTERS, CLASSES, _libcint

ROOT = Path(__file__).resolve().parents[2]
DIMENSION = 15
MATRIX = DIMENSION**2

SHIM = r"""
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#define __host__
#define __device__
#define __forceinline__ inline
#define __noinline__
using std::isfinite;
using std::min;
using std::max;
template <class Value> Value atomicAdd(Value* address, Value value) {
  const auto old = *address; *address += value; return old;
}
inline unsigned long long clock64() { return 0; }
"""


@pytest.fixture(scope="module")
def native_pair_owner(required_native_cxx: Any, tmp_path_factory: Any) -> Any:
    """Compile the complete real owner without importing the public runtime."""
    folder = tmp_path_factory.mktemp("scalar-pair-owner")
    headers = {
        "cuda_runtime.h": SHIM,
        "weighted_eri.cuh": emit_low_order_weighted_header(),
        "order4_weighted_eri.cuh": emit_order4_weighted_header(),
        "order5_weighted_eri.cuh": emit_order5_weighted_header(),
        "generated_direct_fock_accumulation.cuh": emit_direct_fock_accumulation_header(),
    }
    for name, source in headers.items():
        (folder / name).write_text(source)
    library = required_native_cxx.build_shared(
        [ROOT / "tests/native/direct_scalar_separable_host.cpp"],
        folder / "owner.so",
        compile_args=(
            "-std=c++20",
            "-O1",
            "-Wno-unknown-pragmas",
            f"-I{folder}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
        compile_timeout=300,
    )
    loaded = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integer = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    counts = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
    function = cast("Any", loaded.scalar_pair_probe)
    function.argtypes = (
        ctypes.c_uint,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_uint,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_uint,
        integer,
        array,
        array,
        array,
        array,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_bool,
        ctypes.c_uint,
        array,
        counts,
    )
    function.restype = None
    mask = loaded.scalar_pair_full_mask_probe
    mask.argtypes = (
        ctypes.c_uint,
        ctypes.c_uint,
        np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS"),
        integer,
        integer,
        ctypes.c_bool,
        array,
        ctypes.c_double,
        counts,
    )
    mask.restype = ctypes.c_bool
    function.full_mask = mask
    return function


def _inputs(
    unrestricted: bool, system: int = 0, angular: tuple[int, ...] = (2, 1, 1, 1)
) -> dict[str, Any]:
    generator = np.random.default_rng(8831)
    dimension = sum((degree + 1) * (degree + 2) // 2 for degree in angular)
    matrix = dimension**2
    return {
        "class_index": CLASSES.index(angular),
        "unrestricted": unrestricted,
        "system": system,
        "atoms": np.array([0, 1, 2, 3], dtype=np.int32),
        "positions": CENTERS.copy().ravel(),
        "density": generator.normal(
            size=(system + 1) * (2 if unrestricted else 1) * matrix
        ),
        "coefficients": generator.normal(size=(system + 1) * dimension),
        "bounds": np.ones((system + 1) * matrix),
    }


def _run(
    owner: Any,
    inputs: dict[str, Any],
    *,
    enabled: bool = True,
    separate: bool = True,
    orientation: int = 0,
    same_ket: bool = False,
    same_pair: bool = False,
    seeded: bool = False,
    exchange: float = 0.0,
    mixed: bool = False,
    fault: int = 0,
    coulomb: float = 0.73,
    threshold: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.linspace(0.01, 0.24, 24)
    counts = np.zeros(13, dtype=np.uint64)
    seed = np.cos(np.arange(inputs["density"].size))
    pointer = seed.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if seeded else None
    owner(
        inputs["class_index"],
        inputs["unrestricted"],
        separate,
        enabled,
        orientation,
        same_ket,
        same_pair,
        inputs["system"],
        inputs["atoms"],
        inputs["positions"],
        inputs["density"],
        inputs["coefficients"],
        inputs["bounds"],
        pointer,
        threshold,
        coulomb,
        exchange,
        mixed,
        fault,
        output,
        counts,
    )
    return output, counts


@pytest.mark.parametrize("unrestricted", (False, True))
@pytest.mark.parametrize("separate", (False, True))
@pytest.mark.parametrize("orientation", range(8))
@pytest.mark.parametrize("same_ket", (False, True))
@pytest.mark.parametrize("angular", CLASSES)
def test_native_pair_matches_strict_orbit_with_canonical_symmetries(
    native_pair_owner: Any,
    unrestricted: bool,
    separate: bool,
    orientation: int,
    same_ket: bool,
    angular: tuple[int, ...],
) -> None:
    if same_ket and angular[2] != angular[3]:
        pytest.skip("a repeated ket shell must have the same angular degree")
    inputs = _inputs(unrestricted, system=orientation % 2, angular=angular)
    actual, counts = _run(
        native_pair_owner,
        inputs,
        separate=separate,
        orientation=orientation,
        same_ket=same_ket,
    )
    expected, _ = _run(
        native_pair_owner,
        inputs,
        enabled=False,
        separate=separate,
        orientation=orientation,
        same_ket=same_ket,
    )
    np.testing.assert_allclose(actual, expected, rtol=4e-11, atol=4e-12)
    np.testing.assert_array_equal(counts[:4], [1, 1, 0, 0])
    components = [(degree + 1) * (degree + 2) // 2 for degree in angular]
    bra_terms = components[0] * components[1]
    ket_terms = (
        components[2] * (components[2] + 1) // 2
        if same_ket
        else components[2] * components[3]
    )
    # Equal-degree PPPP pairs may exchange bra/ket during canonicalization.
    np.testing.assert_array_equal(counts[[4, 7, 8]], [0, 1, 0])
    np.testing.assert_array_equal(np.sort(counts[5:7]), sorted((bra_terms, ket_terms)))
    np.testing.assert_array_equal(counts[10:], [bra_terms + ket_terms, 1, 0])
    if separate:
        np.testing.assert_array_equal(actual[12:], np.linspace(0.01, 0.24, 24)[12:])


@pytest.mark.parametrize("unrestricted", (False, True))
@pytest.mark.parametrize("coulomb", (0.73, -0.41))
@pytest.mark.parametrize("angular", CLASSES)
@pytest.mark.parametrize("topology", ("distinct", "same_ket", "same_pair"))
def test_native_pair_force_matches_independent_libcint_and_unique_orbits(
    native_pair_owner: Any,
    unrestricted: bool,
    coulomb: float,
    angular: tuple[int, ...],
    topology: str,
) -> None:
    """Independently assemble signed forces from raw libcint center derivatives."""
    if topology == "same_pair" and angular[:2] != angular[2:]:
        pytest.skip("a repeated shell pair must have equal pair signatures")
    if topology == "same_ket" and angular[2] != angular[3]:
        pytest.skip("a repeated ket shell must have equal degrees")
    slots = (
        (0, 1, 0, 1)
        if topology == "same_pair"
        else (0, 1, 2, 2)
        if topology == "same_ket"
        else (0, 1, 2, 3)
    )
    inputs = _inputs(unrestricted, angular=angular)
    components = [(degree + 1) * (degree + 2) // 2 for degree in angular]
    offsets = np.r_[0, np.cumsum(components)]
    dimension = int(offsets[-1])
    matrix = dimension**2
    _, gradient = _libcint(angular, "regular", slots)
    total = inputs["density"][:matrix].copy()
    if unrestricted:
        total += inputs["density"][matrix:]
    expected = np.linspace(0.01, 0.24, 24)
    seen = set()
    for index, component in enumerate(np.ndindex(*components)):
        first, second, third, fourth = (
            int(offsets[slot]) + local
            for slot, local in zip(slots, component, strict=True)
        )
        orbit = frozenset(
            {
                (first, second, third, fourth),
                (second, first, third, fourth),
                (first, second, fourth, third),
                (second, first, fourth, third),
                (third, fourth, first, second),
                (fourth, third, first, second),
                (third, fourth, second, first),
                (fourth, third, second, first),
            }
        )
        if orbit in seen:
            continue
        seen.add(orbit)
        coefficient = sum(
            0.5
            * coulomb
            * total[left + dimension * right]
            * total[ket_left + dimension * ket_right]
            for left, right, ket_left, ket_right in orbit
        )
        coefficient *= np.prod(inputs["coefficients"][[first, second, third, fourth]])
        for center, atom in enumerate(slots):
            expected[3 * atom : 3 * atom + 3] -= (
                coefficient * gradient[3 * center : 3 * center + 3, index]
            )
    actual, counts = _run(
        native_pair_owner,
        inputs,
        orientation=7,
        coulomb=coulomb,
        same_ket=topology == "same_ket",
        same_pair=topology == "same_pair",
    )
    np.testing.assert_allclose(actual, expected, rtol=4e-11, atol=4e-12)
    np.testing.assert_array_equal(counts[:4], [1, 1, 0, 0])


@pytest.mark.parametrize("atoms", ((0, 0, 1, 2), (0, 1, 1, 2), (0, 0, 0, 0)))
def test_native_pair_repeated_atom_folding(
    native_pair_owner: Any, atoms: tuple[int, ...]
) -> None:
    inputs = _inputs(True)
    inputs["atoms"] = np.array(atoms, dtype=np.int32)
    actual, _ = _run(native_pair_owner, inputs)
    expected, _ = _run(native_pair_owner, inputs, enabled=False)
    np.testing.assert_allclose(actual, expected, rtol=4e-11, atol=4e-12)


@pytest.mark.parametrize(
    "refusal", ("mask", "early_extreme", "late_extreme", "late_inf", "late_bound")
)
@pytest.mark.parametrize("angular", CLASSES)
def test_native_pair_refuses_before_publication_and_retries_once(
    native_pair_owner: Any, refusal: str, angular: tuple[int, ...]
) -> None:
    inputs = _inputs(False, angular=angular)
    components = [(degree + 1) * (degree + 2) // 2 for degree in angular]
    offsets = np.r_[0, np.cumsum(components)]
    dimension = int(offsets[-1])
    fault = 1 if refusal == "late_inf" else 2 if refusal == "late_bound" else 0
    if refusal == "mask":
        inputs["bounds"][dimension * offsets[1]] = 0
    elif refusal == "early_extreme":
        inputs["density"][dimension * offsets[1]] = 1e-200
    elif refusal == "late_extreme":
        inputs["density"][offsets[3] - 1 + dimension * (dimension - 1)] = 1e-200
    actual, counts = _run(native_pair_owner, inputs, fault=fault)
    expected, _ = _run(native_pair_owner, inputs, enabled=False, fault=fault)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        counts[:4], [1, 0, int(refusal == "mask"), int(refusal != "mask")]
    )
    assert counts[8] == 1
    if fault:
        assert counts[7] == counts[9] == 2
    else:
        assert counts[7] == 0
    if refusal == "late_extreme":
        np.testing.assert_array_equal(
            counts[5:7], [components[0] * components[1], components[2] * components[3]]
        )


@pytest.mark.parametrize(
    "guard", ({"seeded": True}, {"exchange": -0.29}, {"mixed": True})
)
@pytest.mark.parametrize("angular", CLASSES)
def test_native_pair_unsupported_inputs_keep_original_consumer(
    native_pair_owner: Any, guard: dict[str, Any], angular: tuple[int, ...]
) -> None:
    inputs = _inputs(False, angular=angular)
    actual, counts = _run(native_pair_owner, inputs, **guard)
    expected, _ = _run(native_pair_owner, inputs, enabled=False, **guard)
    np.testing.assert_array_equal(actual, expected)
    assert not counts[:8].any()


def test_native_pair_zero_density_preserves_additive_forces(
    native_pair_owner: Any,
) -> None:
    inputs = _inputs(True)
    inputs["density"][:] = 0
    actual, counts = _run(native_pair_owner, inputs)
    np.testing.assert_array_equal(actual, np.linspace(0.01, 0.24, 24))
    np.testing.assert_array_equal(counts[:4], [1, 1, 0, 0])
    assert counts[7] == counts[8] == 0


def _mask_domain(
    *, same_shell: bool = False, same_pair: bool = False, orientation: int = 0
) -> dict[str, Any]:
    """Independently enumerate AO pairs, including both triangular domains."""
    shells = [0, 1, 2, 2 if same_shell else 3]
    if orientation & 1:
        shells[0], shells[1] = shells[1], shells[0]
    if orientation & 2:
        shells[2], shells[3] = shells[3], shells[2]
    if orientation & 4:
        shells[:2], shells[2:] = shells[2:], shells[:2]
    offsets = np.array([0, 6, 9, 12, 15], dtype=np.int64)
    first = np.array(shells[::2], dtype=np.int32)
    second = np.array(shells[1::2], dtype=np.int32)
    pairs = []
    for left, right in zip(first, second, strict=True):
        pairs.append(
            [
                row + column * DIMENSION
                for row in range(offsets[left], offsets[left + 1])
                for column in range(offsets[right], offsets[right + 1])
                if left != right or row >= column
            ]
        )
    if same_pair:
        pairs[1] = pairs[0]
    return {
        "offsets": offsets + (orientation % 2) * DIMENSION,
        "first": first,
        "second": second,
        "same_pair": same_pair,
        "system": orientation % 2,
        "pairs": pairs,
        "bounds": np.full((1 + orientation % 2) * MATRIX, np.nan),
    }


def _mask_oracle(domain: dict[str, Any], threshold: float) -> bool:
    """Use explicit products, never the implementation's minimum proof."""
    bounds = domain["bounds"][domain["system"] * MATRIX :]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return not any(
            bounds[first] * bounds[second] < threshold
            for first_index, first in enumerate(domain["pairs"][0])
            for second_index, second in enumerate(domain["pairs"][1])
            if not domain["same_pair"] or first_index >= second_index
        )


def _run_mask(
    owner: Any, domain: dict[str, Any], threshold: float
) -> tuple[bool, np.ndarray]:
    counts = np.zeros(4, dtype=np.uint64)
    complete = owner.full_mask(
        DIMENSION,
        domain["system"],
        domain["offsets"],
        domain["first"],
        domain["second"],
        domain["same_pair"],
        domain["bounds"],
        threshold,
        counts,
    )
    return complete, counts


@pytest.mark.parametrize("orientation", range(8))
@pytest.mark.parametrize("same_shell", (False, True))
@pytest.mark.parametrize("same_pair", (False, True))
def test_native_full_mask_matches_explicit_domain_at_fp64_boundaries(
    native_pair_owner: Any, orientation: int, same_shell: bool, same_pair: bool
) -> None:
    domain = _mask_domain(
        orientation=orientation, same_shell=same_shell, same_pair=same_pair
    )
    bounds = domain["bounds"][domain["system"] * MATRIX :]
    entries = sorted(set(domain["pairs"][0] + domain["pairs"][1]))
    generator = np.random.default_rng(997 + orientation)
    exponent_choices = [-1074, -1022, -700, -300, -1, 0, 100, 500, 1023]
    exponents = [
        *exponent_choices,
        generator.choice(exponent_choices, size=len(entries)),
    ]
    for exponent in exponents:
        with np.errstate(under="ignore"):
            bounds[entries] = np.ldexp(
                generator.uniform(0.5, 1.5, len(entries)), exponent
            )
        with np.errstate(over="ignore", under="ignore"):
            minimum = min(bounds[domain["pairs"][0]]) * min(bounds[domain["pairs"][1]])
            thresholds = (
                -np.inf,
                -0.0,
                np.nextafter(0.0, 1.0),
                minimum,
                np.nextafter(minimum, -np.inf),
                np.nextafter(minimum, np.inf),
                np.inf,
                np.nan,
            )
        for threshold in thresholds:
            actual, counts = _run_mask(native_pair_owner, domain, threshold)
            assert actual == _mask_oracle(domain, threshold)
            np.testing.assert_array_equal(
                counts, [0, sum(map(len, domain["pairs"])), 1, 0]
            )


@pytest.mark.parametrize("same_pair", (False, True))
@pytest.mark.parametrize("same_shell", (False, True))
@pytest.mark.parametrize("invalid", (np.nan, np.inf, -np.inf, -1.0))
def test_native_full_mask_invalid_factors_retain_exact_quartet_fallback(
    native_pair_owner: Any, same_pair: bool, same_shell: bool, invalid: float
) -> None:
    domain = _mask_domain(same_pair=same_pair, same_shell=same_shell)
    bounds = domain["bounds"]
    entries = sorted(set(domain["pairs"][0] + domain["pairs"][1]))
    bounds[entries] = 0.5
    for position in (entries[0], entries[-1]):
        bounds[entries] = 0.5
        bounds[position] = invalid
        for threshold in (-1.0, 0.0, 0.25, np.nextafter(0.25, np.inf), np.inf, np.nan):
            actual, counts = _run_mask(native_pair_owner, domain, threshold)
            assert actual == _mask_oracle(domain, threshold)
            assert counts[0] > 0 and 0 < counts[1] <= sum(map(len, domain["pairs"]))
            np.testing.assert_array_equal(counts[2:], [0, 1])


@pytest.mark.parametrize("relative_threshold", (-1, 0, 1))
def test_native_pair_screening_boundary_preserves_force_publication(
    native_pair_owner: Any, relative_threshold: int
) -> None:
    inputs = _inputs(False)
    inputs["bounds"][:] = 1e-6
    product = np.float64(1e-6) * np.float64(1e-6)
    threshold = (
        np.nextafter(product, -np.inf if relative_threshold < 0 else np.inf)
        if relative_threshold
        else product
    )
    actual, counts = _run(native_pair_owner, inputs, threshold=threshold)
    expected, _ = _run(native_pair_owner, inputs, enabled=False, threshold=threshold)
    np.testing.assert_allclose(actual, expected, rtol=4e-11, atol=4e-12)
    np.testing.assert_array_equal(counts[10:], [27, 1, 0])
    if relative_threshold > 0:
        np.testing.assert_array_equal(counts[:4], [1, 0, 1, 0])
        assert not counts[5:8].any()
        np.testing.assert_array_equal(actual, np.linspace(0.01, 0.24, 24))
    else:
        np.testing.assert_array_equal(counts[:4], [1, 1, 0, 0])


@pytest.mark.parametrize("invalid", (np.nan, np.inf, -np.inf, -1.0))
def test_native_pair_invalid_schwarz_preserves_strict_force_fallback(
    native_pair_owner: Any, invalid: float
) -> None:
    inputs = _inputs(True, system=1)
    inputs["bounds"][MATRIX + DIMENSION * 6] = invalid
    actual, counts = _run(native_pair_owner, inputs)
    expected, _ = _run(native_pair_owner, inputs, enabled=False)
    np.testing.assert_allclose(actual, expected, rtol=4e-11, atol=4e-12)
    assert counts[4] > 0
    np.testing.assert_array_equal(counts[11:], [0, 1])
    if counts[2]:
        np.testing.assert_array_equal(actual, expected)
        assert not counts[5:8].any() and counts[8] == 1
