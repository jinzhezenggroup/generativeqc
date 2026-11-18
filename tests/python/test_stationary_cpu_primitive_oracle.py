"""Packaged stationary CPU ERI gradients against independent libcint primitives."""

from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.integral import first_derivative_schedule as schedule
from generativeqc_compiler.integral.first_derivative_native import (
    emit_first_derivative_cpu,
)
from generativeqc_compiler.integral.shell_spec import cartesian_components
from generativeqc_compiler.integral.weighted_eri_native import (
    emit_weighted_eri_primitive_header,
)
from test_factorized_second import reference

POINTER = ct.POINTER(ct.c_double)
REQUESTS = tuple(
    (
        ordinal // schedule.REQUESTS_PER_UNIT,
        ordinal % schedule.REQUESTS_PER_UNIT,
        request[1],
    )
    for ordinal, request in enumerate(
        schedule.derivative_requests(schedule.CPU_AOT_COMPONENTS)
    )
    if request[0] == "four_center_eri"
)
POSITIONS = np.array(
    [
        [0.13, -0.31, 0.24],
        [-0.43, 0.27, 0.51],
        [0.68, -0.14, -0.22],
        [-0.21, 0.48, -0.63],
    ]
)


@pytest.fixture(scope="module")
def packaged() -> Any:
    """Use the supplied real CPU AOT library, never a runtime reference fallback."""
    path = os.environ.get("GENERATIVEQC_LIBRARY")
    if not path:
        pytest.skip("native GenerativeQC library not supplied")
    library = ct.CDLL(str(Path(path)))
    functions = []
    for shard in range(schedule.CPU_AOT_SHARDS):
        function = getattr(library, schedule.cpu_aot_symbol(shard))
        function.argtypes = [ct.c_uint, POINTER, ct.c_size_t, POINTER]
        functions.append(function)
    return functions


@pytest.mark.parametrize("shard,kind,labels", REQUESTS)
@pytest.mark.parametrize("geometry", ["separated", "near_coincident", "diffuse_tight"])
def test_packaged_stationary_eri_primitives_against_libcint(
    packaged: Any, shard: int, kind: int, labels: tuple[str, ...], geometry: str
) -> None:
    """Cover every admitted Cartesian class without a native/reference SCF."""
    positions = POSITIONS.copy()
    exponents = np.array([0.6, 0.8, 1.1, 0.9])
    if geometry == "near_coincident":
        positions[1] = positions[0] + np.array([1e-12, -2e-12, 3e-12])
        positions[3] = positions[2]
    elif geometry == "diffuse_tight":
        exponents = np.array([0.02, 0.7, 15.0, 0.03])
        positions *= 17
    angular = tuple(map(len, labels))
    components = tuple(cartesian_components(order) for order in angular)
    component = np.ravel_multi_index(
        tuple(
            available.index(label)
            for available, label in zip(components, labels, strict=True)
        ),
        tuple(map(len, components)),
    )
    _, gradient = reference(0, angular, positions, exponents=exponents)
    weight = -0.37
    record = np.ascontiguousarray(np.r_[exponents, positions.ravel(), weight])
    output = np.full(14, -781.25)
    assert (
        packaged[shard](
            kind, record.ctypes.data_as(POINTER), 1, output[1:].ctypes.data_as(POINTER)
        )
        == 0
    )
    assert output[0] == output[-1] == -781.25
    actual = output[1:-1]
    np.testing.assert_allclose(
        actual, weight * gradient[:, component], atol=3e-11, rtol=3e-11
    )
    np.testing.assert_allclose(
        actual.reshape(4, 3).sum(axis=0),
        0,
        atol=3e-11 * (1 + np.max(abs(actual))),
        rtol=3e-11,
    )


@pytest.mark.parametrize(
    "failure",
    [
        "zero",
        "negative",
        "nan_coordinate",
        "infinite_weight",
        "count0",
        "count_overflow",
        "null",
        "kind",
    ],
)
def test_packaged_stationary_eri_refusals_are_transactional(
    packaged: Any, failure: str
) -> None:
    """Invalid geometry and batch admission cannot publish derivative prefixes."""
    record = np.ascontiguousarray(np.r_[[0.6, 0.8, 1.1, 0.9], POSITIONS.ravel(), 1.0])
    if failure in ("zero", "negative"):
        record[0] = 0 if failure == "zero" else -0.3
    elif failure == "nan_coordinate":
        record[7] = np.nan
    elif failure == "infinite_weight":
        record[16] = np.inf
    count = 0 if failure == "count0" else 4097 if failure == "count_overflow" else 1
    for shard, kind, _ in REQUESTS:
        output = np.full(14, -781.25)
        assert (
            packaged[shard](
                999 if failure == "kind" else kind,
                POINTER() if failure == "null" else record.ctypes.data_as(POINTER),
                count,
                output[1:].ctypes.data_as(POINTER),
            )
            != 0
        )
        np.testing.assert_array_equal(output, -781.25)


@pytest.mark.parametrize("shard,kind,labels", REQUESTS)
def test_packaged_stationary_eri_signed_batch_and_late_failure(
    packaged: Any, shard: int, kind: int, labels: tuple[str, ...]
) -> None:
    """Signed reductions retain all slots; a later bad record publishes nothing."""
    exponents = np.array([0.6, 0.8, 1.1, 0.9])
    angular = tuple(map(len, labels))
    components = tuple(cartesian_components(order) for order in angular)
    component = np.ravel_multi_index(
        tuple(
            available.index(label)
            for available, label in zip(components, labels, strict=True)
        ),
        tuple(map(len, components)),
    )
    _, gradient = reference(0, angular, POSITIONS, exponents=exponents)
    weights = np.array([1.0, -0.5, 0.0, -0.0, 2**-40, -(2**-40), 0.125])
    records = np.tile(np.r_[exponents, POSITIONS.ravel(), 0.0], (len(weights), 1))
    records[:, 16] = weights
    output = np.full(14, -781.25)
    assert (
        packaged[shard](
            kind,
            records.ctypes.data_as(POINTER),
            len(records),
            output[1:].ctypes.data_as(POINTER),
        )
        == 0
    )
    assert output[0] == output[-1] == -781.25
    np.testing.assert_allclose(
        output[1:-1], weights.sum() * gradient[:, component], atol=3e-11, rtol=3e-11
    )
    records[-1, 0] = -0.3
    output.fill(-781.25)
    assert (
        packaged[shard](
            kind,
            records.ctypes.data_as(POINTER),
            len(records),
            output[1:].ctypes.data_as(POINTER),
        )
        != 0
    )
    np.testing.assert_array_equal(output, -781.25)


@pytest.mark.parametrize("policy", [None, 0, 1, "positive"])
def test_positive_full_policy_is_explicit_even_after_cache_warming(
    policy: object,
) -> None:
    """True and integer one must not share a cache hit before type validation."""
    requests = (("four_center_eri", ("", "", "", "")),)
    emit_first_derivative_cpu(requests, positive_full_moments=True)
    with pytest.raises(TypeError, match="positive_full_moments"):
        emit_first_derivative_cpu(requests, positive_full_moments=policy)
    with pytest.raises(TypeError, match="positive_full_moments"):
        emit_weighted_eri_primitive_header(
            (), backend="cpu", positive_full_moments=policy
        )


def test_positive_full_policy_does_not_admit_unqualified_cuda() -> None:
    with pytest.raises(ValueError, match="qualified CPU"):
        emit_weighted_eri_primitive_header(
            (), backend="cuda", positive_full_moments=True
        )
