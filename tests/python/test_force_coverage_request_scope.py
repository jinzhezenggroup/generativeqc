"""Execute native force policies without relying on profile labels or GPU discovery."""

import ctypes
import itertools
from pathlib import Path

import pytest
from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def force_policy_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> ctypes.CDLL:
    root = tmp_path_factory.mktemp("force-coverage")
    source = root / "probe.cpp"
    source.write_text(
        '#include "scf/cuda/force_queue_policy.hpp"\n'
        'extern "C" bool reject(bool compute, bool required, bool bounded, '
        "bool diagnostic, unsigned long long missing) {\n"
        "return generativeqc::scf::cuda_policy::bounded_force_coverage_rejected(\n"
        "    compute, required, bounded, diagnostic, missing); }\n"
        'extern "C" bool angular(unsigned long long work, unsigned long long classes, '
        "unsigned workers) {\n"
        "return generativeqc::scf::cuda_policy::force_prefers_angular_queue(\n"
        "    work, classes, workers); }\n"
    )
    library = root / "probe.so"
    native_cxx.build_shared(
        [source], library, compile_args=["-std=c++20", "-I", str(ROOT / "src")]
    )
    loaded = ctypes.CDLL(str(library))
    loaded.reject.argtypes = [ctypes.c_bool] * 4 + [ctypes.c_ulonglong]
    loaded.reject.restype = ctypes.c_bool
    loaded.angular.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_uint]
    loaded.angular.restype = ctypes.c_bool
    return loaded


@pytest.mark.parametrize(
    "compute,required,bounded,diagnostic,missing",
    itertools.product(
        (False, True), (False, True), (False, True), (False, True), (0, 1, 1 << 63)
    ),
)
def test_force_coverage_applies_only_to_force_requests(
    force_policy_probe: ctypes.CDLL,
    compute: bool,
    required: bool,
    bounded: bool,
    diagnostic: bool,
    missing: int,
) -> None:
    expected = compute and required and bounded and not diagnostic and missing != 0
    assert (
        force_policy_probe.reject(compute, required, bounded, diagnostic, missing)
        is expected
    )


@pytest.mark.parametrize("class_mask", (0, 1, 3, 1 << 63, (1 << 64) - 1))
@pytest.mark.parametrize("workers", (0, 1, 128, (1 << 32) - 1))
def test_force_angular_crossover_uses_actual_work_and_capacity(
    force_policy_probe: ctypes.CDLL, class_mask: int, workers: int
) -> None:
    capacity = class_mask.bit_count() * workers
    for work in (0, 1, capacity, capacity + 1, (1 << 64) - 1):
        expected = 0 < work <= capacity
        assert force_policy_probe.angular(work, class_mask, workers) is expected
