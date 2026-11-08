"""Execute profile IO, actual topology/arena/binding and queued product adapters.

Synthetic rates exercise legal production selection; they are not device timing
evidence. NumPy independently checks the unchanged native kernel bodies and
vendor-call doubles, not CUDA/cuBLAS or full SCF endpoint qualification.
"""

from __future__ import annotations

import ctypes
import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from _cpp_source_support import cpp_function_definition
from _direct_schedule_test_support import write_direct_policy_headers
from generativeqc_compiler.tensor.matrix_product import emit_matrix_product_cuda

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
POINTER = ctypes.c_void_p


def payload(*, library: bool = True, sms: int = 170, memory: int = 32 << 30) -> str:
    """An intentionally synthetic, reproducible calibration for host probes."""
    rates = (
        "native 1000000 1 1\ncublas 1 1000000 1000000"
        if library
        else ("native 1 1000000 1000000\ncublas 1000000000 1 1")
    )
    return f"""generativeqc-hf-matrix-profile-v1
source_identity {"a" * 64}
provider nvidia
versions 13000 12090 12090 12 9 1
build_mode 1 0
native_build_id ee
runtime_build_id dd
driver_build_id cc
cublas_build_id bb
target 12 0 32 1024 2048 32 65536 49152 101376 102400 {sms} {memory} 64 2000000 14000000 512
bounds 1 32 1 100
evidence_sha256 {"f" * 64}
{rates}
provider_setup_nanoseconds 0
end
"""


@pytest.fixture(scope="module")
def consumer(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> ctypes.CDLL:
    folder = tmp_path_factory.mktemp("hf-matrix-profile")
    fixture = ROOT / "tests/native/fixtures/matrix_mask"
    write_direct_policy_headers(folder)
    (folder / "generated_matrix_product_lowering.hpp").write_text(
        emit_matrix_product_cuda(), encoding="utf-8"
    )
    kernels = (ROOT / "src/scf/cuda/scf_matrix_kernels.cu").read_text()
    source = (
        '#include "cuda_runtime.h"\n#include <cstdint>\n'
        "#define __global__\n#define __host__\n#define __device__\n"
        '#include "scf/cuda/matrix_index.cuh"\n'
        "using generativeqc::scf::cuda_execution::matrix_index;\n"
        "dim3 blockIdx, blockDim, threadIdx;\n"
    )
    wrappers = {
        "copy_selected_matrices_kernel": (
            "host_selected_copy",
            (
                "int batch, int spins, int dimension, const std::uint8_t* mask, "
                "const double* source, double* output"
            ),
            "batch, spins, dimension, mask, source, output",
        ),
        "matrix_product_kernel": (
            "host_matrix_product",
            (
                "int batch, int dimension, const double* left, bool transpose, "
                "const double* right, const std::uint8_t* active, double* output, double scale, "
                "bool transpose_right, int rank"
            ),
            "batch, dimension, left, transpose, right, active, output, scale, transpose_right, rank",
        ),
        "spin_matrix_product_kernel": (
            "host_spin_matrix_product",
            (
                "int batch, int spins, int dimension, const double* left, "
                "bool left_spin, bool transpose, const double* right, bool right_spin, "
                "const std::uint8_t* active, double* output"
            ),
            "batch, spins, dimension, left, left_spin, transpose, right, right_spin, active, output",
        ),
    }
    for kernel, (wrapper, parameters, arguments) in wrappers.items():
        source += cpp_function_definition(kernels, kernel) + "\n"
        source += f"""
void {wrapper}(dim3 grid, dim3 block, {parameters}) {{
  blockDim = block;
  for (blockIdx.x = 0; blockIdx.x < grid.x; ++blockIdx.x)
    for (threadIdx.x = 0; threadIdx.x < block.x; ++threadIdx.x)
      {kernel}({arguments});
}}
"""
    host_kernels = folder / "kernels.cpp"
    host_kernels.write_text(source, encoding="utf-8")
    library = required_native_cxx.build_shared(
        [
            ROOT / "tests/native/fixtures/hf_matrix_profile.cpp",
            fixture / "mock.cpp",
            host_kernels,
            ROOT / "src/scf/cuda/hf_matrix_profile.cpp",
            ROOT / "src/scf/cuda/topology.cpp",
            ROOT / "src/scf/cuda/arena.cpp",
            ROOT / "src/scf/cuda/rhf_policy.cpp",
            ROOT / "src/scf/cuda/matrix_library.cpp",
            ROOT / "src/tensor/cuda_matrix_product.cpp",
            ROOT / "src/runtime/cuda_provider.cpp",
            ROOT / "src/molecule/basis.cpp",
        ],
        folder / "consumer.so",
        compile_args=(
            "-std=c++20",
            "-O1",
            "-UNDEBUG",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DGENERATIVEQC_TEST_HOST_MATRIX_NUMERICS=1",
            f"-I{fixture}",
            f"-I{folder}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
    )
    probe = ctypes.CDLL(str(library))
    probe.profile_prepare.argtypes = [ctypes.c_char_p] + [ctypes.c_int] * 5 + [POINTER]
    probe.profile_prepare.restype = ctypes.c_int
    probe.profile_submit.argtypes = (
        [ctypes.c_int] * 6 + [POINTER] * 5 + [ctypes.c_size_t]
    )
    probe.profile_submit.restype = ctypes.c_int
    probe.probe_reset.argtypes = [ctypes.c_int, ctypes.c_int]
    probe.probe_stat.argtypes = [ctypes.c_int]
    return probe


def prepare(
    consumer: ctypes.CDLL,
    path: Path | None,
    *,
    dimension: int = 7,
    systems: int = 1,
    spins: int = 1,
    variant: int = 0,
    available: bool = True,
) -> list[int]:
    facts = (ctypes.c_uint64 * 9)()
    assert (
        consumer.profile_prepare(
            str(path).encode() if path else None,
            dimension,
            systems,
            spins,
            variant,
            available,
            facts,
        )
        == 0
    )
    return list(facts)


@pytest.mark.parametrize("systems,spins", [(1, 1), (1, 2), (3, 1), (3, 2)])
@pytest.mark.parametrize("dimension", [1, 7, 17, 32])
def test_real_preparation_selects_and_charges_each_legal_route(
    consumer: ctypes.CDLL, tmp_path: Path, systems: int, spins: int, dimension: int
) -> None:
    path = tmp_path / "profile"
    baseline = prepare(
        consumer, None, dimension=dimension, systems=systems, spins=spins
    )
    assert baseline[0:4] == [0, 0, int(dimension >= 17), int(dimension <= 16)]
    routes = []
    for library in (False, True):
        path.write_text(payload(library=library), encoding="utf-8")
        facts = prepare(
            consumer, path, dimension=dimension, systems=systems, spins=spins
        )
        assert facts[0:4] == [4, 1, int(library), int(dimension <= 16)]
        assert facts[5] == (dimension**2 * systems * spins * 8 if library else 0)
        assert facts[7:9] == [1, int(library)]
        routes.append(facts)
    assert routes[1][4] - routes[0][4] == dimension**2 * systems * spins * 8
    fallback = prepare(
        consumer,
        path,
        dimension=dimension,
        systems=systems,
        spins=spins,
        available=False,
    )
    assert fallback[2] == fallback[5] == fallback[8] == 0
    assert fallback[4] == routes[0][4]


@pytest.mark.parametrize("variant", list(range(1, 14)))
def test_exact_stack_rejects_stale_incompatible_or_unknown_identity(
    consumer: ctypes.CDLL, tmp_path: Path, variant: int
) -> None:
    path = tmp_path / "profile"
    path.write_text(payload(), encoding="utf-8")
    facts = prepare(consumer, path, variant=variant)
    assert facts[0:3] == [3, 0, 0]
    assert facts[7] == 0


@pytest.mark.parametrize("sms,variant", [(16, 1), (170, 0)])
def test_another_synthetic_target_can_use_its_own_profile(
    consumer: ctypes.CDLL, tmp_path: Path, sms: int, variant: int
) -> None:
    path = tmp_path / "profile"
    path.write_text(payload(sms=sms), encoding="utf-8")
    assert prepare(consumer, path, variant=variant)[0:3] == [4, 1, 1]


@pytest.mark.parametrize(
    "old,new",
    [
        ("profile-v1", "profile-v2"),
        ("provider nvidia", "provider cumetal"),
        ("bounds 1 32 1 100", "bounds -1 32 1 100"),
        ("bounds 1 32 1 100", "bounds 1 0 1 100"),
        ("bounds 1 32 1 100", "bounds 1 32 0 100"),
        ("bounds 1 32 1 100", "bounds 1 32 1 18446744073709551616"),
        ("native 1000000 1 1", "native 0 1 1"),
        ("native 1000000 1 1", "native nan 1 1"),
        ("cublas 1 1000000 1000000", "cublas inf 1000000 1000000"),
        ("cublas 1 1000000 1000000", "cublas 1 1e-320 1000000"),
        ("provider_setup_nanoseconds 0", "provider_setup_nanoseconds -1"),
        ("build_mode 1 0", "build_mode 2 0"),
        ("source_identity " + "a" * 64, "source_identity no-hash"),
        ("evidence_sha256 " + "f" * 64, "evidence_sha256 missing"),
        ("end", "end\ntrailing"),
        ("end", ""),
    ],
)
def test_invalid_records_never_partially_apply(
    consumer: ctypes.CDLL, tmp_path: Path, old: str, new: str
) -> None:
    path = tmp_path / "profile"
    path.write_text(payload().replace(old, new), encoding="utf-8")
    facts = prepare(consumer, path)
    assert facts[0:3] == [2, 0, 0]
    assert facts[7] == 0


def test_reload_identity_domain_and_first_use_cost(
    consumer: ctypes.CDLL, tmp_path: Path
) -> None:
    path = tmp_path / "profile"
    path.write_text(payload(), encoding="utf-8")
    consumer.profile_forget_previous()
    assert prepare(consumer, path)[6] == 0
    assert prepare(consumer, path)[6] == 1
    changed = payload().replace("native 1000000", "native 1000001")
    path.write_text(changed, encoding="utf-8")
    facts = prepare(consumer, path)
    assert facts[2] == 1 and facts[6] == 0
    path.write_text(changed.replace("f" * 64, "e" * 64), encoding="utf-8")
    assert prepare(consumer, path)[6] == 0
    path.write_text(payload().replace("bounds 1 32", "bounds 8 32"), encoding="utf-8")
    assert prepare(consumer, path)[0:3] == [4, 0, 0]
    path.write_text(
        payload().replace("bounds 1 32 1 100", "bounds 1 32 1 1"), encoding="utf-8"
    )
    assert prepare(consumer, path, spins=2)[0:3] == [4, 0, 0]
    # UHF also executes physical (unbroadcast) products; a profile covering
    # only the larger spin batch must not extrapolate to those smaller calls.
    path.write_text(
        payload().replace("bounds 1 32 1 100", "bounds 1 32 2 100"), encoding="utf-8"
    )
    assert prepare(consumer, path, spins=2)[0:3] == [4, 0, 0]
    path.write_text(
        payload().replace("nanoseconds 0", "nanoseconds 1e15"), encoding="utf-8"
    )
    assert prepare(consumer, path)[0:3] == [4, 1, 0]
    path.write_text("x" * 8193, encoding="utf-8")
    assert prepare(consumer, path)[0:3] == [2, 0, 0]
    path.unlink()
    assert prepare(consumer, path)[0:3] == [1, 0, 0]
    assert prepare(consumer, tmp_path)[0:3] == [1, 0, 0]


@pytest.mark.parametrize("library", [False, True])
@pytest.mark.parametrize("dimension", [1, 7, 17, 32])
@pytest.mark.parametrize("systems,spins", [(1, 1), (3, 2)])
@pytest.mark.parametrize(
    "left_spin,right_spin,transpose", itertools.product((False, True), repeat=3)
)
def test_profile_selected_prepared_adapter_matches_numpy(
    consumer: ctypes.CDLL,
    tmp_path: Path,
    library: bool,
    dimension: int,
    systems: int,
    spins: int,
    left_spin: bool,
    right_spin: bool,
    transpose: bool,
) -> None:
    path = tmp_path / "profile"
    path.write_text(payload(library=library), encoding="utf-8")
    assert prepare(consumer, path, dimension=dimension, systems=systems, spins=spins)[
        2
    ] == int(library)
    # A file edit cannot mutate an already prepared immutable operation binding.
    path.write_text(payload(library=not library), encoding="utf-8")
    rng = np.random.default_rng(597)
    left = rng.normal(size=(systems, spins if left_spin else 1, dimension, dimension))
    right = rng.normal(size=(systems, spins if right_spin else 1, dimension, dimension))
    expected = (left.swapaxes(-1, -2) if transpose else left) @ right
    expected = np.broadcast_to(expected, (systems, spins, dimension, dimension)).copy()
    left_buffer = np.ascontiguousarray(left.swapaxes(-1, -2))
    right_buffer = np.ascontiguousarray(right.swapaxes(-1, -2))
    active = np.ones(systems, dtype=np.uint8)
    if systems > 1:
        active[1] = 0
        expected[1] = 73
    output = np.full((systems, spins, dimension, dimension), 73, dtype=np.float64)
    scratch = np.empty_like(output)
    consumer.probe_reset(0, 0)
    assert (
        consumer.profile_submit(
            systems,
            spins,
            dimension,
            left_spin,
            transpose,
            right_spin,
            left_buffer.ctypes.data,
            right_buffer.ctypes.data,
            active.ctypes.data,
            output.ctypes.data,
            scratch.ctypes.data,
            scratch.size,
        )
        == 0
    )
    consumer.probe_replay()
    np.testing.assert_allclose(
        output.swapaxes(-1, -2), expected, rtol=2e-13, atol=2e-13
    )
    assert bool(consumer.probe_stat(0)) == library
    assert bool(consumer.probe_stat(3)) != library
