"""Independent fixed-density CUDA gates for standalone Direct Rys artifacts.

The qualification build emits the ordinary compiler production shard for the
six value-IR classes in test_direct_rys_values. It supplies that ccache-built
library through GENERATIVEQC_RYS_VALUE_LIBRARY; this test never uses NVRTC or
loads a native reference recurrence as its oracle.
"""

import ctypes
import os
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.integral.production_profile import resolve_production_profile
from generativeqc_compiler.integral.production_rys_tasks import (
    direct_rys_task_candidates,
)
from generativeqc_compiler.integral.production_rys_values import (
    direct_rys_value_candidates,
)

# Qualify the complete compiled inventory, rather than using the earlier six
# arithmetic fixtures as a production class allowlist.
CLASSES = tuple(
    item.spec.name
    for item in (
        direct_rys_task_candidates
        if os.environ.get("GENERATIVEQC_RYS_VALUE_FAMILY") == "task"
        else direct_rys_value_candidates
    )(
        resolve_production_profile(
            Path(__file__).resolve().parents[2]
            / "python/generativeqc_compiler/integral/production_shell_classes.json",
            "sm_120",
        )
    )
)

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("GENERATIVEQC_RYS_VALUE_LIBRARY"),
    reason="requires a Slurm GPU and an explicitly built Rys value library",
)


class ShellTask(ctypes.Structure):
    """The stable compiler task ABI, checked independently of the emitted TU."""

    _fields_ = [
        ("primitive_begin", ctypes.c_uint64 * 4),
        ("primitive_end", ctypes.c_uint64 * 4),
        ("ao_begin", ctypes.c_uint64 * 4),
        ("ao_coefficient_begin", ctypes.c_uint64 * 4),
        ("density_offset", ctypes.c_uint64),
        ("spin_offset", ctypes.c_uint64),
        ("matrix_order", ctypes.c_uint32),
        ("shell_pair", ctypes.c_uint32 * 2),
        ("reversed_shell_pair_mask", ctypes.c_uint32),
        ("shell", ctypes.c_uint32 * 4),
        ("atom", ctypes.c_uint32 * 4),
    ]


@pytest.fixture(
    scope="module",
    ids=lambda case: "-".join(case),
    params=tuple(
        product(
            CLASSES,
            ("cartesian", "coincident", "reversed_pairs"),
        )
    ),
)
def quartet_case(request: pytest.FixtureRequest) -> tuple:
    """Materialize only a small independent oracle tensor on the host."""
    from tools.generativeqc_validation.f_shell_numerics import (
        _reference_integrals,
        eri_orbit,
        make_fixture,
    )

    name, variant = request.param
    fixture = make_fixture(name, variant)
    _, _, block, _ = _reference_integrals(fixture.inputs)
    n = fixture.density.shape[0]
    eri = np.zeros((n,) * 4)
    for component in np.ndindex(block.shape):
        indices = tuple(
            offset + index
            for offset, index in zip(fixture.ao_offsets, component, strict=True)
        )
        for orbit in eri_orbit(indices):
            eri[orbit] = block[component]
    return name, fixture, eri


@pytest.mark.parametrize("unrestricted", (False, True))
@pytest.mark.parametrize("consumer", ("combined", "j", "k", "hf-k"))
@pytest.mark.parametrize("task_count", (1, 33, 129))
def test_value_rys_fixed_density_matrices_match_libcint(
    quartet_case: tuple,
    unrestricted: bool,
    consumer: str,
    task_count: int,
) -> None:
    """Execute one fused queue task, including empty CTA and component tails."""
    import cupy as cp

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    assert ctypes.sizeof(ShellTask) == 192
    name, fixture, eri = quartet_case
    n = fixture.density.shape[0]
    task = ShellTask()
    for center in range(4):
        task.primitive_end[center] = len(fixture.inputs["shells"][center]["primitives"])
        task.ao_begin[center] = task.ao_coefficient_begin[center] = fixture.ao_offsets[
            center
        ]
        task.shell[center] = center
        task.atom[center] = fixture.atom_indices[center]
    task.matrix_order = n
    task.shell_pair[:] = (0, 1)
    task.reversed_shell_pair_mask = (
        fixture.reversed_mask | {"combined": 0, "j": 4, "k": 8, "hf-k": 12}[consumer]
    )
    arrays = [
        cp.asarray(np.frombuffer(bytes(task) * task_count, dtype=np.uint8)),
        cp.asarray([0], dtype=cp.uint32),
        cp.asarray([0, fixture.pair_split, len(fixture.pairs)], dtype=cp.int64),
        cp.asarray(fixture.pairs),
        cp.asarray(fixture.ao_coefficients),
        cp.asarray(fixture.positions),
    ]
    density = fixture.spin_density if unrestricted else fixture.density[None]
    device_density = cp.asarray(np.concatenate([d.ravel(order="F") for d in density]))
    output = cp.zeros(device_density.size, dtype=cp.float64)
    count = cp.asarray([task_count], dtype=cp.uint32)
    head = cp.zeros(1, dtype=cp.uint32)
    library = ctypes.CDLL(os.environ["GENERATIVEQC_RYS_VALUE_LIBRARY"])
    launch = getattr(library, f"generativeqc_launch_generated_{name}_fock")
    pointer = ctypes.c_void_p
    launch.argtypes = (
        [pointer, ctypes.c_bool, ctypes.c_uint]
        + [pointer] * 6
        + [ctypes.c_double]
        + [pointer] * 5
    )
    launch.restype = ctypes.c_int
    stream = cp.cuda.get_current_stream()
    status = launch(
        stream.ptr,
        unrestricted,
        2,
        *(a.data.ptr for a in arrays),
        0.0,
        None,
        device_density.data.ptr,
        output.data.ptr,
        count.data.ptr,
        head.data.ptr,
    )
    assert status == 0
    stream.synchronize()
    raw = cp.asnumpy(output)
    actual = np.stack(
        [
            raw[s * n * n : (s + 1) * n * n].reshape((n, n), order="F")
            for s in range(len(density))
        ]
    )
    j = np.einsum("abcd,cd->ab", eri, fixture.density)
    k = np.stack([np.einsum("acbd,cd->ab", eri, d) for d in density])
    expected = (
        -(1.0 if unrestricted else 0.5) * k
        if consumer == "hf-k"
        else k
        if consumer == "k"
        else np.repeat(j[None], len(density), axis=0)
        if consumer == "j"
        else j[None] - (1.0 if unrestricted else 0.5) * k
    )
    np.testing.assert_allclose(actual, task_count * expected, atol=2e-11, rtol=2e-10)
