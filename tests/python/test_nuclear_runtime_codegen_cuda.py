"""Actual nuclear CUDA consumers against independent energy finite differences."""

import ctypes
import os
import shutil
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.method.nuclear_repulsion import emit_nuclear_repulsion_header

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def nuclear_probe(native_cxx: Any, tmp_path_factory: Any) -> Any:
    """Link the production TU, rather than copying its launch or kernel code."""
    if os.environ.get("GENERATIVEQC_RUN_CUDA_TESTS") != "1":
        pytest.skip("scheduled CUDA qualification is opt-in")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("nuclear runtime device qualification requires finite srun")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.fail("scheduled CUDA qualification requires nvcc")
    compiler = replace(native_cxx, compiler=nvcc)
    directory = tmp_path_factory.mktemp("nuclear-runtime-cuda")
    (directory / "generated_nuclear_repulsion.hpp").write_text(
        emit_nuclear_repulsion_header()
    )
    source = directory / "probe.cu"
    source.write_text(r"""
#include <vector>
#include "scf/cuda/nuclear_kernels.hpp"
#include "scf/cuda/one_electron_export_kernels.hpp"
#include "runtime/cuda_resources.cuh"
extern "C" int nuclear(int systems, const std::int64_t* offsets, const std::int32_t* numbers,
    const double* positions, const std::uint8_t* active, const double* seeds,
    std::int64_t coordinate, double* energy, double* exported_value,
    double* exported_derivative, double* forces) {
  using namespace generativeqc::scf::cuda_execution;
  using generativeqc::runtime::OwnedCudaBuffer;
  using generativeqc::runtime::cuda_resource_check;
  try {
    const auto atoms = offsets[systems];
    std::vector<std::int32_t> membership(atoms);
    for (int system = 0; system < systems; ++system)
      for (auto atom = offsets[system]; atom < offsets[system + 1]; ++atom) membership[atom] = system;
    OwnedCudaBuffer<std::int64_t> device_offsets(0, systems + 1);
    OwnedCudaBuffer<std::int32_t> device_numbers(0, atoms), device_membership(0, atoms);
    OwnedCudaBuffer<std::uint8_t> device_active(0, systems);
    OwnedCudaBuffer<double> device_positions(0, 3 * atoms), device_forces(0, 3 * atoms);
    OwnedCudaBuffer<double> device_energy(0, systems), device_value(0, systems), device_derivative(0, systems);
    cuda_resource_check(cudaMemcpy(device_offsets.get(), offsets, (systems + 1) * sizeof(*offsets), cudaMemcpyHostToDevice));
    cuda_resource_check(cudaMemcpy(device_numbers.get(), numbers, atoms * sizeof(*numbers), cudaMemcpyHostToDevice));
    cuda_resource_check(cudaMemcpy(device_membership.get(), membership.data(), atoms * sizeof(membership[0]), cudaMemcpyHostToDevice));
    cuda_resource_check(cudaMemcpy(device_active.get(), active, systems * sizeof(*active), cudaMemcpyHostToDevice));
    cuda_resource_check(cudaMemcpy(device_positions.get(), positions, 3 * atoms * sizeof(*positions), cudaMemcpyHostToDevice));
    cuda_resource_check(cudaMemcpy(device_forces.get(), seeds, 3 * atoms * sizeof(*seeds), cudaMemcpyHostToDevice));
    DeviceBatch batch{};
    batch.batch_size = systems;
    batch.total_atoms = atoms;
    batch.atom_offsets = device_offsets.get();
    batch.atom_systems = device_membership.get();
    batch.atomic_numbers = device_numbers.get();
    batch.positions = device_positions.get();
    const dim3 grid((systems + 63) / 64), block(64);
    launch_build_nuclear_repulsion_kernel(grid, block, 0, nullptr, batch, device_energy.get());
    cuda_resource_check(cudaGetLastError());
    launch_build_cuda_nuclear_repulsion_kernel(false, grid, block, 0, nullptr, batch, coordinate, device_value.get());
    cuda_resource_check(cudaGetLastError());
    launch_build_cuda_nuclear_repulsion_kernel(true, grid, block, 0, nullptr, batch, coordinate, device_derivative.get());
    cuda_resource_check(cudaGetLastError());
    launch_nuclear_force_kernel(dim3((3 * atoms + 127) / 128), dim3(128), 0, nullptr, batch, device_active.get(), device_forces.get());
    cuda_resource_check(cudaGetLastError());
    cuda_resource_check(cudaDeviceSynchronize());
    cuda_resource_check(cudaMemcpy(energy, device_energy.get(), systems * sizeof(*energy), cudaMemcpyDeviceToHost));
    cuda_resource_check(cudaMemcpy(exported_value, device_value.get(), systems * sizeof(*energy), cudaMemcpyDeviceToHost));
    cuda_resource_check(cudaMemcpy(exported_derivative, device_derivative.get(), systems * sizeof(*energy), cudaMemcpyDeviceToHost));
    cuda_resource_check(cudaMemcpy(forces, device_forces.get(), 3 * atoms * sizeof(*forces), cudaMemcpyDeviceToHost));
    return 0;
  } catch (...) { return 1; }
}
""")
    library = directory / "probe.so"
    compiler.build_shared(
        [ROOT / "src/scf/cuda/nuclear_kernels.cu", source],
        library,
        compile_args=(
            "-forward-unknown-to-host-compiler",
            "-std=c++20",
            "-O3",
            "--fmad=false",
            "--ftz=false",
            "-arch=native",
            f"-I{directory}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
    )
    loaded = ctypes.CDLL(str(library))
    function = loaded.nuclear
    array = lambda dtype: np.ctypeslib.ndpointer(dtype=dtype, flags="C_CONTIGUOUS")
    function.argtypes = [
        ctypes.c_int,
        array(np.int64),
        array(np.int32),
        array(np.float64),
        array(np.uint8),
        array(np.float64),
        ctypes.c_int64,
    ] + [array(np.float64)] * 4
    function.restype = ctypes.c_int
    return function


def _independent_energy(numbers: np.ndarray, coordinates: np.ndarray) -> float:
    """Pair distances define the reference; no generated derivative is reused."""
    value = np.longdouble(0)
    positions = coordinates.astype(np.longdouble)
    for first in range(len(numbers)):
        for second in range(first):
            displacement = positions[first] - positions[second]
            value += (
                int(numbers[first])
                * int(numbers[second])
                / np.sqrt(displacement @ displacement)
            )
    return float(value)


@pytest.mark.parametrize("coordinate", (-1, 0, 2, 4, 8, 999))
def test_ragged_nuclear_values_export_and_active_force_masks(
    nuclear_probe: Any, coordinate: int
) -> None:
    """Independent force signs, local export axes, empty peers and seed preservation."""
    offsets = np.array([0, 3, 3, 5, 6], dtype=np.int64)
    numbers = np.array([8, 1, 1, 7, 9, 2], dtype=np.int32)
    positions = np.array(
        [
            [0.1, -0.2, 0.3],
            [1.3, 0.1, -0.2],
            [-0.4, 1.4, 0.6],
            [0.2, 0.3, -0.1],
            [-1.2, 1.1, 0.7],
            [9.0, -3.0, 7.0],
        ]
    )
    seeds = np.arange(18, dtype=float).reshape(6, 3) / 7 + 1.0
    expected_gradient = np.zeros_like(positions)
    expected_energy = np.zeros(4)
    for system, (first, end) in enumerate(pairwise(offsets)):
        charges, geometry = numbers[first:end], positions[first:end]
        expected_energy[system] = _independent_energy(charges, geometry)
        for atom in range(end - first):
            for axis in range(3):
                direction = np.zeros_like(geometry)
                direction[atom, axis] = 1e-4
                values = [
                    _independent_energy(charges, geometry + multiplier * direction)
                    for multiplier in (-2, -1, 1, 2)
                ]
                expected_gradient[first + atom, axis] = (
                    values[0] - 8 * values[1] + 8 * values[2] - values[3]
                ) / 0.0012
    for mask in ([1, 1, 0, 1], [0, 0, 1, 0], [1, 1, 1, 1]):
        active = np.array(mask, dtype=np.uint8)
        energy, value, derivative = np.empty(4), np.empty(4), np.empty(4)
        forces = np.empty_like(positions)
        assert (
            nuclear_probe(
                4,
                offsets,
                numbers,
                positions,
                active,
                seeds,
                coordinate,
                energy,
                value,
                derivative,
                forces,
            )
            == 0
        )
        np.testing.assert_allclose(energy, expected_energy, rtol=3e-15, atol=1e-14)
        np.testing.assert_array_equal(value, energy)
        for system, (first, end) in enumerate(pairwise(offsets)):
            target = (
                -expected_gradient[first:end] if active[system] else seeds[first:end]
            )
            np.testing.assert_allclose(forces[first:end], target, rtol=2e-9, atol=3e-9)
            expected = (
                expected_gradient[first:end].ravel()[coordinate]
                if 0 <= coordinate < 3 * (end - first)
                else 0.0
            )
            np.testing.assert_allclose(
                derivative[system], expected, rtol=2e-9, atol=3e-9
            )
