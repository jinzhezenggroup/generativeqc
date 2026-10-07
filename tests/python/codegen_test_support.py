"""Shared compatibility imports, fixtures, and constants for split codegen tests.

This module contains only test support. Responsibility-owned tests live in focused
``test_codegen_*.py`` modules; see #489.
"""

from __future__ import annotations

# ruff: noqa: F401  # Intentional re-exports for focused codegen test modules.
import itertools
import json
import math
import os
import re
import subprocess
import time
import typing
from dataclasses import replace
from pathlib import Path

import pytest
from codegen_fixtures import (
    boys_values,
    factored_dppp_variables,
    sample_variables,
)
from generativeqc_compiler.integral import (
    DDDD_SPEC,
    DDPS_SPEC,
    DPDS_SPEC,
    DPPP_SPEC,
    FDDD_SPEC,
    FFPS_SPEC,
    FUSED_SHELL_SPEC_BY_NAME,
    PSPS_SPEC,
    PSSS_SPEC,
    AlgebraPlacement,
    ContractionSpec,
    CudaTargetInfo,
    KernelConsumer,
    OperatorFamily,
    OperatorSpec,
    PairOrientation,
    PairStorage,
    RysRecurrenceKind,
    RysState,
    ScheduleIR,
    ScheduleKind,
    ShellClassSpec,
    TranslationInvariant,
    build_dppp_component_kernel,
    build_dppp_contraction_kernel,
    build_fused_shell_plan,
    build_integral_ir,
    build_ppps_rys_force_program,
    build_rys_axis_program,
    build_rys_force_program,
    build_shell_class_component_kernel,
    build_shell_class_contraction_kernel,
    build_weighted_shell_contraction_kernel,
    cartesian_components,
    cuda_target_info,
    emit_rys_force_root_body_cuda,
    emit_shell_class_fused_cuda,
    evaluate_fused_shell_component,
    evaluate_fused_shell_observables,
    evaluate_fused_shell_value,
    evaluate_ppps_rys_component,
    evaluate_rys_component,
    rys_boys_values,
    schedule_candidates,
    supports_component_lane_rys,
)
from generativeqc_compiler.integral.autotune import (
    supported_schedule_trials,
)
from generativeqc_compiler.integral.benchmark import (
    emit_shell_class_benchmark_cuda,
)
from generativeqc_compiler.integral.capabilities import (
    CAPABILITY_MIXED_FOCK,
    CAPABILITY_STREAMING_FOCK,
    build_capability_report,
)
from generativeqc_compiler.integral.production import (
    _PRODUCTION_PRELUDE,
    _partition_production_selections,
    emit_registry_header,
    emit_registry_source,
    load_production_fock_manifest,
    load_production_kernel_selections,
    load_production_manifest,
    write_production_bundle,
)
from generativeqc_compiler.integral.shell_class import (
    AXES,
)
from generativeqc_compiler.integral.weighted_eri_cuda import (
    emit_low_order_weighted_header,
)

TEST_CUDA_TARGET = cuda_target_info("sm_120")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


RTX5090_DPPP_RESOURCE_LIMITS = {
    "generated_dppp_shell_class_force_rhf_kernel": (168, 40, 2072),
    "generated_dppp_shell_class_force_uhf_kernel": (168, 40, 2072),
    "generated_dppp_shell_class_force_rhf_persistent_kernel": (164, 40, 2080),
    "generated_dppp_shell_class_force_uhf_persistent_kernel": (164, 40, 2080),
}
RTX5090_DPPP_UNIFORM_RYS4_RESOURCE_LIMITS = {
    # Relocatable production shards retain the 1 KiB component-activity table
    # that a whole-program cubin compile may fold into another shared region.
    # Record the larger production-object envelope observed with CUDA 12.9.
    "generated_dppp_shell_class_force_rhf_kernel": (255, 168, 37896),
    "generated_dppp_shell_class_force_uhf_kernel": (255, 168, 37896),
    "generated_dppp_shell_class_force_rhf_persistent_kernel": (
        255,
        168,
        37896,
    ),
    "generated_dppp_shell_class_force_uhf_persistent_kernel": (
        255,
        168,
        37896,
    ),
}
RTX5090_DPDS_RESOURCE_LIMITS = {
    "generated_dpds_shell_class_force_rhf_kernel": (160, 40, 1880),
    "generated_dpds_shell_class_force_uhf_kernel": (160, 40, 1880),
    "generated_dpds_shell_class_force_rhf_persistent_kernel": (160, 40, 1888),
    "generated_dpds_shell_class_force_uhf_persistent_kernel": (160, 40, 1888),
}
RTX5090_DDPS_RESOURCE_LIMITS = {
    "generated_ddps_shell_class_force_rhf_kernel": (164, 64, 1880),
    "generated_ddps_shell_class_force_uhf_kernel": (164, 64, 1880),
    "generated_ddps_shell_class_force_rhf_persistent_kernel": (160, 64, 1888),
    "generated_ddps_shell_class_force_uhf_persistent_kernel": (160, 64, 1888),
}
RTX5090_PSPS_RESOURCE_LIMITS = {
    "generated_psps_shell_class_force_rhf_persistent_kernel": (246, 0, 3408),
    "generated_psps_shell_class_force_uhf_persistent_kernel": (246, 0, 3408),
}
RTX5090_PPSS_RESOURCE_LIMITS = {
    "generated_ppss_shell_class_force_rhf_persistent_kernel": (246, 0, 3408),
    "generated_ppss_shell_class_force_uhf_persistent_kernel": (246, 0, 3408),
}
RTX5090_PPPS_SCALAR_THREAD_RESOURCE_LIMITS = {
    "generated_ppps_shell_class_force_rhf_persistent_kernel": (168, 0, 27000),
    "generated_ppps_shell_class_force_uhf_persistent_kernel": (168, 0, 27000),
}
RTX5090_DPSS_SCALAR_RYS3_RESOURCE_LIMITS = {
    "generated_dpss_shell_class_force_rhf_persistent_kernel": (252, 0, 6224),
    "generated_dpss_shell_class_force_uhf_persistent_kernel": (252, 0, 6224),
}


def _direct_cuda_source() -> typing.Any:
    """Read direct dispatch with its shared contracts and numerical owners."""
    root = REPOSITORY_ROOT / "src/scf"
    return "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "cuda/direct_constants.hpp",
            "cuda/integral_limits.hpp",
            "cuda/scalar_math.cuh",
            "cuda/gaussian_geometry.cuh",
            "cuda/cartesian_angular.cuh",
            "cuda/boys_table.cuh",
            "cuda/hermite_recurrence.cuh",
            "cuda/coulomb_auxiliary.cuh",
            "cuda/one_electron_native_overlap.cuh",
            "cuda/one_electron_native_attraction.cuh",
            "cuda/one_electron_native_contraction.cuh",
            "cuda/one_electron_reference.cu",
            "cuda/nuclear_kernels.cu",
            "cuda/direct_pair_cache.cu",
            "cuda/scf_constants.hpp",
            "cuda/scf_state_kernels.cu",
            "cuda/scf_matrix_kernels.cu",
            "cuda/scf_density_kernels.cu",
            "cuda/scf_diis_kernels.cu",
            "cuda/scf_convergence_kernels.cu",
            "cuda/basis_transform_kernels.cu",
            "cuda/launch_geometry.hpp",
            "cuda/direct_metadata.hpp",
            "cuda/direct_queue_index.cuh",
            "cuda/direct_screening.cuh",
            "cuda/direct_task_encoding.cuh",
            "cuda/direct_page_screening.cuh",
            "cuda/direct_queue_profile.cuh",
            "cuda/direct_tile_validation.cu",
            "cuda/direct_density_bounds.cu",
            "cuda/direct_tile_compaction.cu",
            "cuda/direct_generated_tasks.cu",
            "cuda/direct_resident_tasks.cu",
            "cuda/direct_bounded_pages.cu",
            "cuda/direct_bounded_tasks.cu",
            "cuda/direct_queue_scan.cu",
            "cuda/direct_queue_diagnostics.cu",
            "cuda/direct_gradient_types.cuh",
            "cuda/direct_native_psss.cuh",
            "cuda/eri_tensor_index.cuh",
            "cuda/direct_eri_symmetry.cuh",
            "cuda/direct_fock_accumulation.cuh",
            "cuda/direct_fock_quartet.cuh",
            "cuda/direct_fock_order2.cuh",
            "cuda/direct_force_density.cuh",
            "cuda/direct_force_low_order.cuh",
            "cuda/direct_force_order2.cuh",
            "cuda/direct_force_order3.cuh",
            "cuda/direct_force_quartet.cuh",
            "cuda/direct_bounded_contraction.cuh",
            "cuda/direct_cached_tensor_kernels.cu",
            "cuda/direct_schwarz_kernels.cu",
            "cuda/direct_packed_fock_kernels.cu",
            "cuda/direct_angular_fock.cu",
            "cuda/direct_reference_force.cu",
            "cuda/direct_bounded_dddd.cu",
            "cuda/direct_bounded_exact_force.cu",
            "cuda/direct_bounded_fallback.cu",
            "cuda/direct_angular_force.cu",
            "cuda/direct_jk_kernels.cu",
            "cuda/weighted_eri_kernels.cu",
            "cuda_rhf.cpp",
        )
    )
