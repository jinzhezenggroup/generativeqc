"""Execute real generated registries with deterministic CUDA discovery stubs."""

from __future__ import annotations

import ctypes
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc import profiles
from generativeqc_compiler.integral.production_cost import shell_class_index
from generativeqc_compiler.integral.production_emission import emit_profile_shard
from generativeqc_compiler.integral.production_profile import resolve_production_profile
from generativeqc_compiler.integral.production_registry import (
    emit_multi_registry_header,
    emit_multi_registry_source,
)
from generativeqc_compiler.integral.shell_spec import FUSED_SHELL_SPEC_BY_NAME

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "python/generativeqc_compiler/integral/production_shell_classes.json"


def _registry_sources(
    directory: Path, targets: tuple[tuple[str, str], ...]
) -> list[Path]:
    """Keep actual profile resolution and dispatch, replacing only GPU bodies."""
    profiles = []
    wrappers = []
    for architecture, intent in targets:
        resolved = resolve_production_profile(MANIFEST, architecture, profile=intent)
        selection = next(
            item for item in resolved.selections if item.spec.name == "ssss"
        )
        resolved = replace(
            resolved,
            selections=(selection,),
            optional_fock_variants=False,
            preferred_rys_task_fock_shell_classes=(),
        )
        profiles.append(resolved)
        emitted = emit_profile_shard(resolved, resolved.selections)
        wrappers.extend(
            f"{signature} {{ return cudaSuccess; }}"
            for signature in re.findall(
                r'(extern "C" cudaError_t \w+\(.*?\)) \{', emitted, re.DOTALL
            )
        )
    (directory / "generativeqc_generated_shell_registry.hpp").write_text(
        emit_multi_registry_header(profiles)
    )
    registry = directory / "registry.cpp"
    registry.write_text(emit_multi_registry_source(profiles))
    wrapper = directory / "wrappers.cpp"
    wrapper.write_text('#include "scf/aot_shell_registry.hpp"\n' + "\n".join(wrappers))
    return [registry, wrapper]


def _cuda_stub(directory: Path, architecture: str) -> None:
    capability = int(architecture.removeprefix("sm_"))
    (directory / "cuda_runtime_api.h").write_text(f"""
#pragma once
using cudaStream_t = void*;
enum cudaError_t {{ cudaSuccess = 0, cudaErrorInvalidValue = 1,
                   cudaErrorNotSupported = 801 }};
enum cudaDeviceAttr {{ cudaDevAttrComputeCapabilityMajor,
                      cudaDevAttrComputeCapabilityMinor }};
struct cudaDeviceProp {{ int major; int minor; }};
extern int test_current_device;
extern bool test_runtime_failure;
inline cudaError_t cudaGetDevice(int* device) {{
  *device = test_current_device; return cudaSuccess;
}}
inline cudaError_t cudaSetDevice(int device) {{
  if (device < 0 || device > 1) return cudaErrorInvalidValue;
  test_current_device = device; return cudaSuccess;
}}
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* value, int device) {{
  if (device < 0 || device > 1) return cudaErrorInvalidValue;
  value->major = {capability // 10}; value->minor = {capability % 10};
  if (device == 1) {{ value->major = 8; value->minor = 9; }}
  return cudaSuccess;
}}
inline cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {{
  cudaDeviceProp properties{{}};
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
    return cudaErrorInvalidValue;
  *value = attr == cudaDevAttrComputeCapabilityMajor ? properties.major : properties.minor;
  return cudaSuccess;
}}
inline cudaError_t cudaRuntimeGetVersion(int* value) {{
  *value = 12090; return test_runtime_failure ? cudaErrorInvalidValue : cudaSuccess;
}}
inline cudaError_t cudaDriverGetVersion(int* value) {{
  *value = 13000;
  return cudaSuccess;
}}
""")
    (directory / "runtime.cpp").write_text(
        "int test_current_device = 1;\nbool test_runtime_failure = false;\n"
    )


SELECTION_CASES = (
    ("sm_120", (("sm_120", "auto"),), "sm_120", "matched"),
    ("sm_120", (("sm_120", "portable"),), "portable_cuda", "matched"),
    ("sm_80", (("sm_80", "auto"),), "portable_cuda", "matched"),
    ("sm_90", (("sm_90", "auto"),), "portable_cuda", "matched"),
    ("sm_89", (("sm_120", "auto"),), "generic_cuda", "no_matching_kernel_set"),
    ("sm_120", (), "generic_cuda", "no_matching_kernel_set"),
    ("sm_120", None, "generic_cuda", "aot_disabled"),
    (
        "sm_90",
        (("sm_80", "auto"), ("sm_90", "auto"), ("sm_120", "auto")),
        "portable_cuda",
        "matched",
    ),
)


@pytest.mark.parametrize(
    "architecture,targets,expected_profile,expected_status", SELECTION_CASES
)
def test_native_profile_contract_accepts_compiled_profile(
    tmp_path: Path,
    native_cxx: NativeCxx,
    architecture: str,
    targets: tuple[tuple[str, str], ...] | None,
    expected_profile: str,
    expected_status: str,
) -> None:
    """A physical SM120 does not override an explicitly portable build."""
    _cuda_stub(tmp_path, architecture)
    sources = (
        _registry_sources(tmp_path, targets)
        if targets is not None
        else [ROOT / "src/scf/aot_shell_registry_stub.cpp"]
    )
    executable = tmp_path / "profile-test"
    native_cxx.build_executable(
        [
            ROOT / "tests/native/test_aot_profile.cpp",
            *sources,
            tmp_path / "runtime.cpp",
        ],
        executable,
        compile_args=(
            "-std=c++17",
            f"-DGENERATIVEQC_TEST_AOT_SHELLS={int(targets is not None)}",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
        ),
        compile_timeout=60,
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=10
    )
    assert f" {expected_profile} " in result.stdout
    assert f"registry={expected_status}" in result.stdout


@pytest.mark.parametrize(
    "architecture,targets,expected_profile,expected_status", SELECTION_CASES
)
def test_public_profile_diagnostics_match_dispatch(
    tmp_path: Path,
    native_cxx: NativeCxx,
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
    targets: tuple[tuple[str, str], ...] | None,
    expected_profile: str,
    expected_status: str,
) -> None:
    """Compile the real C API and check ctypes layout, masks and device restore.

    Only discovery and device kernel bodies are fake. This is a host contract
    test, not independent numerical or device-performance qualification.
    """
    _cuda_stub(tmp_path, architecture)
    sources = (
        _registry_sources(tmp_path, targets)
        if targets is not None
        else [ROOT / "src/scf/aot_shell_registry_stub.cpp"]
    )
    (tmp_path / "build_identity.hpp").write_text(
        'constexpr char kGenerativeQCSourceIdentity[] = "host-contract-fixture";\n'
    )
    support = tmp_path / "support.cpp"
    support.write_text("""
#include "runtime/cuda_device_facts.hpp"
#include "scf/aot_shell_registry.hpp"
namespace generativeqc::runtime {
cudaError_t cuda_device_facts(int device, CudaTargetInfo& target, char (&name)[256]) noexcept {
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) return cudaErrorInvalidValue;
  target.compute_capability_major = properties.major;
  target.compute_capability_minor = properties.minor;
  name[0] = 'T';
  return cudaSuccess;
}
}
extern "C" int current_device() { return test_current_device; }
extern "C" void fail_runtime(int enabled) { test_runtime_failure = enabled; }
extern "C" int dispatch_fock() {
  const int previous = test_current_device;
  cudaSetDevice(0);
  double sum = 0;
  const auto result = generativeqc::scf::generated::launch_shell_class_fock(0, nullptr, false, 1,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr,
      {&sum, nullptr}, nullptr, nullptr);
  cudaSetDevice(previous);
  return result;
}
""")
    library_path = tmp_path / "libprobe.so"
    native_cxx.build_shared(
        [
            ROOT / "src/api/c_api_tuning.cpp",
            *sources,
            support,
            tmp_path / "runtime.cpp",
        ],
        library_path,
        compile_args=(
            "-std=c++17",
            "-DGENERATIVEQC_HAS_CUDA=1",
            "-DGENERATIVEQC_TUNING_RELEASE_BUILD=1",
            "-DGENERATIVEQC_CUDA_FAST_COMPILE=0",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
        compile_timeout=60,
    )
    library = ctypes.CDLL(str(library_path))
    for variable in (
        "GENERATIVEQC_AOT_SHELL_CLASSES",
        "GENERATIVEQC_AOT_FOCK_SHELL_CLASSES",
        "GENERATIVEQC_AOT_MIXED_FOCK_SHELL_CLASSES",
    ):
        monkeypatch.delenv(variable, raising=False)
    probe = profiles.probe_device(library)
    device = probe["device"]
    assert device["official_profile"] == expected_profile
    assert device["registry_status"] == expected_status
    assert device["tuned"] == (expected_profile == "sm_120")
    assert device["portable"] == (expected_profile != "sm_120")
    assert device["compatible"] == 0
    assert (device["major"], device["minor"]) == divmod(
        int(architecture.removeprefix("sm_")), 10
    )
    assert library.current_device() == 1
    # The compiled fixture has precisely the ssss class in both consumers.
    expected_mask = (
        1 << shell_class_index(FUSED_SHELL_SPEC_BY_NAME["ssss"])
        if expected_status == "matched"
        else 0
    )
    for field in ("enabled_force_shell_class_mask", "enabled_fock_shell_class_mask"):
        assert device[field] == expected_mask
    assert device["enabled_mixed_fock_shell_class_mask"] == 0
    assert library.dispatch_fock() == (0 if expected_mask else 1)
    obsolete = profiles.DeviceDescriptor()
    obsolete.struct_size = profiles.DeviceDescriptor.registry_status.offset
    obsolete.abi_version = probe["native_abi"]
    assert library.generativeqc_cuda_tuning_device(0, ctypes.byref(obsolete)) == 2
    assert library.current_device() == 1

    # A second device does not inherit the first device's selected kernel set.
    second = profiles.probe_device(library, 1)["device"]
    assert second["registry_status"] == (
        "aot_disabled" if targets is None else "no_matching_kernel_set"
    )
    assert second["enabled_fock_shell_class_mask"] == 0
    assert library.current_device() == 1

    monkeypatch.setenv("GENERATIVEQC_AOT_SHELL_CLASSES", "none")
    monkeypatch.setenv("GENERATIVEQC_AOT_FOCK_SHELL_CLASSES", "ssss")
    restricted = profiles.probe_device(library)["device"]
    assert restricted["enabled_force_shell_class_mask"] == 0
    assert restricted["enabled_fock_shell_class_mask"] == expected_mask
    monkeypatch.setenv("GENERATIVEQC_AOT_FOCK_SHELL_CLASSES", "none")
    masked = profiles.probe_device(library)["device"]
    assert masked["registry_status"] == expected_status
    assert masked["enabled_force_shell_class_mask"] == 0
    assert masked["enabled_fock_shell_class_mask"] == 0
    library.fail_runtime(1)
    with pytest.raises(RuntimeError, match="CUDA tuning device probe failed"):
        profiles.probe_device(library)
    assert library.current_device() == 1


def test_cpu_profile_probe_keeps_not_implemented_contract(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    (tmp_path / "build_identity.hpp").write_text(
        'constexpr char kGenerativeQCSourceIdentity[] = "host-contract-fixture";\n'
    )
    library_path = tmp_path / "libprobe.so"
    native_cxx.build_shared(
        [ROOT / "src/api/c_api_tuning.cpp"],
        library_path,
        compile_args=(
            "-std=c++17",
            "-DGENERATIVEQC_HAS_CUDA=0",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
    )
    library = ctypes.CDLL(str(library_path))
    with pytest.raises(RuntimeError, match="status 3"):
        profiles.probe_device(library)


def test_incompatible_tuned_profile_still_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_production_profile(MANIFEST, "sm_90", profile="sm_120")
