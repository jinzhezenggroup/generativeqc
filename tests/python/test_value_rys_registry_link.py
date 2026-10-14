"""Value-only registries link without inventing uncompiled force consumers."""

from __future__ import annotations

import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.integral.ir import KernelConsumer
from generativeqc_compiler.integral.production_emission import (
    emit_production_shard,
    emit_profile_shard,
)
from generativeqc_compiler.integral.production_profile import resolve_production_profile
from generativeqc_compiler.integral.production_registry import (
    emit_multi_registry_header,
    emit_multi_registry_source,
    emit_registry_header,
    emit_registry_source,
    shell_class_index,
)

if TYPE_CHECKING:
    from conftest import NativeCxx


@pytest.mark.parametrize("multi", (False, True))
def test_value_only_registry_links_and_rejects_force_dispatch(
    tmp_path: Path, native_cxx: NativeCxx, multi: bool
) -> None:
    """Link actual emitted entrypoints and execute host-only dispatch.

    CUDA discovery and device launch bodies are stubs; the emitted registry,
    wrapper ABI and symbol inventory are real. Lazy Python library loading must
    not hide a missing force symbol from executable consumers.
    """
    root = Path(__file__).resolve().parents[2]
    profile = resolve_production_profile(
        root / "python/generativeqc_compiler/integral/production_shell_classes.json",
        "sm_120",
    )
    selections = tuple(
        item
        for item in profile.selections
        if item.consumers == (KernelConsumer.FOCK,)
        and item.recurrence.startswith("rys")
    )
    assert selections
    profile = replace(
        profile, selections=selections, preferred_rys_task_fock_shell_classes=()
    )
    (tmp_path / "cuda_runtime_api.h").write_text(
        "#pragma once\nusing cudaStream_t = void*;\n"
        "enum cudaError_t { cudaSuccess = 0, cudaErrorInvalidValue = 1, "
        "cudaErrorNotSupported = 801 };\n"
        "enum cudaDeviceAttr { cudaDevAttrComputeCapabilityMajor, "
        "cudaDevAttrComputeCapabilityMinor };\n"
        "inline cudaError_t cudaGetDevice(int* device) { *device = 0; return cudaSuccess; }\n"
        "inline cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int) {\n"
        "  *value = attr == cudaDevAttrComputeCapabilityMajor ? 12 : 0;\n"
        "  return cudaSuccess;\n}\n"
    )
    header = (
        emit_multi_registry_header((profile,))
        if multi
        else emit_registry_header(selections)
    )
    registry_source = (
        emit_multi_registry_source((profile,))
        if multi
        else emit_registry_source(selections)
    )
    emitted = (
        emit_profile_shard(profile, selections)
        if multi
        else emit_production_shard(selections)
    )
    (tmp_path / "generativeqc_generated_shell_registry.hpp").write_text(header)
    registry = tmp_path / "registry.cpp"
    registry.write_text(registry_source)
    definitions = [
        f"{signature} {{ *fock.sum += 1.0; return cudaSuccess; }}"
        for signature in re.findall(
            r'(extern "C" cudaError_t \w+\(.*?\)) \{', emitted, re.DOTALL
        )
    ]
    wrappers = tmp_path / "wrappers.cpp"
    wrappers.write_text(
        '#include "generativeqc_generated_shell_registry.hpp"\n'
        + "\n".join(definitions)
    )
    classes = ", ".join(str(shell_class_index(item.spec)) for item in selections)
    driver = tmp_path / "driver.cpp"
    driver.write_text(
        f"""
#include "generativeqc_generated_shell_registry.hpp"
#include <cassert>
using namespace generativeqc::scf::generated;
int main() {{
  double sum = 0;
  for (unsigned shell_class : {{{classes}}}) {{
    auto error = launch_shell_class_fock(shell_class, nullptr, false, 1, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr,
        {{&sum, nullptr}}, nullptr, nullptr);
    assert(error == cudaSuccess);
    error = launch_shell_class(shell_class, nullptr, false, 1, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr,
        nullptr, nullptr, nullptr);
    assert(error == cudaErrorInvalidValue);
  }}
  assert(sum == {len(selections)});
}}
"""
    )
    executable = tmp_path / "dispatch"
    native_cxx.build_executable(
        (driver, registry, wrappers),
        executable,
        compile_args=("-std=c++17", f"-I{tmp_path}", f"-I{root / 'src'}"),
        compile_timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
