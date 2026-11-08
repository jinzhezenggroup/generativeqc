"""Exercise independent raw-K capability, resource refusal and dispatch errors."""

from __future__ import annotations

import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.integral.production_profile import resolve_production_profile
from generativeqc_compiler.integral.production_registry import (
    emit_multi_registry_header,
    emit_multi_registry_source,
)
from generativeqc_compiler.integral.production_shared_rys_xy import (
    shared_rys_xy_streaming_companions,
)

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "python/generativeqc_compiler/integral/production_shell_classes.json"


@pytest.mark.parametrize("architecture", ("sm_80", "sm_90"))
def test_streaming_capability_is_not_paged_hf_admission(architecture: str) -> None:
    """Paged HF flags cannot grant or revoke streaming raw-K capability."""
    profile = resolve_production_profile(MANIFEST, architecture)
    companions = shared_rys_xy_streaming_companions(profile)
    assert {item.incumbent.spec.name for item in companions} == {
        "pppp",
        "dppp",
        "dpds",
        "dpdp",
        "ddps",
        "ddpp",
    }
    without_paged = replace(
        profile,
        selections=tuple(
            replace(item, capabilities=item.capabilities - {"shared_rys_xy_fock"})
            for item in profile.selections
        ),
    )
    assert [
        item.plan for item in shared_rys_xy_streaming_companions(without_paged)
    ] == [item.plan for item in companions]
    without_streaming = replace(
        profile,
        selections=tuple(
            replace(
                item, capabilities=item.capabilities - {"shared_rys_xy_streaming_fock"}
            )
            for item in profile.selections
        ),
    )
    assert shared_rys_xy_streaming_companions(without_streaming) == ()
    for item in companions:
        assert item.plan.schedule.fock_shared_rys_xy
        assert not item.plan.schedule.fock_shared_rys_weighted_z
        assert item.plan.block_threads <= profile.target.maximum_threads_per_block
    assert (
        shared_rys_xy_streaming_companions(
            resolve_production_profile(MANIFEST, "sm_120")
        )
        == ()
    )


def test_generated_registry_resources_and_no_post_enqueue_retry(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Unavailable resources refuse before writes; executed errors never retry."""
    profile = resolve_production_profile(MANIFEST, "sm_80")
    header = emit_multi_registry_header((profile,))
    registry = emit_multi_registry_source((profile,))
    (tmp_path / "generativeqc_generated_shell_registry.hpp").write_text(header)
    registry_path = tmp_path / "registry.cpp"
    registry_path.write_text(registry)
    (tmp_path / "cuda_runtime_api.h").write_text("""#pragma once
using cudaStream_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaErrorInvalidValue=1, cudaErrorNotSupported=801;
constexpr int cudaDevAttrComputeCapabilityMajor=1, cudaDevAttrComputeCapabilityMinor=2;
inline int cudaGetDevice(int* device) { *device=0; return 0; }
inline int cudaDeviceGetAttribute(int* value,int attribute,int) { *value=attribute==1?8:0; return 0; }
""")
    stubs = [
        '#include "scf/aot_shell_registry.hpp"',
        "bool resources=true; int writes=0; int error=0;",
    ]
    for symbol, arguments in re.findall(
        r'extern "C" cudaError_t (\w+)\((.*?)\);', registry, re.DOTALL
    ):
        stubs.append(
            f'extern "C" cudaError_t {symbol}({arguments}) {{ ++writes; return error; }}'
        )
    for symbol in re.findall(r'extern "C" bool (\w+)\(\);', registry):
        stubs.append(f'extern "C" bool {symbol}() {{ return resources; }}')
    driver = tmp_path / "driver.cpp"
    driver.write_text(
        "\n".join(stubs)
        + r"""
#include <cassert>
#include <cstdlib>
int main() {
  using namespace generativeqc::scf::generated;
  select_profile_for_device(0,8,0);
  const auto eligible=available_shared_rys_xy_streaming_fock_shell_class_mask();
  assert(eligible);
  resources=false;
  assert(available_shared_rys_xy_streaming_fock_shell_class_mask()==0);
  resources=true;
  setenv("GENERATIVEQC_AOT_FOCK_SHELL_CLASSES","none",1);
  assert(available_shared_rys_xy_streaming_fock_shell_class_mask()==0);
  unsetenv("GENERATIVEQC_AOT_FOCK_SHELL_CLASSES");
  unsigned shell=0; while(!(eligible & (1ULL<<shell))) ++shell;
  auto launch=[&](bool unrestricted=false,bool mixed=false,double* correction=nullptr) {
    return launch_shell_class_shared_rys_xy_streaming_fock(shell,nullptr,unrestricted,1,
        nullptr,nullptr,nullptr,nullptr,nullptr,0,mixed,0,nullptr,nullptr,
        {nullptr,correction},nullptr,nullptr,nullptr);
  };
  assert(launch()==cudaSuccess && writes==1);
  error=17;
  assert(launch()==17 && writes==2);
  double correction=0;
  assert(launch(true)==cudaErrorNotSupported && writes==2);
  assert(launch(false,true)==cudaErrorNotSupported && writes==2);
  assert(launch(false,false,&correction)==cudaErrorNotSupported && writes==2);
  shell=63;
  assert(launch()==cudaErrorNotSupported && writes==2);
}
"""
    )
    executable = tmp_path / "dispatch"
    required_native_cxx.build_executable(
        [registry_path, driver],
        executable,
        compile_args=("-std=c++20", f"-I{tmp_path}", f"-I{ROOT / 'src'}"),
        compile_timeout=60,
    )
    subprocess.run([str(executable)], check=True, timeout=15)
