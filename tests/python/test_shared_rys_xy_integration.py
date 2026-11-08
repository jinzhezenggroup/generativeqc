"""Executable host admission/dispatch contracts for target-scoped HF companions."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.integral.cuda_emitter import emit_shell_class_fused_cuda
from generativeqc_compiler.integral.production_emission import (
    _scope_profile_identifiers,
    _strip_emitter_includes,
    emit_shared_rys_xy_fock_shard,
)
from generativeqc_compiler.integral.production_profile import resolve_production_profile
from generativeqc_compiler.integral.production_registry import (
    emit_multi_registry_header,
    emit_multi_registry_source,
    emit_registry_header,
    emit_registry_source,
)
from generativeqc_compiler.integral.production_shared_rys_xy import (
    shared_rys_xy_companions,
)

if TYPE_CHECKING:
    from conftest import NativeCxx
    from generativeqc_compiler.common.resources import ResourceIdentity
    from generativeqc_compiler.integral.production_profile import (
        ResolvedProductionProfile,
    )

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "python/generativeqc_compiler/integral/production_shell_classes.json"


def profile(architecture: str = "sm_90") -> ResolvedProductionProfile:
    """Resolve production metadata without loading a native/GPU library."""
    return resolve_production_profile(MANIFEST, architecture)


@pytest.mark.parametrize("architecture", ("sm_80", "sm_90"))
def test_companion_architecture_capability_and_qualified_mapping(
    architecture: str,
) -> None:
    selected = profile(architecture)
    companions = shared_rys_xy_companions(selected)
    assert {c.incumbent.spec.name: c.plan.block_threads for c in companions} == {
        "dppp": 192,
        "pppp": 96,
    }
    assert shared_rys_xy_companions(profile("sm_120")) == ()
    if architecture == "sm_80":
        assert shared_rys_xy_companions(selected, weighted_z=True) == ()
    assert not shared_rys_xy_companions(
        replace(
            selected,
            selections=tuple(
                replace(s, capabilities=s.capabilities - {"shared_rys_xy_fock"})
                for s in selected.selections
            ),
        )
    )
    for c in companions:
        assert c.plan.schedule.fock_shared_rys_xy
        assert tuple(x.value for x in c.plan.kernel.integral.consumers) == ("fock",)
        # Explicit companion construction never turns a primary selection on.
        assert not c.incumbent.schedule.fock_shared_rys_xy
    header = emit_multi_registry_header((selected,))
    assert header.count('"shared_rys_xy_fock"') == 2


@pytest.mark.parametrize("name", ("dppp", "pppp"))
def test_packaging_preserves_qualified_arithmetic(name: str) -> None:
    selected = profile()
    c = next(
        c for c in shared_rys_xy_companions(selected) if c.incumbent.spec.name == name
    )
    # Same FOCK-only plan constructor and emitter as the independently qualified
    # fixture. Packaging only scopes identifiers and adds the host ABI wrapper.
    from tools.generativeqc_validation.shared_rys_xy_fixture import fixture_plan

    qualified = fixture_plan(name, shared=True)
    assert c.plan == qualified
    source = emit_shell_class_fused_cuda(c.incumbent.spec, qualified)
    packaged = emit_shared_rys_xy_fock_shard(selected, (c.incumbent,))
    expected = _scope_profile_identifiers(
        _strip_emitter_includes(source), c.incumbent, "sm90_shared_rys_xy"
    )
    assert expected in packaged
    wrapper = packaged[packaged.index('extern "C"') :]
    assert "fock_uhf_persistent_kernel<<<" not in wrapper
    assert "fock.correction != nullptr" in wrapper
    assert "fock_rhf_persistent_kernel<<<" in wrapper


@pytest.mark.parametrize("architecture", ("sm_89", "sm_100", "sm_120"))
def test_unweighted_companion_follows_declared_capability(
    architecture: str,
) -> None:
    """An explicitly supplied capability is not vetoed by an ISA-name list.

    This synthetic inventory checks compiler legality, not performance authority
    for these targets. Capability and performance qualification remain separate.
    """
    selected = replace(profile(architecture), selections=profile().selections)
    assert {
        item.incumbent.spec.name for item in shared_rys_xy_companions(selected)
    } == {
        "dppp",
        "pppp",
    }
    assert not shared_rys_xy_companions(selected, weighted_z=True)


def test_policy_strict_parse_refusal_and_frozen_identity(
    tmp_path: Path,
    required_native_cxx: NativeCxx,
) -> None:
    header = tmp_path / "generated_one_electron_derivative_policy.cuh"
    import sys

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_one_electron_kernels.py"),
            "--derivatives",
            "--derivative-policy-output",
            str(header),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    source = tmp_path / "policy_test.cpp"
    source.write_text(r"""
#include "scf/cuda/rhf_policy.hpp"
#include <cassert>
#include <cstdio>
using namespace generativeqc::scf::cuda_policy;
int main() {
  for (const char* v : {static_cast<const char*>(nullptr), "", "0"})
    assert(parse_shared_rys_xy_request(v) == SharedRysXyRequest::Off);
  assert(parse_shared_rys_xy_request("1") == SharedRysXyRequest::On);
  for (const char* v : {"true", "false", "01", " 1", "1 ", "-1", "2"})
    assert(parse_shared_rys_xy_request(v) == SharedRysXyRequest::Invalid);
  SharedRysXyContext c{};
  c.quartet_direct = true; c.capability_mask = 3;
  generativeqc::runtime::CudaTargetInfo target{};
  target.warp_size = 32;
  target.maximum_threads_per_block = 1024;
  target.maximum_threads_per_sm = 2048;
  target.maximum_blocks_per_sm = 32;
  unsetenv("GENERATIVEQC_CUDA_HF_SHARED_RYS_XY");
  unsetenv("GENERATIVEQC_CUDA_HF_SHARED_RYS_WEIGHTED_Z");
  assert(shared_rys_xy_requested() == SharedRysXyRequest::Auto);
  setenv("GENERATIVEQC_CUDA_HF_SHARED_RYS_XY", "", 1);
  assert(shared_rys_xy_requested() == SharedRysXyRequest::Auto);
  setenv("GENERATIVEQC_CUDA_HF_SHARED_RYS_WEIGHTED_Z", "0", 1);
  assert(shared_rys_xy_requested() == SharedRysXyRequest::Auto);
  setenv("GENERATIVEQC_CUDA_HF_SHARED_RYS_XY", "0", 1);
  assert(shared_rys_xy_requested() == SharedRysXyRequest::Off);
  setenv("GENERATIVEQC_CUDA_HF_SHARED_RYS_WEIGHTED_Z", "1", 1);
  assert(shared_rys_xy_requested() == SharedRysXyRequest::Invalid);
  unsetenv("GENERATIVEQC_CUDA_HF_SHARED_RYS_WEIGHTED_Z");
  const auto automatic = SharedRysXyRequest::Auto;
  assert(!resolve_shared_rys_xy_policy(automatic, c, target).active());
  target.fp32_fp64_ratio = 2;
  const auto default_on = resolve_shared_rys_xy_policy(automatic, c, target);
  assert(default_on.active() && default_on.class_mask == 3 && !default_on.weighted_z());
  auto on = resolve_shared_rys_xy_policy(SharedRysXyRequest::On, c, target);
  auto off = resolve_shared_rys_xy_policy(SharedRysXyRequest::Off, c, target);
  assert(on.active() && on.class_mask == 3 && !off.active() && on != off);
  assert(on == resolve_shared_rys_xy_policy(SharedRysXyRequest::On, c, target));
  assert(on.selects(0) && on.selects(1) && !on.selects(2) && !on.selects(64));
  assert(!off.selects(0));
  using A = SharedRysXyAdmission;
  // Throughput and compiled availability are independent of ISA identity.
  for (int major : {7, 8, 9, 10, 12, 20}) {
    for (int minor : {0, 1, 6}) {
      auto other = target;
      other.compute_capability_major = major;
      other.compute_capability_minor = minor;
      for (int ratio : {-1, 0, 1, 2, 3, 32, 64}) {
        other.fp32_fp64_ratio = ratio;
        assert(resolve_shared_rys_xy_policy(automatic, c, other).active() ==
               (ratio > 0 && ratio <= 2));
        assert(resolve_shared_rys_xy_policy(SharedRysXyRequest::On, c, other).active());
        assert(!resolve_shared_rys_xy_policy(SharedRysXyRequest::Off, c, other).active());
      }
    }
  }
  auto refuse_resources = [&](const generativeqc::runtime::CudaTargetInfo& other) {
    for (auto request : {automatic, SharedRysXyRequest::On, SharedRysXyRequest::WeightedZOn}) {
      const auto policy = resolve_shared_rys_xy_policy(request, c, other);
      assert(!policy.active() && policy.class_mask == 0 &&
             policy.admission == A::UnsupportedResources);
    }
  };
  auto insufficient = target;
  insufficient.warp_size = 64; refuse_resources(insufficient);
  insufficient = target; insufficient.maximum_threads_per_block = 191;
  refuse_resources(insufficient);
  insufficient = target; insufficient.maximum_threads_per_sm = 191;
  refuse_resources(insufficient);
  insufficient = target; insufficient.maximum_blocks_per_sm = 0;
  refuse_resources(insufficient);
  auto minimum = target;
  minimum.maximum_threads_per_block = minimum.maximum_threads_per_sm = 192;
  minimum.maximum_blocks_per_sm = 1;
  assert(resolve_shared_rys_xy_policy(automatic, c, minimum).active());
  auto refuse = [&](SharedRysXyContext bad, A why) {
    auto p = resolve_shared_rys_xy_policy(SharedRysXyRequest::On, bad, target);
    assert(!p.active() && p.class_mask == 0 && p.admission == why && p != on);
    const auto automatic_refusal = resolve_shared_rys_xy_policy(automatic, bad, target);
    assert(!automatic_refusal.active() && automatic_refusal.class_mask == 0 &&
           automatic_refusal.admission == why);
    std::puts(shared_rys_xy_admission_name(p.admission));
  };
  auto bad = c; bad.unrestricted = true; refuse(bad, A::Unrestricted);
  bad = c; bad.mixed_precision = true; refuse(bad, A::MixedPrecision);
  bad = c; bad.capability_mask = 0; refuse(bad, A::UnsupportedProfile);
  bad = c; bad.quartet_direct = false; refuse(bad, A::NonPaged);
  bad = c; bad.incremental_jk = true; refuse(bad, A::IncrementalJk);
  bad = c; bad.separate_jk = true; refuse(bad, A::SeparateJk);
  bad = c; bad.streaming = true; refuse(bad, A::Streaming);
  bad = c; bad.persistent_eri = true; refuse(bad, A::PersistentEri);
  bad = c; bad.compensated_output = true; refuse(bad, A::CompensatedOutput);
}
""")
    exe = required_native_cxx.build_executable(
        [
            source,
            ROOT / "src/scf/cuda/rhf_policy.cpp",
            ROOT / "src/runtime/cuda_provider.cpp",
        ],
        tmp_path / "policy-test",
        compile_args=(
            "-std=c++20",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
            f"-I{tmp_path}",
        ),
    )
    result = subprocess.run(
        [exe], check=True, capture_output=True, text=True, timeout=20
    )
    assert len(result.stdout.splitlines()) == 9


def _compile_registry(
    tmp_path: Path,
    required_native_cxx: NativeCxx,
    *,
    legacy: bool = False,
    weighted: bool = False,
    weighted_available: bool = True,
) -> Path:
    selected = profile()
    if not weighted_available:
        selected = replace(
            selected,
            selections=tuple(
                replace(s, capabilities=s.capabilities - {"shared_rys_weighted_z_fock"})
                for s in selected.selections
            ),
        )
    selected = replace(
        selected,
        selections=tuple(
            s for s in selected.selections if s.spec.name in ("ssss", "dppp", "pppp")
        ),
    )
    (tmp_path / "generativeqc_generated_shell_registry.hpp").write_text(
        emit_registry_header(selected.selections)
        if legacy
        else emit_multi_registry_header((selected,))
    )
    registry = (
        emit_registry_source(selected.selections)
        if legacy
        else emit_multi_registry_source((selected,))
    )
    # Typed fake CUDA leaves execute the real generated registry on the host.
    (tmp_path / "cuda_runtime_api.h").write_text("""#pragma once
using cudaStream_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaErrorInvalidValue=1, cudaErrorNotSupported=801;
constexpr int cudaDevAttrComputeCapabilityMajor=1, cudaDevAttrComputeCapabilityMinor=2;
inline int cudaGetDevice(int* d) { *d=0; return 0; }
extern int gpu_major;
inline int cudaDeviceGetAttribute(int* v, int a, int) { *v=a==1?gpu_major:0; return 0; }
""")
    declarations = re.findall(
        r'extern "C" cudaError_t (\w+)\((.*?)\);', registry, re.DOTALL
    )
    stubs = [
        '#include "scf/aot_shell_registry.hpp"\n#include "scf/shared_rys_weighted_z.hpp"',
        "int gpu_major=9; int primary=0,candidate=0;",
        "int candidate_error=0;",
    ]
    for symbol, args in declarations:
        shared = (
            "_shared_rys_weighted_z_" if weighted else "_shared_rys_xy_"
        ) in symbol
        count = "candidate" if shared else "primary"
        value = "candidate_error" if shared else "0"
        stubs.append(
            f'extern "C" cudaError_t {symbol}({args}) {{ ++{count}; return {value}; }}'
        )
    registry_path = tmp_path / "registry.cpp"
    registry_path.write_text(registry)
    source = tmp_path / "registry_test.cpp"
    source.write_text(
        "\n".join(stubs)
        + r"""
#include <cassert>
#include <cstdio>
using namespace generativeqc::scf::generated;
int main() {
  select_profile_for_device(0,9,0);
  const auto mask = enabled_shared_rys_xy_fock_shell_class_mask();
  unsigned cls=0; while(cls<64 && !(mask & (1ULL<<cls))) ++cls;
  auto launch = [&](unsigned c, bool u=false, double* correction=nullptr) {
    return launch_shell_class_shared_rys_xy_fock(c,nullptr,u,1,nullptr,nullptr,
      nullptr,nullptr,nullptr,nullptr,0,nullptr,nullptr,{nullptr,correction},nullptr,nullptr);
  };
  if (mask==0) { assert(launch(0)==0 && candidate==0 && primary==1); return 0; }
  assert(launch(cls)==0 && candidate==1 && primary==0);
  candidate_error=17;
  assert(launch(cls)==17 && candidate==2 && primary==0);
  assert(launch(cls,true)==0 && candidate==2 && primary==1);
  double correction=0;
  assert(launch(cls,false,&correction)==0 && candidate==2 && primary==2);
  assert(launch(0)==0 && candidate==2 && primary==3);
  // Ordinary dispatcher never opts in, including after a candidate error.
  assert(launch_shell_class_fock(cls,nullptr,false,1,nullptr,nullptr,nullptr,
    nullptr,nullptr,nullptr,0,nullptr,nullptr,nullptr,nullptr,nullptr)==0);
  assert(primary==4 && candidate==2);
  gpu_major=8; select_profile_for_device(0,8,0);
  assert(enabled_shared_rys_xy_fock_shell_class_mask()==0);
  assert(launch(cls)!=0 && candidate==2);
  std::puts("companion-selection-and-no-post-enqueue-retry:PASS");
}
"""
    )
    if legacy:
        source.write_text(
            source.read_text()
            .replace("select_profile_for_device(0,9,0);", "")
            .replace("select_profile_for_device(0,8,0);", "")
        )
    if weighted:
        program = source.read_text()
        client = program.index("#include <cassert>")
        source.write_text(
            program[:client]
            + program[client:].replace("shared_rys_xy", "shared_rys_weighted_z")
        )
    return required_native_cxx.build_executable(
        [registry_path, source],
        tmp_path / "registry-test",
        compile_args=(
            "-std=c++20",
            f"-I{ROOT / 'src'}",
            f"-I{tmp_path}",
        ),
    )


@pytest.mark.parametrize("legacy", (False, True))
def test_actual_generated_dispatch_fallback_and_error(
    tmp_path: Path, required_native_cxx: NativeCxx, legacy: bool
) -> None:
    exe = _compile_registry(tmp_path, required_native_cxx, legacy=legacy)
    subprocess.run([exe], check=True, capture_output=True, text=True, timeout=20)


def test_cuda_only_exact_prepared_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from generativeqc import estimate_hf_resources
    from test_hf_cuda_fleet_resources import H2, layout_library

    def identity(backend: str) -> ResourceIdentity:
        report = estimate_hf_resources([H2], backend=backend, library=layout_library())
        return report.requests[0].identity

    cpu = identity("cpu")
    seen = []
    for value in (None, "", "0", "1", "invalid"):
        if value is None:
            monkeypatch.delenv("GENERATIVEQC_CUDA_HF_SHARED_RYS_XY", raising=False)
        else:
            monkeypatch.setenv("GENERATIVEQC_CUDA_HF_SHARED_RYS_XY", value)
        current = identity("cuda")
        controls = json.loads(current.schedule)["runtime_policy"]
        assert controls["GENERATIVEQC_CUDA_HF_SHARED_RYS_XY"] == value
        assert identity("cpu") == cpu
        seen.append(current)
    assert len({x.schedule for x in seen}) == 5


@pytest.mark.parametrize("weighted", (False, True))
def test_prepared_dispatch_fixture_compiles_against_native_contract(
    tmp_path: Path, required_native_cxx: NativeCxx, weighted: bool
) -> None:
    """Compile complete generated drivers without a GPU or new oracle work.

    Synthetic arrays exercise only driver/API integration. They are never run
    or presented as numerical evidence. CUDA declarations are a host double;
    all policy, registry, task and output contracts come from current headers.
    """
    import hashlib

    from tools.generativeqc_validation import (
        shared_rys_weighted_z_dispatch_fixture,
        shared_rys_xy_dispatch_fixture,
    )

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    cases = [
        {
            "name": name,
            "variant": variant,
            "n": 1,
            "pair_count": 2,
            "pair_split": 1,
            "reversed_mask": 0,
        }
        for name in ("dppp", "pppp")
        for variant in ("ordinary", "reversed", "displaced")
    ]
    arrays = "".join(
        f"static const {ctype} {case['name']}_{case['variant']}_{field}[16]={{}};\n"
        for case in cases
        for field, ctype in (
            ("pairs", "double"),
            ("coefficients", "double"),
            ("positions", "double"),
            ("bounds", "double"),
            ("density", "double"),
            ("expected", "double"),
            ("ao_offsets", "std::uint64_t"),
            ("atoms", "std::uint32_t"),
            ("primitive_counts", "std::uint64_t"),
        )
    )
    (frozen / "shared_rys_xy_fixture.cu").write_text(arrays)
    (frozen / "receipt.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(arrays.encode()).hexdigest(),
                "cases": cases,
                "masks": ["full", "partial", "one", "two", "zero"],
                "task_mask_sequence": [0, 4, 1, 2, 3, 0, 4],
                "fock_tolerances": {"atol": 2e-11, "rtol": 2e-10},
            }
        )
    )
    driver = (
        shared_rys_weighted_z_dispatch_fixture
        if weighted
        else shared_rys_xy_dispatch_fixture
    )
    output = tmp_path / "prepared"
    driver.prepare_dispatch_fixture(frozen, output)
    cuda = tmp_path / "cuda_runtime_api.h"
    cuda.write_text(r"""
#pragma once
#include <cstddef>
using cudaError_t=int;
using cudaStream_t=void*;
using cudaGraph_t=void*;
using cudaGraphExec_t=void*;
constexpr int cudaSuccess=0, cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2;
constexpr int cudaStreamCaptureModeThreadLocal=0;
constexpr int cudaDevAttrComputeCapabilityMajor=0, cudaDevAttrComputeCapabilityMinor=1;
struct cudaFuncAttributes {
  int numRegs{},maxThreadsPerBlock{};
  std::size_t localSizeBytes{},sharedSizeBytes{};
};
const char* cudaGetErrorString(cudaError_t);
template<class T> cudaError_t cudaMalloc(T**,std::size_t);
cudaError_t cudaMemcpy(void*,const void*,std::size_t,int);
cudaError_t cudaMemsetAsync(void*,int,std::size_t,cudaStream_t);
cudaError_t cudaPeekAtLastError();
cudaError_t cudaStreamBeginCapture(cudaStream_t,int);
cudaError_t cudaStreamEndCapture(cudaStream_t,cudaGraph_t*);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t*,cudaGraph_t,void*,void*,unsigned long long);
cudaError_t cudaGraphLaunch(cudaGraphExec_t,cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);
cudaError_t cudaGraphDestroy(cudaGraph_t);
cudaError_t cudaStreamCreate(cudaStream_t*);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaFree(void*);
cudaError_t cudaGetDevice(int*);
cudaError_t cudaDeviceGetAttribute(int*,int,int);
template<class T> cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*,T);
template<class T> cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*,T,int,std::size_t);
""")
    (tmp_path / "cuda_runtime.h").write_text('#include "cuda_runtime_api.h"\n')
    source = next(output.glob("*.cu"))
    required_native_cxx.compile_object(
        source,
        tmp_path / "driver.o",
        args=(
            "-std=c++20",
            "-x",
            "c++",
            "-UNDEBUG",
            "-I",
            str(tmp_path),
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
        ),
    )
