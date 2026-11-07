"""Executable host admission/dispatch contracts for the opt-in HF companion."""

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


def test_companion_architecture_capability_and_qualified_mapping() -> None:
    selected = profile()
    companions = shared_rys_xy_companions(selected)
    assert {c.incumbent.spec.name: c.plan.block_threads for c in companions} == {
        "dppp": 192,
        "pppp": 96,
    }
    for architecture in ("sm_80", "sm_89", "sm_100", "sm_120"):
        assert shared_rys_xy_companions(profile(architecture)) == ()
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
  c.sm90 = c.quartet_direct = true; c.capability_mask = 3;
  auto on = resolve_shared_rys_xy_policy(SharedRysXyRequest::On, c);
  auto off = resolve_shared_rys_xy_policy(SharedRysXyRequest::Off, c);
  assert(on.active() && on.class_mask == 3 && !off.active() && on != off);
  assert(on == resolve_shared_rys_xy_policy(SharedRysXyRequest::On, c));
  assert(on.selects(0) && on.selects(1) && !on.selects(2) && !on.selects(64));
  assert(!off.selects(0));
  using A = SharedRysXyAdmission;
  auto refuse = [&](SharedRysXyContext bad, A why) {
    auto p = resolve_shared_rys_xy_policy(SharedRysXyRequest::On, bad);
    assert(!p.active() && p.class_mask == 0 && p.admission == why && p != on);
    std::puts(shared_rys_xy_admission_name(p.admission));
  };
  auto bad = c; bad.unrestricted = true; refuse(bad, A::Unrestricted);
  bad = c; bad.mixed_precision = true; refuse(bad, A::MixedPrecision);
  bad = c; bad.sm90 = false; refuse(bad, A::UnsupportedProfile);
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
    assert len(result.stdout.splitlines()) == 10


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
