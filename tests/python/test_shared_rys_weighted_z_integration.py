"""Frozen combined-mode, independently declared linkage and resource identity."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.integral.production_emission import (
    emit_shared_rys_xy_fock_shard,
)
from generativeqc_compiler.integral.production_shared_rys_xy import (
    shared_rys_xy_companions,
)
from test_shared_rys_xy_integration import ROOT, _compile_registry, profile

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import NativeCxx
    from generativeqc_compiler.common.resources import ResourceIdentity


def test_weighted_companion_has_independent_capability_and_mapping() -> None:
    selected = profile()
    weighted = shared_rys_xy_companions(selected, weighted_z=True)
    assert {c.incumbent.spec.name: c.plan.block_threads for c in weighted} == {
        "dppp": 192,
        "pppp": 96,
    }
    assert all(c.plan.schedule.fock_shared_rys_weighted_z for c in weighted)
    without = replace(
        selected,
        selections=tuple(
            replace(s, capabilities=s.capabilities - {"shared_rys_weighted_z_fock"})
            for s in selected.selections
        ),
    )
    assert not shared_rys_xy_companions(without, weighted_z=True)
    assert len(shared_rys_xy_companions(without)) == 2
    for architecture in ("sm_80", "sm_89", "sm_100", "sm_120"):
        assert not shared_rys_xy_companions(profile(architecture), weighted_z=True)
    for item in weighted:
        emitted = emit_shared_rys_xy_fock_shard(
            selected, (item.incumbent,), weighted_z=True
        )
        assert emitted and emitted != emit_shared_rys_xy_fock_shard(
            selected, (item.incumbent,)
        )


@pytest.mark.parametrize(
    "legacy,available", ((False, True), (False, False), (True, True))
)
def test_weighted_actual_registry_fallback_and_no_post_enqueue_retry(
    tmp_path: Path, required_native_cxx: NativeCxx, legacy: bool, available: bool
) -> None:
    exe = _compile_registry(
        tmp_path,
        required_native_cxx,
        legacy=legacy,
        weighted=True,
        weighted_available=available,
    )
    subprocess.run([exe], check=True, capture_output=True, text=True, timeout=20)


def test_combined_selector_layout_and_bidirectional_plan_invalidation(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    import sys

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_one_electron_kernels.py"),
            "--derivatives",
            "--derivative-policy-output",
            str(tmp_path / "generated_one_electron_derivative_policy.cuh"),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    source = tmp_path / "weighted_policy.cpp"
    source.write_text(r"""
#include "scf/cuda/rhf_policy.hpp"
#include <cassert>
#include <cstdlib>
#include <cstddef>
using namespace generativeqc::scf::cuda_policy;
static_assert(static_cast<int>(SharedRysXyRequest::Off)==0);
static_assert(static_cast<int>(SharedRysXyRequest::On)==1);
static_assert(static_cast<int>(SharedRysXyRequest::Invalid)==2);
static_assert(static_cast<int>(SharedRysXyRequest::WeightedZOn)==3);
static_assert(sizeof(SharedRysXyRequest)==4 && sizeof(SharedRysXyPolicy)==16);
static_assert(offsetof(SharedRysXyPolicy,class_mask)==8);
int main() {
  using R=SharedRysXyRequest;using A=SharedRysXyAdmission;
  for(const char* v:{static_cast<const char*>(nullptr),"","0"}) {
    assert(parse_shared_rys_xy_request("1",v)==R::On);
    assert(parse_shared_rys_xy_request(v,"0")==R::Off);
    assert(parse_shared_rys_xy_request(v,"1")==R::Invalid);
  }
  assert(parse_shared_rys_xy_request("1","1")==R::WeightedZOn);
  for(const char* v:{"true","false","01"," 1","1 ","-1","2"}) {
    assert(parse_shared_rys_xy_request("1",v)==R::Invalid);
    assert(parse_shared_rys_xy_request(v,"0")==R::Invalid);
  }
  setenv("GENERATIVEQC_CUDA_HF_SHARED_RYS_XY","0",1);
  setenv("GENERATIVEQC_CUDA_HF_SHARED_RYS_WEIGHTED_Z","1",1);
  assert(shared_rys_xy_requested()==R::Invalid); // before dormant shape admission
  setenv("GENERATIVEQC_CUDA_HF_SHARED_RYS_XY","1",1);
  assert(shared_rys_xy_requested()==R::WeightedZOn);
  SharedRysXyContext c{};c.quartet_direct=true;c.capability_mask=3;
  generativeqc::runtime::CudaTargetInfo target{};
  target.warp_size=32;target.maximum_threads_per_block=1024;
  target.maximum_threads_per_sm=2048;target.maximum_blocks_per_sm=32;
  const auto xy=resolve_shared_rys_xy_policy(R::On,c,target);
  const auto weighted=resolve_shared_rys_xy_policy(R::WeightedZOn,c,target);
  assert(xy.active() && weighted.active() && xy.class_mask==weighted.class_mask);
  assert(!xy.weighted_z() && weighted.weighted_z() && weighted.selects(1));
  assert(xy!=weighted && weighted!=xy); // opposite control cannot reuse plan/graph
  assert(weighted==resolve_shared_rys_xy_policy(R::WeightedZOn,c,target));
  auto refuse=[&](SharedRysXyContext bad,A why) {
    const auto p=resolve_shared_rys_xy_policy(R::WeightedZOn,bad,target);
    assert(!p.active() && !p.selects(1) && p.class_mask==0 && p.admission==why);
  };
  auto bad=c;bad.capability_mask=0;refuse(bad,A::UnsupportedProfile);
  bad=c;bad.unrestricted=true;refuse(bad,A::Unrestricted);
  bad=c;bad.mixed_precision=true;refuse(bad,A::MixedPrecision);
  bad=c;bad.quartet_direct=false;refuse(bad,A::NonPaged);
  bad=c;bad.incremental_jk=true;refuse(bad,A::IncrementalJk);
  bad=c;bad.separate_jk=true;refuse(bad,A::SeparateJk);
  bad=c;bad.streaming=true;refuse(bad,A::Streaming);
  bad=c;bad.persistent_eri=true;refuse(bad,A::PersistentEri);
  bad=c;bad.compensated_output=true;refuse(bad,A::CompensatedOutput);
}
""")
    executable = required_native_cxx.build_executable(
        [
            source,
            ROOT / "src/scf/cuda/rhf_policy.cpp",
            ROOT / "src/runtime/cuda_provider.cpp",
        ],
        tmp_path / "weighted-policy",
        compile_args=(
            "-std=c++20",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
            f"-I{tmp_path}",
        ),
    )
    subprocess.run([executable], check=True, capture_output=True, text=True, timeout=20)


def test_weighted_cuda_only_exact_resource_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc import estimate_hf_resources
    from test_hf_cuda_fleet_resources import H2, layout_library

    def identity(backend: str) -> ResourceIdentity:
        return (
            estimate_hf_resources([H2], backend=backend, library=layout_library())
            .requests[0]
            .identity
        )

    cpu = identity("cpu")
    seen = []
    monkeypatch.setenv("GENERATIVEQC_CUDA_HF_SHARED_RYS_XY", "1")
    for value in (None, "", "0", "1", "invalid"):
        if value is None:
            monkeypatch.delenv(
                "GENERATIVEQC_CUDA_HF_SHARED_RYS_WEIGHTED_Z", raising=False
            )
        else:
            monkeypatch.setenv("GENERATIVEQC_CUDA_HF_SHARED_RYS_WEIGHTED_Z", value)
        current = identity("cuda")
        controls = json.loads(current.schedule)["runtime_policy"]
        assert controls["GENERATIVEQC_CUDA_HF_SHARED_RYS_WEIGHTED_Z"] == value
        assert identity("cpu") == cpu
        seen.append(current)
    assert len({x.schedule for x in seen}) == 5
