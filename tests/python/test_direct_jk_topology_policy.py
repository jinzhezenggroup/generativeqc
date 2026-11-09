"""Execute the actual host packer, cache identity and resident-ERI resource policy."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from _direct_schedule_test_support import write_direct_policy_headers

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_resolved_admission_controls_real_topology_and_cache_identity(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """No device is initialized; synthetic resource facts are legality evidence only."""
    write_direct_policy_headers(tmp_path)
    source = tmp_path / "topology_policy.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <limits>
#include <string>
#include "molecule/basis.hpp"
#include "scf/cuda/arena.hpp"
#include "scf/cuda/topology.hpp"

int main() {
  using namespace generativeqc;
  using namespace scf::cuda_execution;
  using namespace scf::cuda_policy;
  core::System system;
  system.atoms = {{2, {0., 0., 0.}}};
  system.shells = {{0, 0, {{1., 1.}}}, {0, 0, {{0.5, 1.}}}, {0, 2, {{0.8, 1.}}}};
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  std::string detail;
  assert(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS);
  const std::uint64_t eri_bytes = 7U * 7U * 7U * 7U * sizeof(double);
  runtime::CudaTargetInfo ample;
  ample.total_global_memory = std::size_t{32} << 30;
  runtime::CudaTargetInfo constrained = ample;
  constrained.total_global_memory = eri_bytes * 32U - 1U;
  for (bool unrestricted : {false, true}) {
    const auto pack = [&](const runtime::CudaTargetInfo& target,
                          SmallHfProfitabilityProfile profile, bool force_transform = false) {
      HostBatch host;
      assert(pack_host_batch({system}, {nullptr}, host, unrestricted, false, force_transform,
                             ResidentPsssPolicy::Skip, HostBasisPacking::Scf, target, profile));
      assert(host.nbf == 7 && host.direct_nbf == 8);
      return host;
    };
    const auto baseline = pack(ample, {});
    const auto unknown = pack({}, {});
    assert(baseline.persistent_eri && baseline.ao_to_direct_transform.empty());
    assert(same_topology(baseline, unknown));
    auto profile = SmallHfProfitabilityProfile{};
    profile.fallback_persistent_eri_ao_limit = 6;
    const auto direct = pack(ample, profile);
    assert(!direct.persistent_eri && direct.ao_to_direct_transform.size() == 7 * 8);
    assert(!same_topology(baseline, direct));
    const auto layout_for = [&](const HostBatch& host) {
      ArenaLayout layout{};
      assert(make_layout(1, 7, 8, 1, 3, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         3, 2, 0, unrestricted ? 2 : 1, host.persistent_eri,
                         !host.ao_to_direct_transform.empty(), false, false, false,
                         false, false, false, layout));
      return layout;
    };
    const auto resident_layout = layout_for(baseline);
    const auto direct_layout = layout_for(direct);
    assert(resident_layout.bytes > eri_bytes);
    assert(resident_layout.schwarz_bounds - resident_layout.eri == eri_bytes);
    assert(direct_layout.schwarz_bounds == direct_layout.eri);
    profile = {};
    profile.maximum_persistent_eri_bytes = eri_bytes;
    assert(pack(ample, profile).persistent_eri);
    --profile.maximum_persistent_eri_bytes;
    assert(!pack(ample, profile).persistent_eri);
    assert(!pack(constrained, {}).persistent_eri);
    constrained.total_global_memory = eri_bytes * 32U;
    assert(pack(constrained, {}).persistent_eri);
    constrained.total_global_memory -= 1U;
    auto transformed = pack(ample, {}, true);
    auto different_admission = transformed;
    different_admission.persistent_eri = false;
    assert(!same_topology(transformed, different_admission));
    const SmallHfWorkload batch{7, 4, 2, unrestricted ? 4U : 2U};
    profile.maximum_persistent_eri_bytes = 2U * eri_bytes;
    assert(resolve_small_hf_profitability(ample, batch, profile).persistent_eri);
    --profile.maximum_persistent_eri_bytes;
    assert(!resolve_small_hf_profitability(ample, batch, profile).persistent_eri);
  }
  const auto maximum = std::numeric_limits<std::size_t>::max();
  SmallHfProfitabilityProfile unbounded;
  unbounded.fallback_persistent_eri_ao_limit = maximum;
  unbounded.maximum_persistent_eri_bytes = maximum;
  assert(!resolve_small_hf_profitability(ample, {maximum, 1, 1, 1}, unbounded).persistent_eri);
  assert(!resolve_small_hf_profitability(ample, {7, 1, 0, 1}).persistent_eri);
  assert(!resolve_small_hf_profitability(ample, {7, 1, 1, 0}).persistent_eri);
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "topology_policy"
    required_native_cxx.build_executable(
        [
            source,
            ROOT / "src/scf/cuda/topology.cpp",
            ROOT / "src/scf/cuda/arena.cpp",
            ROOT / "src/scf/cuda/rhf_policy.cpp",
            ROOT / "src/runtime/cuda_provider.cpp",
            ROOT / "src/molecule/basis.cpp",
        ],
        executable,
        compile_args=(
            "-std=c++20",
            "-UNDEBUG",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
    )
    subprocess.run([str(executable)], check=True, timeout=30)
