"""Execute source admission and count its real host-packing allocation peak."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PRELUDE = r"""
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <vector>
#include "cc/df_source.hpp"
#include "runtime/bounded_workspace.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/cuda/df_source_domain.hpp"
#include "scf/df_source_capacity.hpp"
#include <generated_df_cc_source_cpu.hpp>
#include <df_mo_source_generated.hpp>

namespace allocation_probe {
bool enabled = false;
std::size_t live = 0, peak = 0;
struct alignas(std::max_align_t) Header { std::size_t bytes; bool tracked; };
void* allocate(std::size_t n) {
  auto* p = static_cast<Header*>(std::malloc(sizeof(Header) + n));
  if (!p) throw std::bad_alloc();
  *p = {n, enabled};
  if (enabled) { live += n; peak = std::max(peak, live); }
  return p + 1;
}
void release(void* value) noexcept {
  if (!value) return;
  auto* p = static_cast<Header*>(value) - 1;
  if (p->tracked) live -= p->bytes;
  std::free(p);
}
}
void* operator new(std::size_t n) { return allocation_probe::allocate(n); }
void* operator new[](std::size_t n) { return allocation_probe::allocate(n); }
void operator delete(void* p) noexcept { allocation_probe::release(p); }
void operator delete[](void* p) noexcept { allocation_probe::release(p); }
void operator delete(void* p, std::size_t) noexcept { allocation_probe::release(p); }
void operator delete[](void* p, std::size_t) noexcept { allocation_probe::release(p); }

int device_entries = 0, factory_entries = 0;
namespace generativeqc {
namespace runtime {
struct CudaDeviceScope { explicit CudaDeviceScope(int) { ++device_entries; } };
}
namespace scf {
__SOURCE_METADATA_CAPACITY__
struct CudaDensityFittingIntegralSource {};
namespace cuda_execution {
// The host-only boundary has no CUDA implementation or ABI claim. Its size is
// passed to the real typed capacity planner, just as the real owner size is.
struct CudaDensityFittingIntegralSourceImpl { std::byte storage[1024]; };
__DF_PUBLIC_EXPANSION__
__DEVICE_OBSERVER__
"""

FACTORY_END = r"""
  // Metadata-only packing must not even reserve the omitted SCF payloads.
  if (host.shell_pair_systems.capacity() || host.shell_pair_first.capacity() ||
      host.shell_pair_second.capacity() || host.psss_resident_tasks.capacity() ||
      host.psss_resident_ket_pairs.capacity() || host.warm_density.capacity() ||
      host.occupied.capacity() || host.warm_mask.capacity())
    throw std::logic_error("DF metadata packing retained SCF payload capacity");
  const auto capacity = df_source_capacity::plan(
      orbital_systems.front(), auxiliary_systems.front(),
      {sizeof(CudaDensityFittingIntegralSourceImpl), sizeof(HostBatch), sizeof(DfPublicAoExpansion)});
  // CUDA is replaced only at the allocation boundary. Charging its complete
  // typed payload to this counted host buffer includes both sides in the peak.
  const auto device_bytes = observed_device_bytes(host, public_nbf, public_naux, metric_elements);
  if (device_bytes != capacity.device_bytes)
    throw std::logic_error("source uploads disagree with the construction device ledger");
  // The public shape-only ABI must also cover the actual uploaded record
  // inventory. The generated metric is temporary, so exclude it here.
  const auto shape_capacity = density_fitting_source_metadata_bytes(
      batch_size, host.atomic_numbers.size(), host.shell_atoms.size(),
      host.ao_shells.size(), host.primitive_exponents.size(),
      batch_size * (public_nbf * cartesian_nbf + public_naux * cartesian_naux));
  if (device_bytes - metric_elements * sizeof(double) > shape_capacity)
    throw std::logic_error("shape-only source capacity underestimates owned uploads");
  std::vector<std::byte> device_payload(device_bytes);
  auto candidate = std::make_unique<CudaDensityFittingIntegralSourceImpl>();
  auto atom_prefix = host.atom_offsets;
  std::vector<void*> upload_owners; upload_owners.reserve(20);
  std::vector<molecule::BasisGeometryIdentity> orbital_ids, auxiliary_ids;
  orbital_ids.reserve(1); auxiliary_ids.reserve(1);
  orbital_ids.emplace_back(orbital_systems.front());
  auxiliary_ids.emplace_back(auxiliary_systems.front());
  std::vector<DfPublicAoExpansion> orbital_transform(public_nbf), auxiliary_transform(public_naux);
  const auto orbital_item = make_public_to_cartesian_transform(orbital_systems.front());
  const auto auxiliary_item = make_public_to_cartesian_transform(auxiliary_systems.front());
  metrics.resize(metric_total_elements);
  nbf = public_nbf; naux = public_naux;
  (void)candidate; (void)atom_prefix; (void)orbital_item; (void)auxiliary_item;
  return GENERATIVEQC_STATUS_SUCCESS;
}
}  // namespace cuda_execution
generativeqc_status create_cuda_density_fitting_integral_source(
    int device, const std::vector<core::System>& orbital, const std::vector<core::System>& auxiliary,
    CudaDensityFittingIntegralSource**, std::vector<double>& metric,
    std::size_t& n, std::size_t& q, std::string& detail,
    const cuda_execution::CudaDfSourcePolicy* policy) {
  assert(policy && policy->value_math == 0 && policy->requested_value_mapping == 1);
  ++factory_entries;
  cuda_execution::CudaDensityFittingIntegralSourceImpl* source = nullptr;
  return cuda_execution::create_cuda_density_fitting_integral_source_impl(
      device, orbital, auxiliary, &source, metric, n, q, detail, *policy);
}
}  // namespace scf
namespace cc {
"""

MAIN = r"""
} // namespace cc
} // namespace generativeqc
using namespace generativeqc;

core::System system(unsigned shells, unsigned angular=0, unsigned primitives=1,
                    bool spherical=false) {
  core::System s;
  s.atoms = {{1,{0,0,0},0}, {1,{0,0,1},0}};
  s.electron_count = 2;
  s.basis_representation = spherical ? GENERATIVEQC_BASIS_SPHERICAL : GENERATIVEQC_BASIS_CARTESIAN;
  for (unsigned i=0; i<shells; ++i) {
    core::Shell shell{i%2, angular, {}};
    for (unsigned j=0; j<primitives; ++j) shell.primitives.push_back({1.0+j, .5});
    s.shells.push_back(std::move(shell));
  }
  return s;
}
hf::PhysicalReference reference(const core::System& s) {
  hf::PhysicalReference r; r.nbf=molecule::ao_count(s); r.nocc=1;
  for (auto* v : {&r.overlap,&r.hcore,&r.fock,&r.coefficients,&r.density,&r.weighted_density})
    v->resize(r.nbf*r.nbf, 0.0);
  r.orbital_energies.resize(r.nbf);
  return r;
}
auto capacity(const core::System& o, const core::System& a) {
  return scf::df_source_capacity::plan(o,a,
      {sizeof(scf::cuda_execution::CudaDensityFittingIntegralSourceImpl),
       sizeof(scf::cuda_execution::HostBatch), sizeof(scf::cuda_execution::DfPublicAoExpansion)});
}
std::size_t external(const core::System& o, const core::System& a, const hf::PhysicalReference& r) {
  auto b=posthf::source_capacity(o)+posthf::source_capacity(a);
  for (const auto* v : {&r.overlap,&r.hcore,&r.fock,&r.coefficients,&r.orbital_energies,
                        &r.density,&r.weighted_density}) b+=v->capacity()*sizeof(double);
  return b;
}
int main(int argc, char** argv) {
  if (argc!=2) return 1;
  const int mode=std::atoi(argv[1]);
  scf::cuda_execution::CudaDfSourcePolicy policy;
  std::string detail;
  setenv("GENERATIVEQC_DF_VALUE_MATH", "generic", 1);
  setenv("GENERATIVEQC_DF_VALUE_MAPPING", "component", 1);
  assert(scf::cuda_execution::resolve_cuda_df_source_policy(policy, detail));
  // Source packing must consume the admitted snapshot, even after an environment change.
  setenv("GENERATIVEQC_DF_VALUE_MATH", "invalid-after-admission", 1);
  setenv("GENERATIVEQC_DF_VALUE_MAPPING", "primitive", 1);
  auto orbital=system(2), auxiliary=system(100);
  auto ref=reference(orbital);
  auto p=capacity(orbital,auxiliary);
  const auto admitted=external(orbital,auxiliary,ref)+p.numeric_bytes;
  if (mode<3) {
    const auto layout=cc::generated::df_source::source_layout(1,1,100);
    const auto legacy_budget=external(orbital,auxiliary,ref)+
        std::max({layout.transform_bytes,layout.packing_bytes,layout.blocks_bytes});
    const auto budget=mode==0 ? legacy_budget : admitted-(mode==1);
    assert(external(orbital,auxiliary,ref)==15408 && layout.transform_bytes==9632);
    // The original unprojected fixture admitted 24,212 bytes. Physical pair
    // projection now adds exactly two n*n*q FP64 arrays (6,400 bytes here).
    // Both payload-only budgets remain below the metric allocation alone.
    const auto pair_projection_bytes=2*layout.source_values*sizeof(double);
    assert(pair_projection_bytes==6400 && legacy_budget==24212+pair_projection_bytes);
    assert(100*100*sizeof(double)>legacy_budget);
    try {
      allocation_probe::enabled=true;
      (void)cc::build_df_source_cuda(orbital,auxiliary,ref,budget,1e-10,0,0,false,&policy);
      allocation_probe::enabled=false;
      if (mode!=2 || factory_entries!=1 || device_entries!=1) return 2;
      if (allocation_probe::peak>p.numeric_bytes || allocation_probe::live) return 3;
    } catch (const std::length_error&) {
      allocation_probe::enabled=false;
      if (mode==2 || factory_entries || device_entries) return 4;
    }
  } else if (mode==3) {
    auto a=scf::df_source_capacity::shape(auxiliary);
    a.public_aos=static_cast<std::size_t>(INT64_MAX);
    try {
      (void)scf::df_source_capacity::plan(scf::df_source_capacity::shape(orbital),a,{1024,1024,64});
      return 5;
    } catch (const std::overflow_error&) { if (factory_entries || device_entries) return 6; }
  } else if (mode==4) {
    // Actual source-prefix copies and topology packing, including overlapping
    // reallocations, are counted. No final-capacity-only approximation is used.
    for (unsigned l=0; l<=3; ++l) for (unsigned primitives : {1U, 37U}) {
      auto o=system(2,l,primitives,l>=2), a=system(13,l,primitives,l>=2);
      auto r=reference(o); auto bound=capacity(o,a);
      allocation_probe::peak=0;
      allocation_probe::enabled=true;
      (void)cc::build_df_source_cuda(o,a,r,1ULL<<30,1e-10,0,0,false,&policy);
      allocation_probe::enabled=false;
      if (allocation_probe::peak>bound.numeric_bytes || allocation_probe::live) return 7;
    }
    // Later g expansion scratch is bounded without admitting g production here.
    allocation_probe::peak=0;
    allocation_probe::enabled=true;
    { const auto expanded=molecule::ao_expansions(4,GENERATIVEQC_BASIS_SPHERICAL); }
    allocation_probe::enabled=false;
    constexpr auto maximum=2*15*sizeof(molecule::CartesianComponent)+15*sizeof(double)+
        3*15*(sizeof(molecule::AoExpansion)+15*sizeof(molecule::CartesianExpansionTerm));
    if (allocation_probe::peak>maximum || allocation_probe::live) return 8;
  } else if (mode==5) {
    allocation_probe::enabled=true;
    (void)capacity(orbital,auxiliary);
    allocation_probe::enabled=false;
    if (allocation_probe::peak || allocation_probe::live) return 9;
  } else if (mode==6) {
    // Auxiliary-g metadata uses the physical six-term public transform and
    // still skips all SCF pair/warm payloads before the reservation branch.
    static_assert(scf::cuda_execution::kDfPublicAoExpansionTerms == 6);
    for (bool spherical : {false, true}) for (unsigned primitives : {1U, 37U}) {
      auto o=system(2,3,primitives,spherical), a=system(2,4,primitives,spherical);
      auto r=reference(o); auto bound=capacity(o,a);
      allocation_probe::peak=0;
      allocation_probe::enabled=true;
      (void)cc::build_df_source_cuda(o,a,r,1ULL<<30,1e-10,0,0,false,&policy);
      allocation_probe::enabled=false;
      if (allocation_probe::peak>bound.numeric_bytes || allocation_probe::live) return 11;
    }
  } else if (mode==7) {
    // Small sources take the sparse-record branch even with only s/p shells;
    // also cover spherical d/f and auxiliary g against the same owned ledger.
    for (unsigned l=0; l<=4; ++l) for (bool spherical : {false, true}) {
      auto o=system(2,std::min(l,3U),1,spherical), a=system(2,l,1,spherical);
      auto r=reference(o);
      (void)cc::build_df_source_cuda(o,a,r,1ULL<<30,1e-10,0,0,false,&policy);
    }
    try {
      (void)scf::density_fitting_source_metadata_bytes(
          1,0,0,std::numeric_limits<std::size_t>::max()/64,0,0);
      return 12;
    } catch (const std::overflow_error&) {}
    try {
      (void)scf::density_fitting_source_metadata_bytes(
          1,0,0,0,0,std::numeric_limits<std::size_t>::max()/8+1);
      return 13;
    } catch (const std::overflow_error&) {}
  } else return 10;
  return 0;
}
"""


@pytest.fixture(scope="module")
def admission_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("df-source-admission")
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    for generator, arguments in (
        ("generate_df_cc_source.py", ["--output-dir", str(directory)]),
        (
            "generate_one_electron_kernels.py",
            [
                "--derivatives",
                "--derivative-policy-output",
                str(directory / "generated_one_electron_derivative_policy.cuh"),
            ],
        ),
        (
            "generate_df_mo_source.py",
            ["--output", str(directory / "df_mo_source_generated.hpp")],
        ),
        (
            "generate_direct_resident_schedule.py",
            [
                "--output",
                str(directory / "generated_direct_resident_psss_schedule.cuh"),
            ],
        ),
    ):
        subprocess.run(
            [sys.executable, "-S", str(ROOT / "tools" / generator), *arguments],
            check=True,
            capture_output=True,
            timeout=60,
        )
    setup = (ROOT / "src/scf/cuda/df_source_setup.cpp").read_text()
    transform = setup[
        setup.index(
            "std::vector<DfPublicAoExpansion> make_public_to_cartesian_transform("
        ) : setup.index(
            "}  // namespace",
            setup.index(
                "std::vector<DfPublicAoExpansion> make_public_to_cartesian_transform("
            ),
        )
    ]
    start = setup.index(
        "generativeqc_status create_cuda_density_fitting_integral_source_impl("
    )
    factory = setup[start : setup.index("  cudaError_t cuda_error =", start)]
    source = (ROOT / "src/cc/df_source_cuda.cu").read_text()
    helpers = source[
        source.index("using posthf::checked_add;") : source.index("struct SourceDelete")
    ]
    start = source.index("DFSourceResult build_df_source_cuda(")
    prefix = source[
        start : source.index(
            "  std::unique_ptr<scf::CudaDensityFittingIntegralSource", start
        )
    ]
    kernels = (ROOT / "src/scf/cuda/df_source_kernels.hpp").read_text()
    start = kernels.index("struct DfPublicAoExpansion {")
    expansion = kernels[start : kernels.index("\n};", start) + 4]
    start = kernels.index("inline constexpr std::size_t kDfPublicAoExpansionTerms")
    constants = kernels[start : kernels.index(";", start) + 1]
    uploads = re.findall(
        r"GENERATIVEQC_UPLOAD_SOURCE_FIELD\(\w+,\s*host\.(\w+)\);", setup
    )
    assert uploads and len(uploads) == len(set(uploads))
    observer = (
        "std::size_t observed_device_bytes(const HostBatch& h, std::size_t n, "
        "std::size_t q, std::size_t metric) {\n  std::size_t bytes=0;\n"
        + "".join(
            f"  bytes += h.{field}.size()*sizeof(h.{field}[0]);\n" for field in uploads
        )
        + "  return bytes+(n+q)*sizeof(DfPublicAoExpansion)+metric*sizeof(double);\n}\n"
    )
    prelude = PRELUDE.replace("__DF_PUBLIC_EXPANSION__", constants + "\n" + expansion)
    planner = (ROOT / "src/scf/density_fitting.cpp").read_text()
    start = planner.index("bool checked_multiply(")
    multiply = planner[start : planner.index("\n}", start) + 2]
    start = planner.index("std::size_t density_fitting_source_metadata_bytes(")
    metadata = planner[start : planner.index("\n}", start) + 2]
    prelude = prelude.replace(
        "__SOURCE_METADATA_CAPACITY__",
        "namespace {\n" + multiply + "\n}\n" + metadata,
    )
    prelude = prelude.replace("__DEVICE_OBSERVER__", observer)
    program = prelude + transform + factory + FACTORY_END + helpers + prefix
    program += (
        "check_status(source_status, detail);\n"
        "(void)started; (void)stage; (void)work; return result;\n}\n" + MAIN
    )
    path = directory / "admission.cpp"
    path.write_text(program)
    objects = []
    for i, cpp in enumerate(
        (
            path,
            ROOT / "src/molecule/basis.cpp",
            ROOT / "src/scf/cuda/topology.cpp",
            ROOT / "src/scf/cuda/df_source_domain.cpp",
            ROOT / "src/scf/cuda/rhf_policy.cpp",
        )
    ):
        obj = directory / f"part-{i}.o"
        compiled = subprocess.run(
            [
                cache,
                compiler,
                "-std=c++20",
                "-O1",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-ffunction-sections",
                "-fdata-sections",
                "-I" + str(ROOT / "src"),
                "-I" + str(ROOT / "include"),
                "-isystem",
                str(directory),
                "-c",
                str(cpp),
                "-o",
                str(obj),
            ],
            env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
            check=False,
            capture_output=True,
            timeout=90,
        )
        assert compiled.returncode == 0, compiled.stderr.decode()
        objects.append(str(obj))
    executable = directory / "admission"
    subprocess.run(
        [compiler, *objects, "-Wl,--gc-sections", "-o", str(executable)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize("mode", range(8))
def test_source_setup_admission_and_allocation_peak(
    admission_probe: Path, mode: int
) -> None:
    subprocess.run([str(admission_probe), str(mode)], check=True, timeout=30)
