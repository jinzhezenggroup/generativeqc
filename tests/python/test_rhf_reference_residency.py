"""Exercise optional CUDA RHF ERI residency, exact admission and device fallback."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc import _native

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RCCSD_CUDA_TEST") != "1",
    reason="requires an explicitly allocated CUDA device",
)


@pytest.fixture(scope="module")
def reference_residency_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile a native owner probe; the library supplies all numerical work."""
    compiler = shutil.which("c++")
    cache = shutil.which("ccache")
    nvcc = shutil.which("nvcc")
    if compiler is None or cache is None or nvcc is None:
        pytest.skip("requires C++, ccache and CUDA headers")
    directory = tmp_path_factory.mktemp("rhf-reference-residency")
    source, executable = directory / "probe.cpp", directory / "probe"
    source.write_text(CPP)
    library = Path(_native.load_library(device="cuda")._name).resolve()
    cuda = Path(os.environ.get("CUDA_PATH", str(Path(nvcc).resolve().parents[1])))
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-I" + str(cuda / "include"),
            str(source),
            str(library),
            "-Wl,-rpath," + str(library.parent),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    return executable


@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
def test_reference_residency_preserves_physical_frame_and_fallback(
    reference_residency_probe: Path, tmp_path: Path, representation: str
) -> None:
    """Compare complete references, then reject the optional cache two ways."""
    fixture = json.loads(
        (ROOT / "tests/reference_data/cc/gradients/h2o.json").read_text()
    )["inputs"]
    atoms = fixture["atomic_numbers"]
    coords = fixture["coordinates"]
    shells = fixture["shells"]
    lines = [f"{len(atoms)} {len(shells)} {int(representation == 'spherical')}"]
    lines += [
        " ".join(map(str, [z, *xyz])) for z, xyz in zip(atoms, coords, strict=True)
    ]
    for shell in shells:
        lines.append(
            f"{shell['atom_index']} {shell['angular_momentum']} {len(shell['primitives'])}"
        )
        lines += [" ".join(map(str, primitive)) for primitive in shell["primitives"]]
    path = tmp_path / "system.txt"
    path.write_text("\n".join(lines) + "\n")
    result = subprocess.run(
        [str(reference_residency_probe), str(path)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    (tmp_path / "trace.json").write_text(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(result.stdout)
    assert record["resident_bytes"] == 7**4 * 8
    assert record["tight_peak"] == record["minimum_budget"]
    assert record["allocation_rejections"] == 1
    assert record["live_after_release"] == 0


CPP = r"""
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include "molecule/basis.hpp"
#include "runtime/resource_ledger.hpp"
#include "scf/cuda/rhf_bucket_internal.hpp"
#include "scf/mean_field.hpp"

using namespace generativeqc;
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void matrix(const std::vector<double>& a, const std::vector<double>& b) {
  require(a.size()==b.size(), "physical frame shape");
  for(std::size_t i=0;i<a.size();++i)
    require(std::abs(a[i]-b[i])<2e-9, "physical frame differs from independent CPU solve");
}
struct Owner {
  scf::CudaRhfBucketPlan* plan=nullptr;
  ~Owner() { scf::destroy_rhf_cuda_bucket_plan(plan); }
};
scf::ScfResult solve(Owner& owner, const core::System& system, const scf::ScfOptions& options) {
  auto rows=scf::run_rhf_cuda_bucket_cached(&owner.plan,{system},options,{nullptr},0);
  require(rows.size()==1 && rows[0].status==GENERATIVEQC_STATUS_SUCCESS,
          "CUDA reference endpoint failed");
  require(rows[0].scf.converged && rows[0].scf.reference, "missing physical reference");
  require(rows[0].scf.reference->numeric_capacity_bytes<=options.reference_memory_budget_bytes,
          "reference exceeded complete budget");
  return rows[0].scf;
}
void compare(const scf::ScfResult& actual,const scf::ScfResult& expected) {
  require(std::abs(actual.energy-expected.energy)<2e-10, "reference energy");
  matrix(actual.reference->density,expected.reference->density);
  matrix(actual.reference->fock,expected.reference->fock);
  matrix(actual.reference->orbital_energies,expected.reference->orbital_energies);
}
int main(int argc,char** argv) {
 try {
  require(argc==2,"input");std::ifstream in(argv[1]);std::size_t atoms=0,shells=0;int spherical=0;
  in>>atoms>>shells>>spherical;core::System system;
  system.atoms.resize(atoms);system.shells.resize(shells);
  for(auto& a:system.atoms) in>>a.atomic_number>>a.position[0]>>a.position[1]>>a.position[2];
  for(auto& s:system.shells) {
    std::size_t count=0;in>>s.atom_index>>s.angular_momentum>>count;s.primitives.resize(count);
    for(auto& p:s.primitives) in>>p.exponent>>p.coefficient;
  }
  require(bool(in),"fixture parse");
  system.basis_representation=spherical?GENERATIVEQC_BASIS_SPHERICAL:GENERATIVEQC_BASIS_CARTESIAN;
  std::string detail;require(molecule::validate_and_normalize(system,detail)==GENERATIVEQC_STATUS_SUCCESS,
                              "fixture normalization");
  scf::ScfOptions options;options.compute_forces=false;options.export_physical_reference=true;
  options.screening_tolerance=0;options.max_iterations=200;
  options.energy_tolerance=1e-13;options.density_tolerance=1e-11;
  options.reference_memory_budget_bytes=512ULL<<20;
  const auto expected=scf::run_rhf(system,options);
  require(expected.converged && expected.reference,"independent CPU reference");
  std::size_t resident=0,minimum=0,owned=0;
  {
    Owner roomy;const auto result=solve(roomy,system,options);compare(result,expected);
    resident=roomy.plan->resources.reference_eri_bytes_;
    require(resident==7*7*7*7*sizeof(double),"roomy reference did not retain ERIs");
    minimum=result.reference->numeric_capacity_bytes-resident;
    owned=scf::hf_cuda_owned_device_bytes(roomy.plan)-resident;
  }
  for (std::size_t budget : {minimum+resident-1, minimum+resident}) {
    Owner boundary;options.reference_memory_budget_bytes=budget;
    compare(solve(boundary,system,options),expected);
    require(boundary.plan->resources.reference_eri_bytes_==
            (budget==minimum+resident ? resident : 0),"optional-cache exact admission");
  }
  std::size_t tight_peak=0;
  {
    Owner tight;options.reference_memory_budget_bytes=minimum;
    const auto result=solve(tight,system,options);compare(result,expected);
    require(tight.plan->resources.reference_eri_==nullptr,"tight budget did not fall back");
    require(scf::hf_cuda_owned_device_bytes(tight.plan)==owned,"resident storage accounting");
    tight_peak=result.reference->numeric_capacity_bytes;
  }
  {
    Owner rejected;options.reference_memory_budget_bytes=minimum-1;bool refused=false;
    try { (void)solve(rejected,system,options); }
    catch(const std::length_error&) { refused=true; }
    require(refused,"minimum-minus-one was admitted");
  }
  // Inject device pressure through the real numeric ledger, without exhausting
  // a shared GPU or overriding scheduler-provided visibility.
  auto ledger=std::make_shared<runtime::DeviceResourceLedger>();
  ledger->device=0;ledger->limit=owned;ledger->active=true;
  runtime::active_device_resource_ledger=ledger;
  {
    Owner pressure;options.reference_memory_budget_bytes=512ULL<<20;
    const auto result=solve(pressure,system,options);compare(result,expected);
    require(pressure.plan->resources.reference_eri_==nullptr,"allocation failure did not fall back");
    require(ledger->rejected==1,"optional allocation was not rejected exactly once");
  }
  runtime::active_device_resource_ledger.reset();
  require(ledger->live==0,"device allocations leaked after reference release");
  // Changed coordinates must rebuild integrals; a second solve also exercises
  // teardown/recreation after an optional allocation failure.
  system.atoms[1].position[2]+=0.01;
  require(molecule::validate_and_normalize(system,detail)==GENERATIVEQC_STATUS_SUCCESS,"changed geometry");
  Owner changed;compare(solve(changed,system,options),scf::run_rhf(system,options));
  // Higher angular momentum keeps the same independent matrix-direct fallback.
  core::System helium;helium.atoms={{2,{0.,0.,0.}}};
  helium.shells={{0,0,{{1.5,1.0}}},{0,2,{{0.8,1.0}}}};
  helium.basis_representation=system.basis_representation;
  require(molecule::validate_and_normalize(helium,detail)==GENERATIVEQC_STATUS_SUCCESS,"d-shell fixture");
  Owner angular;compare(solve(angular,helium,options),scf::run_rhf(helium,options));
  require(angular.plan->resources.reference_eri_bytes_==0,"unqualified d-shell cache selected");
  std::cout<<"{\"resident_bytes\":"<<resident<<",\"minimum_budget\":"<<minimum
           <<",\"tight_peak\":"<<tight_peak<<",\"allocation_rejections\":"<<ledger->rejected
           <<",\"live_after_release\":"<<ledger->live<<"}\n";
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
"""
