"""Exercise the real native descriptor gate without allocating a CUDA owner."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from tools.generate_xc_split_hybrid_registry import emit_registry

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_native_split_hybrid_descriptor_gate(
    tmp_path: Path, required_native_cxx: object
) -> None:
    (tmp_path / "generated_split_hybrid_registry.cuh").write_text(
        emit_registry(), encoding="utf-8"
    )
    executable = tmp_path / "split-hybrid-admission"
    required_native_cxx.build_executable(
        [ROOT / "tests/native/split_hybrid_admission_probe.cpp"],
        executable,
        compile_args=(
            "-std=c++20",
            "-DGENERATIVEQC_HAS_CUDA=1",
            "-I" + str(ROOT / "include"),
            "-I" + str(ROOT / "src"),
            "-I" + str(tmp_path),
        ),
        compile_timeout=120,
    )
    subprocess.run([str(executable)], check=True, timeout=30)


def test_single_and_batch_share_descriptor_admission(
    tmp_path: Path, required_native_cxx: "NativeCxx"
) -> None:
    """Execute both factory entries and reject before entering a provider."""
    source = (ROOT / "src/methods/registry.cpp").read_text(encoding="utf-8")
    declarations = source[
        source.index("using ValidateSystem =") : source.index(
            "constexpr MethodDefinition"
        )
    ]

    def definition(signature: str) -> str:
        start = source.index(signature)
        opening = source.index("{", start)
        depth = 1
        end = opening + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        return source[start:end]

    (tmp_path / "generated_split_hybrid_registry.cuh").write_text(
        emit_registry(), encoding="utf-8"
    )
    unit = tmp_path / "registry_admission.cpp"
    unit.write_text(
        r"""
#include <cassert>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include "dft/application_precision.hpp"
#include "methods/application_precision.hpp"
#include "methods/dft_admission.hpp"
#include "methods/method.hpp"
#include "runtime/provider_registry.hpp"
#include "scf/initial_guess/automatic_policy.hpp"
namespace generativeqc::methods {
"""
        + declarations
        + r"""
bool provider_entered=false;
std::unique_ptr<PreparedCalculation> admitted_single(const Capabilities&,core::ContextState&,
 const core::System&,const generativeqc_method_descriptor&,const core::System*) {
 provider_entered=true;return {};
}
std::unique_ptr<PreparedBatch> admitted_batch(const Capabilities&,core::ContextState&,
 std::vector<core::System>,const generativeqc_method_descriptor&,generativeqc_batch_flags,
 const core::System*) {provider_entered=true;return {};}
const MethodDefinition& require_available(generativeqc_method method) {
 static MethodDefinition value;
 value.provider.domain.method=method;
 value.provider.domain.family=GENERATIVEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL;
 value.provider.domain.supports_batch=true;
 value.prepare_calculation=admitted_single;value.prepare_batch=admitted_batch;
 return value;
}
void validate_system(const MethodDefinition&,const core::System&) {}
void validate_preliminary_system(const generativeqc_method_descriptor&,const core::System&) {}
namespace detail {
std::unique_ptr<PreparedCalculation> prepare_composed_dft_calculation(const Capabilities&,
 core::ContextState&,const core::System&,const generativeqc_method_descriptor&,
 const core::System*,const named_ks::Dispersion&,std::uint64_t) {std::abort();}
std::unique_ptr<PreparedBatch> prepare_composed_dft_batch(const Capabilities&,core::ContextState&,
 std::vector<core::System>,const generativeqc_method_descriptor&,generativeqc_batch_flags,
 const core::System*,const named_ks::Dispersion&,std::uint64_t) {std::abort();}
}
"""
        + definition("void validate_option_family(")
        + "\n"
        + definition("std::unique_ptr<PreparedCalculation> prepare_calculation(")
        + "\n"
        + definition("std::unique_ptr<PreparedBatch> prepare_batch(")
        + r"""
}
int main() {
 using namespace generativeqc;
 for(bool batch:{false,true}) for(int scenario:{0,1,2,3,4}) {
  core::ContextState context;context.requested_backend=GENERATIVEQC_BACKEND_CUDA;
  core::System system;
  generativeqc_ks_semilocal_component components[]={{"HYB_MGGA_X_M06_2X",1.},{"MGGA_C_M06_2X",1.}};
  generativeqc_ks_exchange_term exchange{GENERATIVEQC_KS_EXCHANGE_FULL_RANGE,27./50.,0.,-27./100.};
  generativeqc_ks_options ks{};ks.struct_size=sizeof(ks);ks.abi_version=GENERATIVEQC_ABI_VERSION;
  ks.spin_channels=1;ks.semilocal_components=components;ks.semilocal_component_count=2;
  ks.exchange_terms=&exchange;ks.exchange_term_count=1;ks.xc_execution_schedule=GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED;
  generativeqc_method_descriptor descriptor{};descriptor.struct_size=sizeof(descriptor);
  descriptor.abi_version=GENERATIVEQC_ABI_VERSION;descriptor.method=GENERATIVEQC_METHOD_PBE_RKS;
  descriptor.precision_mode=GENERATIVEQC_PRECISION_FP64;descriptor.ks_options=&ks;
  generativeqc_status expected=GENERATIVEQC_STATUS_SUCCESS;
  if(scenario==1) {context.requested_backend=GENERATIVEQC_BACKEND_CPU_REFERENCE;expected=GENERATIVEQC_STATUS_NOT_IMPLEMENTED;}
  if(scenario==2) {exchange.coefficient=exchange.fock_coefficient=0.;expected=GENERATIVEQC_STATUS_NOT_IMPLEMENTED;}
  if(scenario==3) {ks.struct_size=0;expected=GENERATIVEQC_STATUS_ABI_MISMATCH;}
  if(scenario==4) {ks.exchange_terms=nullptr;expected=GENERATIVEQC_STATUS_INVALID_ARGUMENT;}
  methods::provider_entered=false;generativeqc_status actual=GENERATIVEQC_STATUS_SUCCESS;
  try {
   if(batch) (void)methods::prepare_batch(context,{system},descriptor,0,nullptr);
   else (void)methods::prepare_calculation(context,system,descriptor,nullptr);
  } catch(const methods::MethodError& error) {actual=error.status();}
  assert(actual==expected);
  assert(methods::provider_entered==(expected==GENERATIVEQC_STATUS_SUCCESS));
 }
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "registry_admission"
    required_native_cxx.build_executable(
        [unit],
        executable,
        compile_args=(
            "-std=c++20",
            "-DGENERATIVEQC_HAS_CUDA=1",
            "-I" + str(ROOT / "include"),
            "-I" + str(ROOT / "src"),
            "-I" + str(tmp_path),
        ),
    )
    subprocess.run([str(executable)], check=True, timeout=30)
