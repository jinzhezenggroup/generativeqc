"""Shared, evidence-bounded admission for native XC matrix lowerings."""


def emit_indexed_fp64_profile() -> str:
    """Emit one guarded adapter for density and potential in the same CUDA unit.

    Capability probes occur only during admitted runtime preparation. Generation
    stays device independent; complete cost/resource ranking remains generic.
    """
    return r"""
#ifndef GENERATIVEQC_XC_INDEXED_FP64_PROFILE
#define GENERATIVEQC_XC_INDEXED_FP64_PROFILE
#include "runtime/cuda_device_facts.hpp"
#include "runtime/cuda_lowering_profile.hpp"
namespace generativeqc::dft::cuda_xc_detail {
inline std::string_view indexed_fp64_profile_rejection(
    const CudaXcLayout& layout, std::size_t budget) {
  constexpr runtime::CudaLoweringProfile profile{true,{512,1U<<19,4,1}};
  const std::array<std::size_t,4> extents{
      layout.nao,layout.npoint,layout.work_jets,layout.tile_points};
  auto rejection=runtime::cuda_lowering_workload_rejection(profile,extents);
  if (!budget || !layout.local_ao || !layout.ao_map_entries || layout.response)
    rejection="indexed physical domain or optional budget is unavailable";
  if (layout.ao_precision!=CudaXcAoPrecision::Fp64 || layout.fp32_contractions)
    rejection="optional XC profile requires strict FP64 arithmetic";
  if (!rejection.empty()) return rejection;
  int device{}; runtime::CudaTargetInfo target{}; char name[256]{};
  generativeqc_tensor::cuda_check(cudaGetDevice(&device));
  const auto status=runtime::cuda_device_facts(device,target,name);
  if (status==cudaErrorNotSupported || status==cudaErrorInvalidValue) {
    const auto pending=cudaGetLastError();
    if (pending!=cudaSuccess && pending!=status) generativeqc_tensor::cuda_check(pending);
    return "optional lowering device facts are unavailable";
  }
  generativeqc_tensor::cuda_check(status);
  return runtime::cuda_lowering_profile_rejection(profile,target,extents);
}
}
#endif
"""
