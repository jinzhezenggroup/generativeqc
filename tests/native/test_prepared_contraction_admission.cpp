#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

#include "cc/df_prepared_regions.hpp"
#include "tensor/cuda_matrix_product_admission.hpp"

struct Snapshot {
  bool df_auxiliary_reduction{}, df_matrix_gemm{}, df_audit_matrix_gemm{}, df_primal_matrix_gemm{},
      df_core_reuse{}, df_mixed_precision{};
  std::size_t df_auxiliary_batch_size{}, df_provider_allowance_bytes{};
};

int main() {
  generativeqc::cc::DFPreparedRegions regions;
  Snapshot snapshot;
  regions.report(snapshot);
  assert(!snapshot.df_matrix_gemm && snapshot.df_auxiliary_batch_size == 1);
  regions.reduced = regions.retained_core = true;
  regions.response = regions.audit = 16;
  regions.primal = 4;
  regions.provider_allowance = 96ULL << 20;
  regions.report(snapshot);
  assert(snapshot.df_matrix_gemm && snapshot.df_audit_matrix_gemm &&
         snapshot.df_primal_matrix_gemm && snapshot.df_core_reuse);
  snapshot = {};
  assert(regions.response == 16 && regions.audit == 16 && regions.primal == 4 &&
         regions.batch() == 16 && regions.retained_core);
  regions.audit.reset();
  regions.primal.reset();
  regions.report(snapshot);
  assert(snapshot.df_matrix_gemm && !snapshot.df_audit_matrix_gemm &&
         !snapshot.df_primal_matrix_gemm && snapshot.df_provider_allowance_bytes == 96ULL << 20);

  using generativeqc::tensor::CudaMatrixProductAdmission;
  using generativeqc::tensor::CudaMatrixProductProfile;
  generativeqc::runtime::CudaTargetInfo target;
  target.multiprocessor_count = 8;
  target.maximum_blocks_per_sm = 16;
  target.maximum_threads_per_sm = 2048;
  const auto native = CudaMatrixProductAdmission::resolve(target, 16, 1);
  const auto library = CudaMatrixProductAdmission::resolve(target, 17, 1);
  assert(!native.requires_provider_context() && !native.calibrated());
  assert(library.requires_provider_context() && !library.calibrated());
  assert(!library.available_if(false).requires_provider_context());
  assert(!native.available_if(true).requires_provider_context());
  assert(native.estimate().flops == 8192 && native.estimate().native_semantic_bytes == 67584 &&
         native.estimate().provider_semantic_bytes == 6144);
  assert(native.estimate().native_blocks == 8 && native.estimate().native_waves == 1);
  CudaMatrixProductProfile profile;
  profile.calibration = {100, 1, 1000, 1, 1000, 1000};
  const auto calibrated = CudaMatrixProductAdmission::resolve(target, 8, 1, profile);
  assert(calibrated.calibrated() && calibrated.requires_provider_context() &&
         calibrated.provider_nanoseconds() < calibrated.native_nanoseconds());
  profile.provider_setup_nanoseconds = 1e6;
  const auto cold = CudaMatrixProductAdmission::resolve(target, 8, 1, profile);
  assert(cold.calibrated() && !cold.requires_provider_context());
  profile.provider_setup_nanoseconds = std::numeric_limits<double>::infinity();
  const auto invalid_setup = CudaMatrixProductAdmission::resolve(target, 32, 1, profile);
  assert(!invalid_setup.calibrated() && invalid_setup.requires_provider_context());
  profile.provider_setup_nanoseconds = 0;
  const auto empty_batch = CudaMatrixProductAdmission::resolve(target, 32, 0, profile);
  assert(!empty_batch.requires_provider_context() && empty_batch.estimate().flops == 0);
  profile.calibration.native_fp64_flops_per_nanosecond = std::numeric_limits<double>::denorm_min();
  const auto overflowed = CudaMatrixProductAdmission::resolve(target, 32, 1, profile);
  assert(!overflowed.calibrated() && overflowed.requires_provider_context() &&
         overflowed.native_nanoseconds() == 0 && overflowed.provider_nanoseconds() == 0);
  profile.calibration.native_fp64_flops_per_nanosecond = 1;
  profile.calibration.cublas_launch_nanoseconds = std::numeric_limits<double>::quiet_NaN();
  assert(!CudaMatrixProductAdmission::resolve(target, 8, 1, profile).requires_provider_context());
  const auto huge = CudaMatrixProductAdmission::resolve(
      target, std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max());
  assert(huge.estimate().flops == std::numeric_limits<std::uint64_t>::max());
  const auto empty = CudaMatrixProductAdmission::resolve(target, 0, 1, profile);
  assert(!empty.requires_provider_context() && empty.estimate().flops == 0);
}
