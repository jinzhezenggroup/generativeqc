// Compose the production physical-source VJP with borrowed device nuclear weights.
// RawSource supplies normalized metadata only; its CPU read() is never called.
#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "cc/df_source_response.hpp"
#include "posthf/raw_source.hpp"
#include "runtime/cuda_resources.cuh"
#include "scf/cuda_df_nuclear_sink.hpp"

extern "C" int df_nuclear_sink_probe(void* opaque, const double* coefficients, std::size_t occupied,
                                     const double* const* inputs, std::size_t budget, int failure,
                                     double* gradient, double* frame, std::size_t* counts,
                                     char* error, std::size_t error_size) noexcept {
  using namespace generativeqc;
  try {
    const auto& raw = *static_cast<posthf::RawSource*>(opaque);
    const auto n = raw.nbf(), q = raw.naux(), v = n - occupied;
    hf::PhysicalReference ref;
    ref.nbf = n;
    ref.nocc = occupied;
    ref.coefficients.assign(coefficients, coefficients + n * n);
    auto fitted = cc::build_df_source_cuda(raw.orbital(), raw.auxiliary(), ref, 1ULL << 30, 1e-10,
                                           0, 0, true);
    auto source = fitted.response_state;
    cc::DFFactorResponseResult seeds;
    seeds.source_identity = fitted.source_identity;
    const std::array<std::size_t, 3> sizes{q * occupied * occupied, q * occupied * v, q * v * v};
    const std::array<std::vector<double>*, 3> sectors{&seeds.boo, &seeds.bov, &seeds.bvv};
    std::size_t caller_values = 2 * n * n + 3 * raw.orbital().atoms.size();
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      sectors[i]->assign(inputs[i], inputs[i] + sizes[i]);
      caller_values += sizes[i];
    }
    fitted = {};
    std::vector<double>().swap(ref.coefficients);
    std::vector<double> bar_c(n * n);
    runtime::OwnedCudaStream other_stream(0);
    // These owners must die before the retained source releases its stream.
    scf::CudaDfNuclearSink sink(0, raw.orbital(), raw.auxiliary(), budget);
    const auto external = sink.numeric_capacity_bytes() + caller_values * sizeof(double) +
                          posthf::source_capacity(raw.orbital()) +
                          posthf::source_capacity(raw.auxiliary());
    std::size_t raw_values = 0, metric_values = 0;
    const auto response = cc::pullback_df_source_cuda(
        source, seeds,
        [&](std::size_t mu, const double* row, cudaStream_t stream) {
          sink.consume(0, {mu * n * q, 1, 1, 1}, row, n * q, stream);
          raw_values += n * q;
          if (failure == 1) throw std::runtime_error("injected failure after nuclear row");
        },
        [&](const double* dc, const double* dm, cudaStream_t stream) {
          sink.consume(1, {0, 1, 1, 1}, dm, q * q, stream);
          metric_values += q * q;
          runtime::cuda_resource_check(cudaMemcpyAsync(bar_c.data(), dc, n * n * sizeof(double),
                                                       cudaMemcpyDeviceToHost, stream));
          if (failure == 2) throw std::runtime_error("injected failure after nuclear metric");
          if (failure == 3) sink.consume(1, {0, 1, 1, 1}, dm, 1, nullptr);
          if (failure == 5) sink.consume(1, {0, 1, 1, 1}, dm, 1, other_stream.get());
        },
        budget, external);
    // Producer success is the publication gate. A callback exception never gets
    // here, even if a partial gradient was already computed on the device.
    const auto result = sink.finish();
    if (failure == 4) {
      bool refused = false;
      try {
        (void)sink.finish();
      } catch (const std::logic_error&) {
        refused = true;
      }
      if (!refused) throw std::runtime_error("sink published twice");
    }
    const auto stats = sink.resources();
    const std::size_t work[]{response.numeric_capacity_bytes,
                             sink.numeric_capacity_bytes(),
                             raw_values,
                             metric_values,
                             stats.host_to_device_bytes,
                             stats.response_host_to_device_bytes,
                             stats.device_to_host_bytes,
                             stats.device_response_bytes,
                             stats.tiles,
                             stats.stream_synchronizations};
    std::copy(result.begin(), result.end(), gradient);
    std::copy(bar_c.begin(), bar_c.end(), frame);
    std::copy(std::begin(work), std::end(work), counts);
    return 0;
  } catch (const std::exception& exception) {
    if (error && error_size) std::snprintf(error, error_size, "%s", exception.what());
    return 1;
  }
}
