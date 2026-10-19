/** Test-only device-resident residual seam with synthetic Hermite distributions.
 * No public ABI, host ERI provider or production allocation is introduced. */
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "scf/cuda/direct_md_j.hpp"
#include "generated_md_j_masked.cuh"

using namespace generativeqc::scf::cuda_execution;

namespace {
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct Storage {
  std::vector<void*> pointers;
  ~Storage() {
    for (auto pointer : pointers) cudaFree(pointer);
  }
  template <class Value>
  Value* copy(const Value* source, std::size_t count) {
    Value* pointer{};
    check(cudaMalloc(&pointer, count * sizeof(Value)));
    pointers.push_back(pointer);
    if (source) check(cudaMemcpy(pointer, source, count * sizeof(Value), cudaMemcpyHostToDevice));
    else check(cudaMemset(pointer, 0, count * sizeof(Value)));
    return pointer;
  }
};

/** Identity slots let the same dense synthetic E oracle exercise a compact-E
 * native snapshot without changing its production accessor or basis layout. */
template <class View, class Pair>
void attach_transform_slots(View& md, std::vector<Pair>& pairs, Storage& storage) {
  if constexpr (requires { md.transform_slots; }) {
    std::vector<std::uint16_t> slots;
    for (auto& pair : pairs) {
      pair.transform_map = slots.size();
      const unsigned count = pair.first_count * pair.second_count * pair.hermites;
      for (unsigned index = 0; index < count; ++index) slots.push_back(index);
    }
    md.transform_slots = storage.copy(slots.data(), slots.size());
  }
}
}  // namespace

extern "C" int run_masked_gpu(
    unsigned ket_hermites, unsigned bra_pairs, unsigned ket_pairs,
    const unsigned* descriptors, bool same_pair, bool unrestricted,
    double screening, const double* bounds, const double* density, const double* beta,
    unsigned bra_primitives, unsigned ket_primitives, const double* bra_transform,
    const double* ket_transform, const double* bra_parameters, const double* ket_parameters,
    bool masked, bool counted, unsigned preprojected, double* result, unsigned long long* counts) {
  try {
    constexpr unsigned nbf = 20, elements = 3 * nbf * nbf;
    Storage storage;
    std::vector<MdJPair> pairs;
    std::vector<MdJPrimitive> primitives;
    std::vector<double> transforms;
    const unsigned pair_count = same_pair ? 1 : bra_pairs + ket_pairs;
    for (unsigned pair_index = 0; pair_index < pair_count; ++pair_index) {
      const auto* descriptor = descriptors + 4 * pair_index;
      const bool bra_side = pair_index < bra_pairs;
      const unsigned local_index = bra_side ? pair_index : pair_index - bra_pairs;
      const unsigned hermites = bra_side ? 10 : ket_hermites;
      const unsigned primitive_count = bra_side ? bra_primitives : ket_primitives;
      const double* parameters = (bra_side ? bra_parameters : ket_parameters) + local_index * primitive_count * 4;
      const unsigned components = descriptor[2] * descriptor[3];
      const double* coefficients = (bra_side ? bra_transform : ket_transform) +
                                  local_index * primitive_count * components * hermites;
      MdJPair pair{};
      pair.system = 1;
      pair.first_ao = descriptor[0];
      pair.second_ao = descriptor[1];
      pair.first_count = descriptor[2];
      pair.second_count = descriptor[3];
      pair.angular = hermites == 4 ? 1 : 2;
      pair.hermites = hermites;
      pair.primitive_begin = primitives.size();
      for (unsigned index = 0; index < primitive_count; ++index) {
        MdJPrimitive primitive{};
        primitive.pair = pair_index;
        primitive.transform_offset = transforms.size();
        primitive.exponent = parameters[4 * index];
        primitive.product = {parameters[4 * index + 1], parameters[4 * index + 2],
                             parameters[4 * index + 3]};
        primitives.push_back(primitive);
        const auto* begin = coefficients + index * components * hermites;
        transforms.insert(transforms.end(), begin, begin + components * hermites);
      }
      pair.primitive_end = primitives.size();
      pairs.push_back(pair);
    }
    std::vector<double> minimum(pair_count), maximum(pair_count), density_bounds(pair_count, 1e20);
    bool zero_density = true;
    for (unsigned index = nbf * nbf; index < 2 * nbf * nbf; ++index)
      if (density[index] + (unrestricted ? beta[index] : 0.0) != 0.0) zero_density = false;
    if (zero_density) std::fill(density_bounds.begin(), density_bounds.end(), 0.0);
    for (unsigned pair_index = 0; pair_index < pair_count; ++pair_index) {
      const auto pair = pairs[pair_index];
      minimum[pair_index] = 1e300;
      for (int first = 0; first < pair.first_count; ++first)
        for (int second = 0; second < pair.second_count; ++second) {
          const unsigned row = pair.first_ao + first, column = pair.second_ao + second;
          for (auto index : {nbf * nbf + row * nbf + column, nbf * nbf + column * nbf + row}) {
            minimum[pair_index] = std::min(minimum[pair_index], bounds[index]);
            maximum[pair_index] = std::max(maximum[pair_index], bounds[index]);
          }
        }
    }
    MdJView md{};
    md.pair_count = pair_count;
    md.residual_candidate_count = pair_count * (pair_count + 1) / 2;
    md.masked_residual = masked;
    attach_transform_slots(md, pairs, storage);
    md.pairs = storage.copy(pairs.data(), pairs.size());
    md.primitives = storage.copy(primitives.data(), primitives.size());
    md.transforms = storage.copy(transforms.data(), transforms.size());
    md.minimum_bounds = storage.copy(minimum.data(), minimum.size());
    md.maximum_bounds = storage.copy(maximum.data(), maximum.size());
    const double global_maximum = *std::max_element(maximum.begin(), maximum.end());
    md.maximum_bound = storage.copy(&global_maximum, 1);
    md.density_bounds = storage.copy(density_bounds.data(), density_bounds.size());
    md.source_cursor = storage.copy<unsigned long long>(nullptr, 1);
    if (counted) md.work_counts = storage.copy<MdJWorkCounts>(nullptr, 1);
    std::vector<unsigned> active(pair_count);
    for (unsigned index = 0; index < pair_count; ++index) active[index] = index;
    std::stable_sort(active.begin(), active.end(), [&](unsigned first, unsigned second) {
      return pairs[first].angular < pairs[second].angular;
    });
    md.active_pairs = storage.copy(active.data(), pair_count);
    for (unsigned angular = 0; angular < 5; ++angular) {
      md.active_pair_offsets[angular + 1] = md.active_pair_offsets[angular];
      for (const auto& pair : pairs)
        md.active_pair_offsets[angular + 1] += pair.angular == angular;
    }
    auto* device_bounds = storage.copy(bounds, elements);
    auto* device_density = storage.copy(density, elements);
    auto* device_beta = storage.copy(beta, elements);
    auto* output = storage.copy(result, elements);
    DeviceBatch batch{};
    batch.batch_size = 3;
    batch.nbf = nbf;
    md.masked_tiles = preprojected == 5 ? 2 : preprojected >= 3 ? preprojected - 2 : 0;
    md.residual_tiled = preprojected != 5;
    if (preprojected && preprojected < 3) {
      std::vector<MdMaskedResidentPair> resident(pair_count);
      std::size_t suffixes = 0;
      for (unsigned index = 0; index < pair_count; ++index) {
        if (preprojected == 2 && index == 0) continue;
        const auto pair = pairs[index];
        resident[index].suffix_offset = suffixes;
        suffixes += (pair.primitive_end - pair.primitive_begin) *
                    (2 * pair.first_count * pair.second_count + 1) * pair.hermites;
      }
      md.masked_pairs = storage.copy(resident.data(), resident.size());
      md.masked_suffixes = storage.copy<double>(nullptr, suffixes);
      md.masked_suffix_count = suffixes;
      check(prepare_md_j_masked(nullptr, batch, md, device_bounds));
    }
    if (preprojected) {
      launch_md_source_jk(nullptr, batch, md, 1, 1, true, false, unrestricted, screening,
                          device_bounds, device_density, device_beta, output, nullptr, nullptr);
      check(cudaMemset(device_density, 0, elements * sizeof(double)));
      check(cudaMemset(device_beta, 0, elements * sizeof(double)));
      check(cudaMemset(md.density_bounds, 0, density_bounds.size() * sizeof(double)));
      launch_md_source_jk(nullptr, batch, md, 1, 1, true, false, unrestricted, screening,
                          device_bounds, device_density, device_beta, output, nullptr, nullptr);
      std::vector<double> zero(elements);
      check(cudaMemcpy(zero.data(), output, elements * sizeof(double), cudaMemcpyDeviceToHost));
      for (unsigned index = nbf * nbf; index < 2 * nbf * nbf; ++index)
        if (zero[index] != 0.0) throw std::runtime_error("stale projected density");
      check(cudaMemcpy(device_density, density, elements * sizeof(double), cudaMemcpyHostToDevice));
      check(cudaMemcpy(device_beta, beta, elements * sizeof(double), cudaMemcpyHostToDevice));
      check(cudaMemcpy(md.density_bounds, density_bounds.data(),
                       density_bounds.size() * sizeof(double), cudaMemcpyHostToDevice));
      if (counted) check(cudaMemset(md.work_counts, 0, sizeof(MdJWorkCounts)));
    }
    launch_md_source_jk(nullptr, batch, md, 1, 1, true, false, unrestricted, screening,
                        device_bounds, device_density, device_beta, output, nullptr, nullptr);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(result, output, elements * sizeof(double), cudaMemcpyDeviceToHost));
    MdJWorkCounts work{};
    if (counted) check(cudaMemcpy(&work, md.work_counts, sizeof(work), cudaMemcpyDeviceToHost));
    std::copy_n(work.residual_tasks, 25, counts);
    std::copy_n(work.masked_tasks, 25, counts + 25);
    std::copy_n(work.masked_groups, 25, counts + 50);
    std::copy_n(work.masked_summands, 25, counts + 75);
    std::copy_n(work.masked_output_summands, 25, counts + 100);
    std::copy_n(work.masked_unfused_output_summands, 25, counts + 125);
    std::copy_n(work.residual_candidates, 25, counts + 150);
    std::copy_n(work.residual_screened_candidates, 25, counts + 175);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "masked probe: %s\n", error.what());
    return 1;
  } catch (cudaError_t error) {
    std::fprintf(stderr, "masked probe CUDA: %s\n", cudaGetErrorString(error));
    return 2;
  } catch (...) {
    return 2;
  }
}
