#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "scf/cuda/df_shell_kernel.cuh"
#include "tools/generativeqc_validation/df_shell_fixture.hpp"

namespace {
constexpr unsigned angular_a = DF_TEST_ANGULAR_A, angular_b = DF_TEST_ANGULAR_B,
                   angular_c = DF_TEST_ANGULAR_C;
using generativeqc::df_benchmark::check;
using generativeqc::scf::DfDerivativePairs;
using generativeqc::scf::DfShellBasisView;

/** Exercise the production panel and packet kernels, including subgroup tails. */
template <unsigned Variant, bool Rys, bool Mixed>
void launch(DfShellBasisView first, DfShellBasisView second, DfShellBasisView auxiliary,
            const double* positions, std::size_t begin, std::size_t count, const double* weights,
            double* gradient, unsigned long long* precision, DfDerivativePairs pairs, bool packet) {
  using namespace generativeqc::scf;
  using Schedule = generated_df_shell::Schedule<angular_a, angular_b, angular_c, Variant>;
  const auto tasks = first.count[angular_a];
  const auto blocks = static_cast<unsigned>((tasks + Schedule::groups - 1) / Schedule::groups);
  if (packet) {
    SignaturePacket descriptor;
    descriptor.count = 1;
    descriptor.blocks = blocks;
    descriptor.slices[0] = {first.shell_ids,
                            second.shell_ids,
                            auxiliary.shell_ids,
                            first.begin[angular_a],
                            second.begin[angular_b],
                            auxiliary.begin[angular_c],
                            tasks,
                            1,
                            1,
                            2,
                            2,
                            2,
                            0,
                            tasks,
                            false};
    shell_packet<angular_a, angular_b, angular_c, Variant, Rys, false, Mixed>
        <<<blocks, Schedule::lanes * Schedule::groups>>>(first, auxiliary, positions, begin, count,
                                                         weights, gradient, nullptr, pairs,
                                                         descriptor, nullptr, precision);
  } else {
    shell_panel<angular_a, angular_b, angular_c, Variant, Rys, false, Mixed>
        <<<blocks, Schedule::lanes * Schedule::groups>>>(first, second, auxiliary, positions, begin,
                                                         count, weights, gradient, nullptr, pairs,
                                                         false, tasks, nullptr, precision);
  }
  check(cudaGetLastError());
}
}  // namespace

/** Only the test injects representability failures, after physical normalization.
 * Mode 2 fails after a valid primitive contribution, detecting partial scatter
 * and cache reuse bugs on replay. No production override or GPU probe is used. */
extern "C" int probe(unsigned variant, bool rys, bool mixed, unsigned pair_mode, bool packet,
                     bool spherical, bool clipped, unsigned fault, unsigned replicas,
                     const double* coordinates, const double* exponents, const double* coefficients,
                     const double* weights, double* output, unsigned long long* observed) {
  using namespace generativeqc;
  try {
    if (!replicas || replicas > 5) return -3;
    core::System orbital;
    orbital.multiplicity = 2;
    orbital.basis_representation =
        spherical ? GENERATIVEQC_BASIS_SPHERICAL : GENERATIVEQC_BASIS_CARTESIAN;
    for (unsigned center = 0; center < 3; ++center)
      orbital.atoms.push_back(
          {1, {coordinates[3 * center], coordinates[3 * center + 1], coordinates[3 * center + 2]}});
    auto auxiliary = orbital;
    const std::array<unsigned, 3> degrees{angular_a, angular_b, angular_c};
    for (unsigned center = 0; center < 3; ++center) {
      core::Shell shell;
      shell.atom_index = center;
      shell.angular_momentum = degrees[center];
      for (unsigned primitive = 0; primitive < 2; ++primitive)
        shell.primitives.push_back(
            {exponents[2 * center + primitive], coefficients[2 * center + primitive]});
      (center == 2 ? auxiliary : orbital).shells.push_back(shell);
    }
    std::string detail;
    if (molecule::validate_and_normalize(orbital, detail) != GENERATIVEQC_STATUS_SUCCESS ||
        molecule::validate_and_normalize(auxiliary, detail) != GENERATIVEQC_STATUS_SUCCESS)
      return -1;
    if (fault == 2) auxiliary.shells[0].primitives[1].coefficient *= 1e-50;
    df_benchmark::Storage storage;
    auto first = storage.basis(orbital), second = first, third = storage.basis(auxiliary);
    first.shell_ids = storage.upload(std::vector<std::int32_t>(replicas, 0));
    second.shell_ids = storage.upload(std::vector<std::int32_t>{1});
    third.shell_ids = storage.upload(std::vector<std::int32_t>{0});
    std::fill_n(first.begin, 4, 0);
    std::fill_n(second.begin, 4, 0);
    std::fill_n(third.begin, 4, 0);
    std::fill_n(first.count, 4, 0);
    std::fill_n(second.count, 4, 0);
    std::fill_n(third.count, 4, 0);
    first.count[angular_a] = second.count[angular_b] = third.count[angular_c] = 1;
    first.count[angular_a] = replicas;
    first.primitives = second.primitives = third.primitives = 2;
    const auto first_count =
        molecule::ao_expansions(angular_a, orbital.basis_representation).size();
    const auto second_count = first.basis.nbf - first_count;
    const auto begin = clipped && third.basis.nbf > 1 ? 1U : 0U;
    const auto count = third.basis.nbf - begin;
    const auto pairs = static_cast<DfDerivativePairs>(pair_mode);
    const auto stride = pairs == DfDerivativePairs::packed
                            ? first.basis.nbf * (first.basis.nbf + 1) / 2
                            : first.basis.nbf * first.basis.nbf;
    std::vector<double> panel(count * stride);
    for (std::size_t first_ao = 0; first_ao < first_count; ++first_ao)
      for (std::size_t second_ao = 0; second_ao < second_count; ++second_ao)
        for (std::size_t auxiliary_ao = begin; auxiliary_ao < third.basis.nbf; ++auxiliary_ao) {
          double weight =
              weights[(first_ao * second_count + second_ao) * third.basis.nbf + auxiliary_ao];
          if (fault == 1) weight *= 1e40;
          if (fault == 3) weight *= 1e-50;
          const auto row = first_count + second_ao, offset = (auxiliary_ao - begin) * stride;
          if (pairs == DfDerivativePairs::packed)
            panel[offset + row * (row + 1) / 2 + first_ao] = weight;
          else if (pairs == DfDerivativePairs::symmetric) {
            panel[offset + first_ao * first.basis.nbf + row] = weight * .4;
            panel[offset + row * first.basis.nbf + first_ao] = weight * .6;
          } else
            panel[offset + first_ao * first.basis.nbf + row] = weight;
        }
    const auto positions = storage.upload(std::vector<double>(coordinates, coordinates + 9));
    const auto response = storage.upload(panel);
    const auto gradient = storage.upload(std::vector<double>(9, 0.0));
    const auto precision = storage.upload(std::vector<unsigned long long>(3, 0));
    const auto dispatch_variant = [&]<bool Rys, bool Mixed>() {
      const auto dispatch = [&]<unsigned Variant>() {
        launch<Variant, Rys, Mixed>(first, second, third, positions, begin, count, response,
                                    gradient, precision, pairs, packet);
      };
      if (variant == 0)
        dispatch.template operator()<0>();
      else if (variant == 1)
        dispatch.template operator()<1>();
      else if (variant == 2)
        dispatch.template operator()<2>();
      else
        throw std::runtime_error("invalid test schedule");
    };
    const auto dispatch_rys = [&]<bool Mixed>() {
      if constexpr (scf::generated_df_shell::rys_available<angular_a, angular_b, angular_c>) {
        if (rys) return dispatch_variant.template operator()<true, Mixed>();
      }
      if (rys) throw std::runtime_error("unavailable test quadrature");
      dispatch_variant.template operator()<false, Mixed>();
    };
    if (mixed)
      dispatch_rys.template operator()<true>();
    else
      dispatch_rys.template operator()<false>();
    check(cudaMemcpy(output, gradient, 9 * sizeof(double), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(observed, precision, 3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    return 0;
  } catch (const std::exception&) {
    return -2;
  }
}
