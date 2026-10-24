#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "cc/lambda_response.hpp"

namespace {
using namespace generativeqc::cc;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

/** Independent noninteracting reference: R1=(eps_a-eps_i)*t1 and
 * R2=(eps_a+eps_b-eps_i-eps_j)*t2. With an external energy source S, the
 * exact Lambda is S/D in dense coordinates, including off-diagonal pair
 * orbits. No generated derivative or CPU response supplies the expected values.
 */
void qualify(bool cuda, bool derived, bool resident = false, std::size_t device_budget = 0,
             bool df = false) {
  constexpr std::size_t o = 2, v = 3, n1 = o * v, n2 = n1 * n1;
  const double occupied[o]{-20.0, -1.0}, virtuals[v]{0.3, 2.0, 17.0};
  Problem p;
  p.nocc = o;
  p.nvir = v;
  p.foo.resize(o * o);
  p.fov.resize(n1);
  p.fvv.resize(v * v);
  p.ovov.resize(n2);
  p.ovvo.resize(n2);
  p.oovv.resize(n2);
  p.ovvv.resize(o * v * v * v);
  p.ovoo.resize(o * v * o * o);
  p.oooo.resize(o * o * o * o);
  p.vvvv.resize(v * v * v * v);
  p.d1.resize(n1);
  p.d2.resize(n2);
  p.initial_t1.resize(n1);
  p.initial_t2.resize(n2);
  if (df) {
    p.naux = 5;
    p.ovvv.clear();
    p.vvvv.clear();
    p.df_bov.resize(p.naux * n1);
    p.df_bvv.resize(p.naux * v * v);
  }
  for (std::size_t i = 0; i < o; ++i) p.foo[i * o + i] = occupied[i];
  for (std::size_t a = 0; a < v; ++a) p.fvv[a * v + a] = virtuals[a];
  std::vector<double> source1(n1), source2(n2), expected1(n1), expected2(n2);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a) {
      const auto k = i * v + a;
      p.d1[k] = occupied[i] - virtuals[a];
      source1[k] = 0.002 * (k + 1);
      expected1[k] = source1[k] / p.d1[k];
    }
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          const auto k = ((i * o + j) * v + a) * v + b;
          const auto mate = ((j * o + i) * v + b) * v + a;
          p.d2[k] = (occupied[i] - virtuals[a]) + (occupied[j] - virtuals[b]);
          source2[k] = 0.001 * (std::min(k, mate) + 1);
          expected2[k] = source2[k] / p.d2[k];
        }
  if (derived) {
    std::vector<double> eps(occupied, occupied + o);
    eps.insert(eps.end(), virtuals, virtuals + v);
    initialize_canonical_denominators(p, eps, SolverOptions{}, true);
    require(p.d2.empty(), "Lambda must consume canonical doubles without materialization");
  }
  SolverResult cc;
  cc.status = SolveStatus::Converged;
  cc.t1.resize(n1);
  cc.t2.resize(n2);
  LambdaOptions options;
  options.gmres.absolute_tolerance = 1e-12;
  options.gmres.max_iterations = 200;
  const auto capacity = lambda_cpu_numeric_capacity(p, cc, options, true);
  // Exact admitted host capacity includes the audit/diagonal buffer already.
  options.max_bytes = capacity;
  if (device_budget) options.max_bytes = std::max(capacity, device_budget);
  options.cuda_resident_krylov = resident;
  auto solve = [&] {
#if GENERATIVEQC_HAS_CUDA
    if (cuda) return solve_lambda_cuda_with_energy_source(p, cc, source1, source2, 0, options);
#else
    require(!cuda, "CUDA unavailable in this test build");
#endif
    return solve_lambda_cpu_with_energy_source(p, cc, source1, source2, options);
  };
  auto check = [&](const LambdaResult& result, bool preconditioned) {
    require(result.converged(), "Lambda convergence");
    double error = 0.0;
    for (std::size_t k = 0; k < n1; ++k)
      error = std::max(error, std::abs(result.lambda1[k] - expected1[k]));
    for (std::size_t k = 0; k < n2; ++k)
      error = std::max(error, std::abs(result.lambda2[k] - expected2[k]));
    require(error < 1e-11, "independent noninteracting Lambda oracle mismatch");
    require(result.diagnostic.diagonal_preconditioned == preconditioned,
            "preconditioner selection");
    require((result.diagnostic.preconditioner_actions > 0) == preconditioned,
            "preconditioner work");
    require(result.diagnostic.independent_residual_norm < 1e-9, "physical residual");
    if (df && cuda)
      require(result.diagnostic.numeric_capacity_bytes <= options.max_bytes,
              "complete DF host/device numeric bound");
    else
      require(result.diagnostic.numeric_capacity_bytes == capacity, "unchanged numeric bound");
    if (cuda) {
      require(result.diagnostic.resident_krylov == resident, "resident schedule selection");
      require(
          result.diagnostic.resident_krylov_refused == (options.cuda_resident_krylov && !resident),
          "resident refusal selection");
      require(result.diagnostic.owned_device_bytes <= options.max_bytes,
              "simultaneously-live device arenas exceed budget");
      if (resident) {
        const auto dimension = n1 + (n2 + n1) / 2;
        const auto metadata_bytes = (n2 + n1) * sizeof(std::size_t) + dimension * sizeof(double);
        const auto expected_h2d =
            metadata_bytes + dimension * sizeof(double) * (preconditioned ? 2 : 1);
        require(result.diagnostic.krylov_h2d_bytes == expected_h2d,
                "resident iterations uploaded dimension-sized vectors");
        require(!result.diagnostic.resident_krylov_refused &&
                    result.diagnostic.resident_krylov_bytes > 0,
                "resident admission diagnostics");
        require(result.diagnostic.resident_provider_observed_bytes <=
                        result.diagnostic.resident_provider_allowance_bytes &&
                    result.diagnostic.resident_device_admission_bytes <= options.max_bytes,
                "resident provider growth exceeds complete device admission");
        require(result.diagnostic.resident_provider_version > 0 &&
                    result.diagnostic.resident_runtime_version > 0,
                "resident execution provenance");
      }
    }
    if (preconditioned)
      require(result.diagnostic.iterations == 1, "diagonal solve needs one step");
    else
      require(result.diagnostic.iterations > 1, "nontrivial unpreconditioned comparison");
    std::cout << (cuda ? "cuda" : "cpu") << " df=" << df << " resident=" << resident
              << " budget=" << options.max_bytes << " preconditioned=" << preconditioned
              << " iterations=" << result.diagnostic.iterations
              << " actions=" << result.diagnostic.operator_actions << " error=" << error
              << " krylov_h2d=" << result.diagnostic.krylov_h2d_bytes
              << " krylov_d2h=" << result.diagnostic.krylov_d2h_bytes
              << " fences=" << result.diagnostic.krylov_synchronizations
              << " device_bytes=" << result.diagnostic.owned_device_bytes << '\n';
  };
  const auto initial = solve();
  check(initial, true);
  if (cuda && !resident && df) {
    const auto device_limit = options.df_max_device_bytes;
    options.df_max_device_bytes = initial.diagnostic.owned_device_bytes;
    options.cuda_resident_krylov = true;
    check(solve(), true);
    options.cuda_resident_krylov = false;
    options.df_max_device_bytes = device_limit;
  }
  if (cuda && !resident && !df) {
    options.cuda_resident_krylov = true;
    check(solve(), true);
    options.cuda_resident_krylov = false;
  }
  options.diagonal_preconditioning = false;
  require(lambda_cpu_numeric_capacity(p, cc, options, true) == capacity, "optional capacity");
  check(solve(), false);
  if (derived) return;  // Supplied-array fallback cases below require explicit mode.
  options.diagonal_preconditioning = true;
  const auto first = p.d1[0];
  // Alter only optional solver denominators: the physical Fock operator and
  // independent expected Lambda stay unchanged. Unsafe diagonals must fall back.
  for (const double unsafe : {0.0, 1e-14}) {
    p.d1[0] = unsafe;
    check(solve(), false);
  }
  p.d1[0] = first;
  p.d2[1] += 1.0;  // Break the (i,j,a,b)<->(j,i,b,a) denominator orbit.
  check(solve(), false);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const bool cuda = argc == 2 && std::string_view(argv[1]) == "--cuda";
    qualify(cuda, false);
    qualify(cuda, true);
    if (cuda)
      for (const auto budget : {128ULL << 20, 256ULL << 20}) {
        qualify(true, false, true, budget);
        qualify(true, true, true, budget);
      }
    // Explicit DF factors belong to the existing CUDA factorized action owner;
    // the conventional CPU owner must keep rejecting that unsupported input.
    if (cuda)
      for (const auto budget : {256ULL << 20, 512ULL << 20}) {
        qualify(true, false, false, budget, true);
        qualify(true, true, false, budget, true);
        qualify(true, false, true, budget, true);
        qualify(true, true, true, budget, true);
      }
    std::cout << "Lambda preconditioner oracle, fallback and budget gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
