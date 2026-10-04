// Compare complete scalar/matrix Lambda solves and parameter cotangents using
// exactly one physical RHF/DF/CCSD state and one triples amplitude source.
// This isolates schedule arithmetic from independent cold-reference variation.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cc/df_triples.hpp"
#include "cc/lambda_response.hpp"
#include "methods/rccsd_method.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "runtime/execution_context.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using generativeqc::posthf::checked_add;
using generativeqc::posthf::checked_mul;

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
std::size_t capacity(std::initializer_list<const std::vector<double>*> arrays) {
  std::size_t count = 0;
  for (const auto* a : arrays) count = checked_add(count, a->capacity());
  return checked_mul(count, sizeof(double));
}
std::size_t remainder(std::size_t total, std::size_t live) {
  if (live > total) throw std::length_error("shared-state diagnostic exceeds budget");
  return total - live;
}
void read_shells(std::istream& input, generativeqc::core::System& system, std::size_t count) {
  system.shells.resize(count);
  for (auto& shell : system.shells) {
    std::size_t primitives = 0;
    input >> shell.atom_index >> shell.angular_momentum >> primitives;
    if (!input || !primitives || primitives > 1000)
      throw std::invalid_argument("invalid molecular probe shell");
    shell.primitives.resize(primitives);
    for (auto& p : shell.primitives) input >> p.exponent >> p.coefficient;
  }
  if (!input) throw std::invalid_argument("truncated molecular probe basis");
  std::string detail;
  if (generativeqc::molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS)
    throw std::invalid_argument(detail);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::invalid_argument("usage: df-lambda-shared-state INPUT OUTPUT_JSON");
    std::ifstream input(argv[1]);
    std::size_t atoms = 0, orbital_shells = 0, auxiliary_shells = 0, budget = 0;
    input >> atoms >> orbital_shells >> auxiliary_shells >> budget;
    if (!input || !atoms || atoms > 1000 || !orbital_shells || !auxiliary_shells || !budget)
      throw std::invalid_argument("invalid molecular probe dimensions");
    generativeqc::core::System orbital, auxiliary;
    orbital.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
    orbital.atoms.resize(atoms);
    for (auto& atom : orbital.atoms)
      input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    auxiliary = orbital;
    read_shells(input, orbital, orbital_shells);
    read_shells(input, auxiliary, auxiliary_shells);
    generativeqc::core::ContextState context;
    context.requested_backend = GENERATIVEQC_BACKEND_CUDA;
    generativeqc::runtime::ExecutionContext execution(context);
    generativeqc_method_descriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
    descriptor.method = GENERATIVEQC_METHOD_RCCSD;
    descriptor.precision_mode = GENERATIVEQC_PRECISION_FP64;
    descriptor.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
    descriptor.max_iterations = 150;
    descriptor.energy_tolerance = 1e-12;
    descriptor.density_tolerance = 1e-11;
    descriptor.ccsd_max_iterations = 150;
    descriptor.ccsd_diis_history = 6;
    descriptor.ccsd_energy_tolerance = 1e-12;
    descriptor.ccsd_residual_tolerance = 1e-10;
    descriptor.correlation_memory_budget_bytes = budget;
    auto phase = Clock::now();
    auto state = generativeqc::methods::detail::run_rccsd_native_state(
        execution, orbital, descriptor, nullptr, nullptr, nullptr, 0, &auxiliary, true, true);
    if (!state.solved.converged()) throw std::runtime_error("shared-state CCSD failed");
    const auto primal_seconds = elapsed(phase);
    std::cerr << "Shared native primal completed: " << primal_seconds << " s\n";
    const auto& p = state.problem;
    const auto base = checked_add(p.reference_retained_bytes,
                                  checked_add(generativeqc::cc::problem_host_bytes(p),
                                              capacity({&state.solved.t1, &state.solved.t2})));
    const auto borrowed =
        checked_mul(p.df_bov.size() + p.df_bvv.size() + p.ovoo.size() + p.ovov.size() +
                        p.fov.size() + state.solved.t1.size() + state.solved.t2.size() +
                        state.eps_o.size() + state.eps_v.size(),
                    sizeof(double));
    phase = Clock::now();
    const auto t = generativeqc::cc::triples::pullback_df_cuda(
        p.nocc, p.nvir, p.naux, p.df_bov.data(), p.df_bvv.data(), p.ovoo.data(), p.ovov.data(),
        p.fov.data(), state.solved.t1.data(), state.solved.t2.data(), state.eps_o.data(),
        state.eps_v.data(), 1e-10, budget, execution.device_id(), remainder(base, borrowed));
    const auto triples_seconds = elapsed(phase);
    const auto tbytes =
        capacity({&t.bov, &t.bvv, &t.ovoo, &t.ovov, &t.fov, &t.t1, &t.t2, &t.eps_o, &t.eps_v});
    generativeqc::cc::LambdaOptions options;
    options.cc_tolerance = options.lambda_tolerance = 1e-9;
    options.gmres.absolute_tolerance = 1e-12;
    options.max_bytes = remainder(budget, tbytes);
    options.gmres.max_workspace_bytes = options.max_bytes;
    phase = Clock::now();
    const auto matrix = generativeqc::cc::solve_lambda_parameter_response_cuda_with_energy_source(
        p, state.solved, t.t1, t.t2, execution.device_id(), options);
    const auto matrix_seconds = elapsed(phase);
    if (!matrix.lambda.converged() || !matrix.lambda.diagnostic.df_matrix_gemm)
      throw std::runtime_error("shared-state matrix Lambda failed or fell back");
    std::cerr << "Shared-state matrix Lambda completed: " << matrix_seconds << " s\n";
    // The first detached publication stays live during the second solve. Charge
    // every numeric vector, including empty conventional-only slots, explicitly.
    const auto first_bytes =
        capacity({&matrix.lambda.lambda1, &matrix.lambda.lambda2, &matrix.foo, &matrix.fov,
                  &matrix.fvv, &matrix.ovov, &matrix.ovvo, &matrix.oovv, &matrix.ovvv, &matrix.ovoo,
                  &matrix.oooo, &matrix.vvvv, &matrix.df_bov, &matrix.df_bvv});
    options.df_matrix_gemm = false;
    options.max_bytes = remainder(budget, checked_add(tbytes, first_bytes));
    options.gmres.max_workspace_bytes = options.max_bytes;
    phase = Clock::now();
    const auto scalar = generativeqc::cc::solve_lambda_parameter_response_cuda_with_energy_source(
        p, state.solved, t.t1, t.t2, execution.device_id(), options);
    const auto scalar_seconds = elapsed(phase);
    if (!scalar.lambda.converged() || scalar.lambda.diagnostic.df_matrix_gemm)
      throw std::runtime_error("shared-state scalar Lambda failed");
    std::ofstream output(argv[2]);
    if (!output) throw std::runtime_error("cannot open shared-state output");
    output << std::setprecision(17) << "{\n";
    const auto field = [&](const char* name, auto value) {
      output << "  " << std::quoted(name) << ": " << value << ",\n";
    };
    field("nocc", p.nocc);
    field("nvir", p.nvir);
    field("naux", p.naux);
    field("primal_seconds", primal_seconds);
    field("triples_pullback_seconds", triples_seconds);
    field("matrix_seconds", matrix_seconds);
    field("scalar_seconds", scalar_seconds);
    field("matrix_independent_residual", matrix.lambda.diagnostic.independent_residual_norm);
    field("scalar_independent_residual", scalar.lambda.diagnostic.independent_residual_norm);
    field("matrix_iterations", matrix.lambda.diagnostic.iterations);
    field("scalar_iterations", scalar.lambda.diagnostic.iterations);
    field("matrix_actions", matrix.lambda.diagnostic.operator_actions);
    field("scalar_actions", scalar.lambda.diagnostic.operator_actions);
    field("first_publication_reserved_bytes", first_bytes);
    field("atol", 3e-10);
    field("rtol", 3e-10);
    output << "  \"arrays\": {\n";
    bool first = true, passed = true;
    const auto compare = [&](const char* name, const auto& a, const auto& b) {
      if (a.size() != b.size()) throw std::runtime_error("shared-state output shape differs");
      double maximum = 0.0, scaled = 0.0;
      std::size_t failed = 0;
      for (std::size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
          throw std::runtime_error("nonfinite shared-state response");
        const auto error = std::abs(a[i] - b[i]);
        const auto tolerance = 3e-10 + 3e-10 * std::abs(b[i]);
        maximum = std::max(maximum, error);
        scaled = std::max(scaled, error / tolerance);
        if (error > tolerance) ++failed;
      }
      if (!first) output << ",\n";
      first = false;
      output << "    " << std::quoted(name) << ": {\"size\": " << a.size()
             << ", \"max_abs_error\": " << maximum << ", \"max_gate_fraction\": " << scaled
             << ", \"failed_elements\": " << failed << "}";
      passed = passed && !failed;
    };
    compare("lambda1", matrix.lambda.lambda1, scalar.lambda.lambda1);
    compare("lambda2", matrix.lambda.lambda2, scalar.lambda.lambda2);
    compare("foo", matrix.foo, scalar.foo);
    compare("fov", matrix.fov, scalar.fov);
    compare("fvv", matrix.fvv, scalar.fvv);
    compare("ovov", matrix.ovov, scalar.ovov);
    compare("ovvo", matrix.ovvo, scalar.ovvo);
    compare("oovv", matrix.oovv, scalar.oovv);
    compare("ovoo", matrix.ovoo, scalar.ovoo);
    compare("oooo", matrix.oooo, scalar.oooo);
    compare("df_bov", matrix.df_bov, scalar.df_bov);
    compare("df_bvv", matrix.df_bvv, scalar.df_bvv);
    output << "\n  },\n  \"passed\": " << (passed ? "true" : "false") << "\n}\n";
    output.close();
    if (!output) throw std::runtime_error("failed publishing shared-state output");
    return passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
