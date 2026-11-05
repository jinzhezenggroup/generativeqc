#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "api/error.hpp"
#include "cc/uccsd_solver.hpp"
#include "core/work_budget.hpp"
#include "hf/df_reference_status.hpp"
#include "hf/reference.hpp"
#include "scf/fock_symmetry.hpp"

namespace {
constexpr const char* variable = "GENERATIVEQC_CORRELATION_MAX_SCALAR_WORK";

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

/** Restore the invoking process policy even when an assertion throws. */
class WorkPolicy {
 public:
  WorkPolicy() {
    if (const auto* value = std::getenv(variable)) previous_ = value;
    unsetenv(variable);
  }
  ~WorkPolicy() {
    if (previous_)
      setenv(variable, previous_->c_str(), 1);
    else
      unsetenv(variable);
  }

 private:
  std::optional<std::string> previous_;
};

void work_admission() {
  WorkPolicy policy;
  require(generativeqc::core::correlation_work_limit() == 100000000000ULL,
          "default scalar-work policy changed");
  generativeqc::cc::SolverOptions options;
  options.max_iterations = 150;
  options.max_bytes = 64ULL << 30;
  try {
    generativeqc::cc::unrestricted_capacity(28, 8, 7, options, true, 1024);
    throw std::runtime_error("NO/DZ default work policy unexpectedly admitted");
  } catch (const generativeqc::core::WorkLimitError& error) {
    const std::string message(error.what());
    for (const auto* field :
         {"phase=uccsd_primal", "nbf=28", "nocc_alpha=8", "nocc_beta=7", "iteration_ceiling=150",
          "estimated_work=", "allowed_work=", "numeric_storage_bytes="})
      require(message.find(field) != std::string::npos, "missing admission diagnostic");
    std::string detail;
    require(generativeqc::api::map_exception(&detail) == GENERATIVEQC_STATUS_RESOURCE_LIMIT,
            "work refusal was mislabeled as memory/numerical failure");
    require(detail == message, "work diagnostic lost at public boundary");
  }
  setenv(variable, "1000000000000", 1);
  require(
      generativeqc::cc::unrestricted_capacity(28, 8, 7, options, true, 1024) < options.max_bytes,
      "explicit finite work policy did not admit NO/DZ");
  options.max_bytes = 1;
  try {
    generativeqc::cc::unrestricted_capacity(28, 8, 7, options, true, 1024);
    throw std::runtime_error("work override bypassed numeric memory admission");
  } catch (const std::length_error&) {
  }
  for (const auto* invalid : {"", "0", "-1", "+1", " 1", "1x", "18446744073709551616"}) {
    setenv(variable, invalid, 1);
    try {
      generativeqc::core::correlation_work_limit();
      throw std::runtime_error("malformed work policy accepted");
    } catch (const std::invalid_argument&) {
    }
  }
}

void reference_status() {
  using generativeqc::hf::require_df_reference_result;
  std::vector<generativeqc::scf::RhfBucketItem> results(1);
  for (const auto status : {GENERATIVEQC_STATUS_OUT_OF_MEMORY, GENERATIVEQC_STATUS_CUDA_ERROR,
                            GENERATIVEQC_STATUS_NOT_IMPLEMENTED, GENERATIVEQC_STATUS_NOT_CONVERGED,
                            GENERATIVEQC_STATUS_INVALID_ARGUMENT}) {
    results[0].status = status;
    results[0].detail = "injected provider failure";
    try {
      require_df_reference_result(results);
      throw std::runtime_error("failed DF reference accepted");
    } catch (const generativeqc::Error& error) {
      require(error.status() == status, "DF provider status was overwritten");
      require(std::string(error.what()).find(results[0].detail) != std::string::npos,
              "DF provider detail was discarded");
    }
  }
  results[0].status = GENERATIVEQC_STATUS_SUCCESS;
  for (const bool converged : {false, true}) {
    results[0].scf.converged = converged;
    try {
      require_df_reference_result(results);
      throw std::runtime_error("invalid DF export accepted");
    } catch (const generativeqc::Error& error) {
      require(error.status() == (converged ? GENERATIVEQC_STATUS_NUMERICAL_FAILURE
                                           : GENERATIVEQC_STATUS_NOT_CONVERGED),
              "DF convergence/export failures were conflated");
    }
  }
  results[0].scf.reference = std::make_shared<generativeqc::hf::PhysicalReference>();
  require_df_reference_result(results);
  try {
    require_df_reference_result({});
    throw std::runtime_error("missing DF backend record accepted");
  } catch (const generativeqc::Error& error) {
    require(error.status() == GENERATIVEQC_STATUS_INTERNAL_ERROR, "invalid DF cardinality status");
  }
}

void fock_symmetry() {
  std::vector<double> first{1.0, 0.1 + 2e-13, 0.1, 2.0};
  std::vector<double> second{1.2, 0.2, 0.2 + 3e-13, 2.1};
  require(std::abs(1e5 * (first[1] - first[2]) - 99999 * (second[1] - second[2])) > 1e-9,
          "fixture does not amplify raw Fock roundoff");
  generativeqc::scf::preserve_fock_symmetry(first, 2);
  generativeqc::scf::preserve_fock_symmetry(second, 2);
  for (std::size_t entry = 0; entry < first.size(); ++entry)
    first[entry] = 1e5 * first[entry] - 99999 * second[entry];
  require(first[1] == first[2], "Pulay mixing amplified Hermitian roundoff");
  generativeqc::scf::preserve_fock_symmetry(first, 2);
  for (auto invalid : {std::vector<double>{1.0, 0.1, 0.2, 2.0},
                       std::vector<double>{1.0, 0.0, 0.0, std::numeric_limits<double>::infinity()},
                       std::vector<double>{1.0, 0.0, 2.0}}) {
    try {
      generativeqc::scf::preserve_fock_symmetry(invalid, 2);
      throw std::runtime_error("invalid Fock source was silently repaired");
    } catch (const std::invalid_argument&) {
    }
  }
}
}  // namespace

int main() {
  try {
    work_admission();
    reference_status();
    fock_symmetry();
    std::cout << "correlation admission/status contracts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
