#include <array>
#include <stdexcept>

#include "scf/fleet_item.hpp"

namespace {

using generativeqc::scf::execute_fleet_item_attempt;
using generativeqc::scf::FleetItemResult;
using generativeqc::scf::retry_fleet_item_cold;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void check_failure_mapping() {
  FleetItemResult item;
  const auto check = [&](auto failure, generativeqc_status expected) {
    execute_fleet_item_attempt(item, [&] { throw failure; });
    require(item.status == expected, "fleet exception status changed");
  };
  check(std::bad_alloc{}, GENERATIVEQC_STATUS_OUT_OF_MEMORY);
  check(std::invalid_argument("shape"), GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  check(generativeqc::Error(GENERATIVEQC_STATUS_CUDA_ERROR, "driver"),
        GENERATIVEQC_STATUS_CUDA_ERROR);
  check(std::runtime_error("numerics"), GENERATIVEQC_STATUS_NUMERICAL_FAILURE);
  check(17, GENERATIVEQC_STATUS_INTERNAL_ERROR);
  execute_fleet_item_attempt(item, [&] { item.status = GENERATIVEQC_STATUS_SCF_NOT_CONVERGED; });
  require(item.status == GENERATIVEQC_STATUS_SCF_NOT_CONVERGED,
          "an explicit backend failure was published as success");
}

void check_retry_policy() {
  constexpr std::array backends{GENERATIVEQC_BACKEND_CPU_REFERENCE, GENERATIVEQC_BACKEND_CUDA};
  constexpr std::array statuses{
      GENERATIVEQC_STATUS_SUCCESS,           GENERATIVEQC_STATUS_SCF_NOT_CONVERGED,
      GENERATIVEQC_STATUS_NUMERICAL_FAILURE, GENERATIVEQC_STATUS_INVALID_ARGUMENT,
      GENERATIVEQC_STATUS_CUDA_ERROR,        GENERATIVEQC_STATUS_OUT_OF_MEMORY};
  for (const auto backend : backends) {
    for (const auto status : statuses) {
      for (const bool warm : {false, true}) {
        for (const bool diagnostic : {false, true}) {
          FleetItemResult item;
          item.status = status;
          item.warm_start_used = warm;
          item.executed_backend = backend;
          item.bucket_id = 7;
          unsigned calls = 0;
          const auto cold_attempt = [&] {
            ++calls;
            item.status = GENERATIVEQC_STATUS_SCF_NOT_CONVERGED;
          };
          const bool retry = warm && !diagnostic && status != GENERATIVEQC_STATUS_SUCCESS &&
                             status != GENERATIVEQC_STATUS_CUDA_ERROR &&
                             status != GENERATIVEQC_STATUS_OUT_OF_MEMORY;
          retry_fleet_item_cold(item, diagnostic, cold_attempt);
          require(calls == (retry ? 1U : 0U) && item.warm_start_fallback == retry,
                  "CPU/CUDA warm retry policy diverged");
          require(item.status == (retry ? GENERATIVEQC_STATUS_SCF_NOT_CONVERGED : status),
                  "cold nonconvergence was hidden");
          retry_fleet_item_cold(item, diagnostic, cold_attempt);
          require(calls <= 1, "a failed cold attempt was retried again");
          require(item.executed_backend == backend && item.bucket_id == 7,
                  "retry changed item placement or source-order metadata");
        }
      }
    }
  }
}

void check_failed_retry_isolation() {
  constexpr std::array backends{GENERATIVEQC_BACKEND_CPU_REFERENCE, GENERATIVEQC_BACKEND_CUDA};
  for (const auto backend : backends) {
    for (const bool allocation_failure : {false, true}) {
      FleetItemResult failed;
      failed.executed_backend = backend;
      failed.status = GENERATIVEQC_STATUS_SCF_NOT_CONVERGED;
      failed.warm_start_used = true;
      unsigned calls = 0;
      const auto cold_attempt = [&] {
        ++calls;
        if (allocation_failure) throw std::bad_alloc{};
        throw std::runtime_error("cold attempt failed");
      };
      retry_fleet_item_cold(failed, false, cold_attempt);
      retry_fleet_item_cold(failed, false, cold_attempt);
      require(calls == 1 && failed.warm_start_fallback,
              "cold failure escaped its bounded item lifecycle");
      require(failed.status == (allocation_failure ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                                   : GENERATIVEQC_STATUS_NUMERICAL_FAILURE),
              "cold failure lost its status");
      FleetItemResult neighbor;
      neighbor.executed_backend = backend;
      execute_fleet_item_attempt(neighbor, [&] {
        neighbor.scf.converged = true;
        neighbor.status = GENERATIVEQC_STATUS_SUCCESS;
      });
      require(neighbor.status == GENERATIVEQC_STATUS_SUCCESS && !neighbor.warm_start_used &&
                  !neighbor.warm_start_fallback,
              "a failed ragged item contaminated its neighbor");
    }
  }
}

}  // namespace

int main() {
  check_failure_mapping();
  check_retry_policy();
  check_failed_retry_isolation();
}
