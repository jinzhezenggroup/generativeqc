#ifndef GENERATIVEQC_SCF_FLEET_ITEM_HPP
#define GENERATIVEQC_SCF_FLEET_ITEM_HPP

#include <cstddef>
#include <new>
#include <stdexcept>
#include <utility>

#include "generativeqc/generativeqc.hpp"
#include "scf/types.hpp"

namespace generativeqc::scf {

/** Source-order result shared by independent and batched backend schedules. */
struct FleetItemResult {
  generativeqc_status status{GENERATIVEQC_STATUS_INTERNAL_ERROR};
  ScfResult scf;
  std::size_t bucket_id{};
  bool warm_start_used{};
  bool warm_start_fallback{};
  generativeqc_backend executed_backend{GENERATIVEQC_BACKEND_CPU_REFERENCE};
};

/** Isolate an attempt without replacing the backend's explicit result status.
 * Callbacks own solver work and diagnostics, not fleet failure policy. Returns
 * false when an exception prevents a complete backend result. */
template <class Attempt>
bool execute_fleet_item_attempt(FleetItemResult& item, Attempt&& attempt) {
  try {
    std::forward<Attempt>(attempt)();
    return true;
  } catch (const std::bad_alloc&) {
    item.status = GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument&) {
    item.status = GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const generativeqc::Error& error) {
    item.status = error.status();
  } catch (const std::exception&) {
    item.status = GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  } catch (...) {
    item.status = GENERATIVEQC_STATUS_INTERNAL_ERROR;
  }
  return false;
}

/** A failed warm seed permits at most one ordinary cold attempt. Resource and
 * driver failures cannot be repaired by changing the seed. Fock-only diagnostic
 * runs are not SCF attempts and must never trigger another solve. */
template <class Attempt>
void retry_fleet_item_cold(FleetItemResult& item, bool fock_only_diagnostic, Attempt&& attempt) {
  if (!item.warm_start_used || item.warm_start_fallback || fock_only_diagnostic ||
      item.status == GENERATIVEQC_STATUS_SUCCESS || item.status == GENERATIVEQC_STATUS_CUDA_ERROR ||
      item.status == GENERATIVEQC_STATUS_OUT_OF_MEMORY)
    return;
  item.warm_start_fallback = true;
  execute_fleet_item_attempt(item, std::forward<Attempt>(attempt));
}

}  // namespace generativeqc::scf

#endif
