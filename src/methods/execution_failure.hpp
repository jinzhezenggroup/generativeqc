#pragma once

#include <exception>
#include <new>
#include <stdexcept>

#include "generativeqc/generativeqc.hpp"
#include "methods/method.hpp"

namespace generativeqc::methods {

/** Preserve typed backend/resource errors at every method's item boundary.
 * Unknown runtime exceptions retain the legacy numerical status, but that
 * status alone never authorizes resubmitting possibly asynchronous work. */
inline generativeqc_status item_exception_status(
    const std::exception_ptr& failure = std::current_exception()) noexcept {
  if (!failure) return GENERATIVEQC_STATUS_INTERNAL_ERROR;
  try {
    std::rethrow_exception(failure);
  } catch (const MethodError& error) {
    return error.status();
  } catch (const generativeqc::Error& error) {
    return error.status();
  } catch (const std::bad_alloc&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::length_error&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  } catch (...) {
    return GENERATIVEQC_STATUS_INTERNAL_ERROR;
  }
}

/** Only an explicit seed/numerical rejection permits a cold solve. Allocation,
 * driver, capability and untyped runtime failures must propagate unchanged. */
inline bool retryable_seed_failure(generativeqc_status status,
                                   const std::exception_ptr& failure) noexcept {
  if (status != GENERATIVEQC_STATUS_NOT_CONVERGED &&
      status != GENERATIVEQC_STATUS_NUMERICAL_FAILURE &&
      status != GENERATIVEQC_STATUS_INVALID_ARGUMENT)
    return false;
  if (!failure) return true;
  try {
    std::rethrow_exception(failure);
  } catch (const MethodError&) {
    return true;
  } catch (const generativeqc::Error&) {
    return true;
  } catch (const std::invalid_argument&) {
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace generativeqc::methods
