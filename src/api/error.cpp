#include "api/error.hpp"

#include <exception>
#include <new>
#include <stdexcept>

#include "generativeqc/generativeqc.hpp"
#include "methods/method.hpp"

namespace generativeqc::api {

generativeqc_status map_exception(std::string* detail) noexcept {
  try {
    throw;
  } catch (const methods::MethodError& error) {
    if (detail != nullptr) *detail = error.what();
    return error.status();
  } catch (const generativeqc::Error& error) {
    if (detail != nullptr) *detail = error.what();
    return error.status();
  } catch (const std::bad_alloc& error) {
    if (detail != nullptr) *detail = error.what();
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument& error) {
    if (detail != nullptr) *detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    if (detail != nullptr) *detail = error.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  } catch (...) {
    if (detail != nullptr) *detail = "unknown internal exception";
    return GENERATIVEQC_STATUS_INTERNAL_ERROR;
  }
}

}  // namespace generativeqc::api
