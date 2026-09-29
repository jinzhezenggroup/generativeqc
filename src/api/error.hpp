#ifndef GENERATIVEQC_API_ERROR_HPP
#define GENERATIVEQC_API_ERROR_HPP

#include <cstddef>
#include <string>

#include "generativeqc/generativeqc.h"

namespace generativeqc::api {

template <typename T>
bool valid_descriptor(const T* descriptor) {
  return descriptor != nullptr && descriptor->struct_size >= sizeof(T) &&
         descriptor->abi_version == GENERATIVEQC_ABI_VERSION;
}

/** Map the active C++ exception to the stable public status vocabulary. */
generativeqc_status map_exception(std::string* detail = nullptr) noexcept;

}  // namespace generativeqc::api

#endif
