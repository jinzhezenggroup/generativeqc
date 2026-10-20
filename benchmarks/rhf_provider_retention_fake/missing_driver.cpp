/** Call one exact-version observer export without a next provider definition. */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cusolverDn.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>

int main() {
  using CreateParams = cusolverStatus_t (*)(cusolverDnParams_t*);
  auto function = reinterpret_cast<CreateParams>(
      ::dlvsym(RTLD_DEFAULT, "cusolverDnCreateParams", "libcusolver.so.11"));
  if (!function) {
    std::printf("{\"case\":\"resolution_failure\",\"lookup\":\"missing\"}\n");
    return 2;
  }
  cusolverDnParams_t parameters{};
  const auto status = function(&parameters);
  std::printf(
      "{\"case\":\"resolution_failure\",\"lookup\":\"found\","
      "\"status\":%d,\"parameters\":%llu}\n",
      static_cast<int>(status),
      static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(parameters)));
  return 0;
}
