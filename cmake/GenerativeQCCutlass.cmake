include_guard(GLOBAL)

# Header-only optional dependency. Availability does not qualify resource bounds
# or select this provider for a scientific region. Ordinary builds never search
# the filesystem for CUTLASS and never download an SDK during configuration.
function(generativeqc_configure_cutlass target)
  if(NOT GENERATIVEQC_ENABLE_CUTLASS)
    return()
  endif()
  if(NOT GENERATIVEQC_ENABLE_CUDA OR NOT GENERATIVEQC_CUDA_PROVIDER STREQUAL "nvidia")
    message(FATAL_ERROR "CUTLASS requires the NVIDIA CUDA backend")
  endif()
  if(GENERATIVEQC_PYTHON_WHEEL)
    message(FATAL_ERROR "CUTLASS wheel artifact packaging is not implemented; use a native build")
  endif()
  if(NOT GENERATIVEQC_CUTLASS_ROOT)
    set(GENERATIVEQC_CUTLASS_ROOT "$ENV{GENERATIVEQC_CUTLASS_ROOT}")
  endif()
  # Do not let a cached find_path silently select a previous SDK after ROOT
  # changes. Read and propagate the same explicit tree on every configuration.
  set(_include "${GENERATIVEQC_CUTLASS_ROOT}/include")
  if(NOT GENERATIVEQC_CUTLASS_ROOT OR
     NOT EXISTS "${_include}/cutlass/version.h" OR
     NOT EXISTS "${_include}/cutlass/gemm/device/gemm_batched.h")
    message(FATAL_ERROR "GENERATIVEQC_CUTLASS_ROOT must name a CUTLASS 3.9.2 source tree")
  endif()
  # Replacing the SDK in place must rerun admission before incremental builds.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_include}/cutlass/version.h")
  foreach(_part MAJOR MINOR PATCH)
    file(STRINGS "${_include}/cutlass/version.h" _version_line
         REGEX "^#define CUTLASS_${_part} +[0-9]+ *$")
    if(NOT _version_line MATCHES "^#define CUTLASS_${_part} +([0-9]+) *$")
      message(FATAL_ERROR "Cannot verify CUTLASS ${_part} version")
    endif()
    set(_${_part} "${CMAKE_MATCH_1}")
  endforeach()
  if(NOT "${_MAJOR}.${_MINOR}.${_PATCH}" STREQUAL "3.9.2")
    message(FATAL_ERROR "The fixed CUTLASS AOT family currently requires qualified headers 3.9.2")
  endif()
  if(NOT TARGET generativeqc_cutlass)
    add_library(generativeqc_cutlass INTERFACE)
    target_include_directories(generativeqc_cutlass SYSTEM INTERFACE "${_include}")
    target_compile_definitions(generativeqc_cutlass INTERFACE GENERATIVEQC_HAS_CUTLASS=1)
  endif()
  # Native test consumers must see the same provider macro and SDK as the
  # library; private propagation would allow different internal class layouts.
  target_link_libraries(${target} PUBLIC generativeqc_cutlass)
endfunction()
