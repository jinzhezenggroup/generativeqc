# Provider-free CUDA host-library trampolines for Python wheels.
#
# GenerativeQC's ordinary native SDK build still links CUDA::cudart/cuBLAS/cuSOLVER.
# Python wheels instead compile against CUDA headers and define the referenced
# host symbols through hidden Implib.so trampolines. Each trampoline lazily
# dlopens the reviewed CUDA-12 SONAME at runtime, so wheel construction never
# needs the proprietary provider shared libraries and the final ELF carries no
# CUDA provider DT_NEEDED entries.

include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/GenerativeQCGenerated.cmake")

function(generativeqc_register_cuda_implib_codegen
         sources_variable codegen_target_variable output_dir base load_name implib_target implib_root)
  set(symbols ${ARGN})
  string(REGEX REPLACE "[^A-Za-z0-9_]" "_" implib_key "${base}")
  set(codegen_target "generativeqc_cuda_implib_${implib_key}_${implib_target}_codegen")
  set(outputs
      "${output_dir}/${base}.tramp.S"
      "${output_dir}/${base}.init.c")

  # One build-graph owner per provider/architecture. Multiple DSOs can consume
  # the same generated sources without rerunning Python during configure.
  if(NOT TARGET ${codegen_target})
    file(MAKE_DIRECTORY "${output_dir}")
    set(symbol_file "${output_dir}/${base}.symbols")
    set(symbol_text "")
    foreach(symbol IN LISTS symbols)
      string(APPEND symbol_text "${symbol}\n")
    endforeach()
    # Keep the tiny curated symbol inventory in CMake's generation phase while
    # deferring Python/template expansion to the normal incremental build graph.
    file(GENERATE OUTPUT "${symbol_file}" CONTENT "${symbol_text}")

    generativeqc_register_generated_sources(
      NAME ${codegen_target}
      GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cuda_implib.py"
      OUTPUTS ${outputs}
      DEPENDS
        "${symbol_file}"
        "${implib_root}/arch/${implib_target}/config.ini"
        "${implib_root}/arch/${implib_target}/table.S.tpl"
        "${implib_root}/arch/${implib_target}/trampoline.S.tpl"
        "${implib_root}/arch/common/init.c.tpl"
      ARGS
        --base-name "${base}"
        --symbol-list "${symbol_file}"
        --load-name "${load_name}"
        --target "${implib_target}"
        --implib-root "${implib_root}"
        --outdir "${output_dir}"
      COMMENT "Generating ${load_name} lazy CUDA imports")
  endif()

  set(${sources_variable} "${outputs}" PARENT_SCOPE)
  set(${codegen_target_variable} "${codegen_target}" PARENT_SCOPE)
endfunction()

function(generativeqc_ensure_cuda_wheel_import_interface)
  if(TARGET generativeqc_cuda_wheel_imports)
    return()
  endif()

  add_library(generativeqc_cuda_wheel_imports INTERFACE)
  target_include_directories(generativeqc_cuda_wheel_imports INTERFACE
    "${PROJECT_SOURCE_DIR}/src/runtime/nvidia_host_api"
    ${GENERATIVEQC_CUDA_TOOLKIT_INCLUDE_DIRS})
  target_link_libraries(generativeqc_cuda_wheel_imports INTERFACE ${CMAKE_DL_LIBS})
  target_link_options(generativeqc_cuda_wheel_imports INTERFACE "LINKER:-z,defs")
endfunction()

function(generativeqc_attach_cuda_implib target)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "GenerativeQC provider-free CUDA wheels currently require Linux ELF")
  endif()
  if(NOT CMAKE_GENERATOR MATCHES "Ninja|Makefiles")
    message(FATAL_ERROR
      "GenerativeQC provider-free CUDA wheels require a Ninja or Makefile generator")
  endif()
  if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Python is required to generate provider-free CUDA wheel imports")
  endif()
  if(NOT CMAKE_C_COMPILER)
    message(FATAL_ERROR "A C compiler is required to generate provider-free CUDA wheel imports")
  endif()

  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" processor)
  if(processor MATCHES "^(x86_64|amd64)$")
    set(implib_target x86_64)
  elseif(processor MATCHES "^(aarch64|arm64)$")
    set(implib_target aarch64)
  else()
    message(FATAL_ERROR "unsupported CUDA wheel architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  # Derive imports from the strict final-link diagnostics instead of maintaining
  # a hand-written CUDA/cuBLAS/cuSOLVER symbol inventory. NVIDIA CUDA host-link
  # rules bypass CMake's language linker launchers, so use the supported CXX
  # final host-link rule even for CUDA-only targets. CUDA source compilation
  # and separable/device-link settings remain owned by the original target.
  get_target_property(_generativeqc_existing_link_launcher
                     ${target} CXX_LINKER_LAUNCHER)
  if(_generativeqc_existing_link_launcher)
    message(FATAL_ERROR
      "cannot compose CUDA wheel auto-implib with an existing "
      "CXX_LINKER_LAUNCHER on ${target}")
  endif()
  set_property(TARGET ${target} PROPERTY LINKER_LANGUAGE CXX)

  set(_generativeqc_implib_launcher
      "${PROJECT_SOURCE_DIR}/tools/link_cuda_implib.py")
  set(_generativeqc_implib_root
      "${PROJECT_SOURCE_DIR}/cmake/3rdparty/implib")
  set(_generativeqc_implib_output
      "${PROJECT_BINARY_DIR}/generated/cuda_implib/${target}")
  set(_generativeqc_link_launcher
      "${Python3_EXECUTABLE}"
      "${_generativeqc_implib_launcher}"
      --cc "${CMAKE_C_COMPILER}"
      --implib-root "${_generativeqc_implib_root}"
      --work-dir "${_generativeqc_implib_output}"
      --target "${implib_target}"
      --)
  set_property(TARGET ${target} PROPERTY
               CXX_LINKER_LAUNCHER "${_generativeqc_link_launcher}")

  generativeqc_ensure_cuda_wheel_import_interface()
  target_link_libraries(${target} PRIVATE generativeqc_cuda_wheel_imports)
endfunction()

# Native GFN2 needs two driver metadata queries, but loading a CUDA-enabled
# library for CPU preflight must not require an installed NVIDIA driver.
function(generativeqc_attach_cuda_driver_implib target)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Lazy CUDA driver imports require Linux ELF")
  endif()
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" processor)
  if(processor MATCHES "^(x86_64|amd64)$")
    set(implib_target x86_64)
  elseif(processor MATCHES "^(aarch64|arm64)$")
    set(implib_target aarch64)
  else()
    message(FATAL_ERROR "Unsupported CUDA driver import architecture")
  endif()
  set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/cuda_driver_implib")
  set(implib_root "${CMAKE_CURRENT_SOURCE_DIR}/cmake/3rdparty/implib")
  generativeqc_register_cuda_implib_codegen(
    generated_sources codegen_target "${output_dir}" libcuda.so libcuda.so.1
    "${implib_target}" "${implib_root}" cuGetErrorString cuMemGetAddressRange_v2)
  add_dependencies(${target} ${codegen_target})
  target_sources(${target} PRIVATE ${generated_sources})
  target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
endfunction()
