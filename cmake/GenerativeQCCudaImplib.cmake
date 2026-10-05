# Provider-free CUDA host-library trampolines for Python wheels.
#
# GenerativeQC's ordinary native SDK build still links CUDA::cudart/cuBLAS/cuSOLVER.
# Python wheels instead compile against CUDA headers and define the referenced
# host symbols through hidden Implib.so trampolines. Each trampoline lazily
# dlopens the reviewed CUDA-12 SONAME at runtime, so wheel construction never
# needs the proprietary provider shared libraries and the final ELF carries no
# CUDA provider DT_NEEDED entries.

function(generativeqc_write_cuda_symbol_file output_path)
  file(WRITE "${output_path}" "")
  foreach(symbol IN LISTS ARGN)
    file(APPEND "${output_path}" "${symbol}\n")
  endforeach()
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

  string(TOLOWER "\${CMAKE_SYSTEM_PROCESSOR}" processor)
  if(processor MATCHES "^(x86_64|amd64)$")
    set(implib_target x86_64)
  elseif(processor MATCHES "^(aarch64|arm64)$")
    set(implib_target aarch64)
  else()
    message(FATAL_ERROR "unsupported CUDA wheel architecture: \${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  # Derive imports from the strict final-link diagnostics instead of maintaining
  # a hand-written CUDA/cuBLAS/cuSOLVER symbol inventory. This observes the
  # object-level names after CUDA header aliases (for example *_v2) and the
  # registration symbols emitted by NVCC. The launcher only synthesizes known
  # provider families; unrelated unresolved symbols remain hard link failures.
  get_target_property(_generativeqc_existing_link_launcher \${target} RULE_LAUNCH_LINK)
  if(_generativeqc_existing_link_launcher)
    message(FATAL_ERROR
      "cannot compose CUDA wheel auto-implib with an existing RULE_LAUNCH_LINK on \${target}")
  endif()
  set(_generativeqc_implib_launcher
      "\${CMAKE_CURRENT_SOURCE_DIR}/tools/link_cuda_implib.py")
  set(_generativeqc_implib_root
      "\${CMAKE_CURRENT_SOURCE_DIR}/cmake/3rdparty/implib")
  set(_generativeqc_implib_output
      "\${CMAKE_CURRENT_BINARY_DIR}/generated/cuda_implib/\${target}")
  set(_generativeqc_link_launcher
      "\"\${Python3_EXECUTABLE}\" \"\${_generativeqc_implib_launcher}\""
      " --cc \"\${CMAKE_C_COMPILER}\""
      " --implib-root \"\${_generativeqc_implib_root}\""
      " --work-dir \"\${_generativeqc_implib_output}\""
      " --target \"\${implib_target}\" --")
  set_property(TARGET \${target} PROPERTY RULE_LAUNCH_LINK
               "\${_generativeqc_link_launcher}")

  target_include_directories(\${target} BEFORE PRIVATE
    "\${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/nvidia_host_api")
  target_include_directories(\${target} PRIVATE \${GENERATIVEQC_CUDA_TOOLKIT_INCLUDE_DIRS})
  target_link_libraries(\${target} PRIVATE \${CMAKE_DL_LIBS})
  target_link_options(\${target} PRIVATE "LINKER:-z,defs")
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
  file(MAKE_DIRECTORY "${output_dir}")
  set(symbol_file "${output_dir}/libcuda.so.symbols")
  generativeqc_write_cuda_symbol_file("${symbol_file}" cuGetErrorString cuMemGetAddressRange_v2)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cuda_implib.py"
      --base-name libcuda.so --symbol-list "${symbol_file}" --load-name libcuda.so.1
      --target "${implib_target}"
      --implib-root "${CMAKE_CURRENT_SOURCE_DIR}/cmake/3rdparty/implib"
      --outdir "${output_dir}"
    COMMAND_ERROR_IS_FATAL ANY)
  target_sources(${target} PRIVATE
    "${output_dir}/libcuda.so.tramp.S" "${output_dir}/libcuda.so.init.c")
  target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
endfunction()
