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

function(generativeqc_attach_cuda_implib target)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "GenerativeQC provider-free CUDA wheels currently require Linux ELF")
  endif()
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" processor)
  if(processor MATCHES "^(x86_64|amd64)$")
    set(implib_target x86_64)
  elseif(processor MATCHES "^(aarch64|arm64)$")
    set(implib_target aarch64)
  else()
    message(FATAL_ERROR "unsupported CUDA wheel architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  set(implib_root "${CMAKE_CURRENT_SOURCE_DIR}/cmake/3rdparty/implib")
  set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/cuda_implib")

  # Object-level names after CUDA header macro expansion. Keep these curated:
  # -z defs on the final wheel target turns any newly introduced CUDA host API
  # into a link failure instead of silently adding a provider dependency.
  set(GENERATIVEQC_CUDART_SYMBOLS
    __cudaInitModule
    __cudaPopCallConfiguration
    __cudaPushCallConfiguration
    __cudaRegisterFatBinary
    __cudaRegisterFatBinaryEnd
    __cudaRegisterFunction
    __cudaRegisterVar
    __cudaUnregisterFatBinary
    cudaDeviceGetAttribute
    cudaDeviceGetLimit
    cudaDeviceSetLimit
    cudaDriverGetVersion
    cudaEventCreate
    cudaEventCreateWithFlags
    cudaEventDestroy
    cudaEventElapsedTime
    cudaEventRecord
    cudaEventSynchronize
    cudaFree
    cudaFreeAsync
    cudaFreeHost
    cudaFuncGetAttributes
    cudaGetDevice
    cudaGetDeviceCount
    cudaGetDeviceProperties_v2
    cudaGetErrorString
    cudaGetLastError
    cudaGraphDestroy
    cudaGraphExecDestroy
    cudaGraphGetNodes
    cudaGraphInstantiate
    cudaGraphLaunch
    cudaGraphUpload
    cudaLaunchKernel
    cudaMalloc
    cudaMallocAsync
    cudaMallocHost
    cudaMemGetInfo
    cudaMemcpy
    cudaMemcpy2DAsync
    cudaMemcpyAsync
    cudaMemsetAsync
    cudaOccupancyMaxActiveBlocksPerMultiprocessor
    cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
    cudaPeekAtLastError
    cudaPointerGetAttributes
    cudaRuntimeGetVersion
    cudaSetDevice
    cudaStreamBeginCapture
    cudaStreamCreateWithFlags
    cudaStreamDestroy
    cudaStreamEndCapture
    cudaStreamGetFlags
    cudaStreamIsCapturing
    cudaStreamSynchronize
    cudaStreamWaitEvent
  )
  set(GENERATIVEQC_CUBLAS_SYMBOLS
    cublasCreate_v2
    cublasDaxpy_v2
    cublasDcopy_v2
    cublasDdot_v2
    cublasDnrm2_v2
    cublasDscal_v2
    cublasDestroy_v2
    cublasDgeam
    cublasDgemmStridedBatched
    cublasDgemm_v2
    cublasDgemv_v2
    cublasDsyr2k_v2
    cublasDsyrk_v2
    cublasGetPointerMode_v2
    cublasGetProperty
    cublasGetStream_v2
    cublasGetVersion_v2
    cublasSetMathMode
    cublasSetPointerMode_v2
    cublasSetStream_v2
    cublasSetWorkspace_v2
    cublasSgemm_v2
    cublasSgemmStridedBatched
  )
  set(GENERATIVEQC_CUSOLVER_SYMBOLS
    cusolverDnCreate
    cusolverDnCreateParams
    cusolverDnCreateSyevjInfo
    cusolverDnDestroy
    cusolverDnDestroyParams
    cusolverDnDestroySyevjInfo
    cusolverDnDsyevjBatched
    cusolverDnDsyevjBatched_bufferSize
    cusolverDnSetStream
    cusolverDnXsyevBatched
    cusolverDnXsyevBatched_bufferSize
    cusolverDnXsyevd
    cusolverDnXsyevd_bufferSize
    cusolverDnXsyevjSetMaxSweeps
    cusolverDnXsyevjSetSortEig
    cusolverDnXsyevjSetTolerance
    cusolverGetProperty
  )

  set(bases libcudart.so libcublas.so libcusolver.so)
  set(sonames libcudart.so.12 libcublas.so.12 libcusolver.so.11)
  set(symbol_sets GENERATIVEQC_CUDART_SYMBOLS GENERATIVEQC_CUBLAS_SYMBOLS GENERATIVEQC_CUSOLVER_SYMBOLS)
  foreach(base soname symbol_set IN ZIP_LISTS bases sonames symbol_sets)
    generativeqc_register_cuda_implib_codegen(
      generated_sources codegen_target "${output_dir}" "${base}" "${soname}"
      "${implib_target}" "${implib_root}" ${${symbol_set}})
    add_dependencies(${target} ${codegen_target})
    target_sources(${target} PRIVATE ${generated_sources})
  endforeach()

  target_include_directories(${target} BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/nvidia_host_api")
  target_include_directories(${target} PRIVATE ${GENERATIVEQC_CUDA_TOOLKIT_INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
  target_link_options(${target} PRIVATE "LINKER:-z,defs")
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
