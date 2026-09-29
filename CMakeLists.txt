cmake_minimum_required(VERSION 3.24)

project(GenerativeQC VERSION 0.1.0 LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter QUIET)
find_package(Threads REQUIRED)
include(GNUInstallDirs)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerativeQCGenerated.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerativeQCGeneratedSources.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerativeQCSources.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerativeQCCuda.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerativeQCCudaImplib.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerativeQCCpuLinalg.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerativeQCGfn2Runtime.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerativeQCTests.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerativeQCSourceIdentity.cmake")

option(GENERATIVEQC_ENABLE_CUDA "Build the CUDA runtime backend" ON)
option(GENERATIVEQC_ENABLE_AOT_SHELLS
       "Build generated shell kernels in addition to the generic CUDA path" ON)
option(GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT
       "Build packaged stationary CUDA force artifacts" ON)
option(GENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT
       "Build packaged stationary CPU first-derivative artifacts" ON)
option(GENERATIVEQC_BUILD_TESTS "Build native tests" ON)
option(GENERATIVEQC_BUILD_CLI "Build the Python-free native command-line executable" ON)
option(GENERATIVEQC_PYTHON_WHEEL
       "Install one unversioned shared library for a Python wheel" OFF)
option(GENERATIVEQC_BUNDLE_XTB_OPENBLAS
       "Bundle the reviewed private LP64 OpenBLAS provider for GFN2 wheels"
       ${GENERATIVEQC_PYTHON_WHEEL})
option(GENERATIVEQC_CUDA_SEPARABLE_COMPILATION
       "Enable CUDA relocatable device code and the device-link stage" OFF)
option(GENERATIVEQC_CUDA_DIRECT_DEVICE_LINK
       "Share native direct numerical code through a bounded NVIDIA device link" ON)
option(GENERATIVEQC_CUDA_FAST_COMPILE
       "Use compiler-oriented CUDA options (never for release measurements)" OFF)
option(GENERATIVEQC_ENABLE_CXX_PCH
       "Precompile stable standard-library headers for host C++ compilation" OFF)
set(GENERATIVEQC_COMPILER_CACHE "auto" CACHE STRING
    "Compiler cache launcher: auto, off, or an executable name/path")
set(GENERATIVEQC_CUDA_ARCHITECTURES "" CACHE STRING
    "CUDA architectures shipped by GenerativeQC; defaults to CMAKE_CUDA_ARCHITECTURES")
set(GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES "" CACHE STRING
    "NVCC architecture values, retaining 120-real/120-virtual intent")
set(GENERATIVEQC_CUDA_COMPILE_JOBS "2" CACHE STRING
    "Maximum concurrent native CUDA compiler jobs in the Ninja job pool")
set(GENERATIVEQC_AOT_COMPILE_JOBS "" CACHE STRING
    "Optional independent generated-AOT CUDA job limit; empty shares the native CUDA pool")
set(GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS "1" CACHE STRING
    "NVCC split-compile threads for generated AOT kernels; 1 preserves production code generation")
set(GENERATIVEQC_CUDA_SPLIT_COMPILE_THREADS "1" CACHE STRING
    "NVCC split-compile threads for native direct kernels; 1 preserves production code generation")
set(GENERATIVEQC_CUDA_PROVIDER "nvidia" CACHE STRING
    "CUDA execution provider identity: nvidia or cumetal")
set_property(CACHE GENERATIVEQC_CUDA_PROVIDER PROPERTY STRINGS nvidia cumetal)
set(GENERATIVEQC_AOT_PROFILE "auto" CACHE STRING
    "AOT profile for every target: strict auto, explicit portable, sm_XX, or a named profile")
set(GENERATIVEQC_AOT_PROFILES "" CACHE STRING
    "Optional explicit list of AOT profiles, independent of architecture order")
set(GENERATIVEQC_AOT_SHARDS "8" CACHE STRING
    "Number of independently compiled CUDA shards per AOT architecture")
set(GENERATIVEQC_AOT_UNIT_MODE "stable-shards" CACHE STRING
    "Generated source layout: stable-shards (release) or class (development)")
set_property(CACHE GENERATIVEQC_AOT_UNIT_MODE PROPERTY STRINGS stable-shards class)

# Repeated CUDA experiments frequently return to byte-identical source after an
# A/B test or branch switch.  Reuse those expensive objects automatically when
# a supported cache is installed, while preserving explicit user launchers.
unset(_generativeqc_compiler_cache)
if(GENERATIVEQC_COMPILER_CACHE STREQUAL "auto")
  find_program(_generativeqc_compiler_cache NAMES sccache ccache NO_CACHE)
elseif(NOT GENERATIVEQC_COMPILER_CACHE STREQUAL "off"
       AND NOT GENERATIVEQC_COMPILER_CACHE STREQUAL "none"
       AND NOT GENERATIVEQC_COMPILER_CACHE STREQUAL "")
  if(IS_ABSOLUTE "${GENERATIVEQC_COMPILER_CACHE}")
    if(NOT EXISTS "${GENERATIVEQC_COMPILER_CACHE}")
      message(FATAL_ERROR
              "GENERATIVEQC_COMPILER_CACHE does not exist: ${GENERATIVEQC_COMPILER_CACHE}")
    endif()
    set(_generativeqc_compiler_cache "${GENERATIVEQC_COMPILER_CACHE}")
  else()
    find_program(_generativeqc_compiler_cache
                 NAMES "${GENERATIVEQC_COMPILER_CACHE}" NO_CACHE)
    if(NOT _generativeqc_compiler_cache)
      message(FATAL_ERROR
              "Compiler cache executable not found: ${GENERATIVEQC_COMPILER_CACHE}")
    endif()
  endif()
endif()
if(_generativeqc_compiler_cache)
  get_filename_component(_generativeqc_compiler_cache_name
                         "${_generativeqc_compiler_cache}" NAME)
  set(_generativeqc_compiler_cache_launcher "${_generativeqc_compiler_cache}")
  if(_generativeqc_compiler_cache_name STREQUAL "ccache")
    # Generated CUDA/AOT sources live below the build tree. Without a stable
    # base directory, their absolute paths make byte-identical objects miss
    # when the same source is rebuilt in another worktree or build directory.
    set(_generativeqc_compiler_cache_launcher
        "${CMAKE_COMMAND};-E;env;CCACHE_BASEDIR=${CMAKE_BINARY_DIR};${_generativeqc_compiler_cache}")
  endif()
  if(NOT CMAKE_CXX_COMPILER_LAUNCHER)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${_generativeqc_compiler_cache_launcher}")
  endif()
  if(NOT CMAKE_CUDA_COMPILER_LAUNCHER)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${_generativeqc_compiler_cache_launcher}")
  endif()
  message(STATUS "GenerativeQC compiler cache: ${_generativeqc_compiler_cache}")
endif()

if(NOT Python3_Interpreter_FOUND)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

if(NOT GENERATIVEQC_CUDA_SPLIT_COMPILE_THREADS MATCHES "^[0-9]+$")
  message(FATAL_ERROR
          "GENERATIVEQC_CUDA_SPLIT_COMPILE_THREADS must be a non-negative integer")
endif()
if(NOT GENERATIVEQC_AOT_SHARDS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "GENERATIVEQC_AOT_SHARDS must be a positive integer")
endif()
if(NOT GENERATIVEQC_CUDA_COMPILE_JOBS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "GENERATIVEQC_CUDA_COMPILE_JOBS must be a positive integer")
endif()
if(NOT GENERATIVEQC_AOT_COMPILE_JOBS STREQUAL ""
   AND NOT GENERATIVEQC_AOT_COMPILE_JOBS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR
          "GENERATIVEQC_AOT_COMPILE_JOBS must be empty or a positive integer")
endif()
if(NOT GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR
          "GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS must be a positive integer")
endif()
if(NOT GENERATIVEQC_CUDA_PROVIDER STREQUAL "nvidia"
   AND NOT GENERATIVEQC_CUDA_PROVIDER STREQUAL "cumetal")
  message(FATAL_ERROR "GENERATIVEQC_CUDA_PROVIDER must be nvidia or cumetal")
endif()
if(NOT GENERATIVEQC_AOT_UNIT_MODE STREQUAL "stable-shards"
   AND NOT GENERATIVEQC_AOT_UNIT_MODE STREQUAL "class")
  message(FATAL_ERROR "GENERATIVEQC_AOT_UNIT_MODE must be stable-shards or class")
endif()
math(EXPR GENERATIVEQC_AOT_LAST_SHARD "${GENERATIVEQC_AOT_SHARDS} - 1")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(GENERATIVEQC_ENABLE_CUDA)
  # Resolve only caller-supplied target sources before CUDA language enablement.
  # CMake may otherwise initialize CMAKE_CUDA_ARCHITECTURES from NVCC's legacy
  # compiler default, which must not become an implicit GenerativeQC target.
  set(_generativeqc_cuda_requested_architectures "")
  if(GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES)
    set(_generativeqc_cuda_requested_architectures
        ${GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES})
  elseif(GENERATIVEQC_CUDA_ARCHITECTURES)
    set(_generativeqc_cuda_requested_architectures ${GENERATIVEQC_CUDA_ARCHITECTURES})
  elseif(DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_CUDA_ARCHITECTURES)
    set(_generativeqc_cuda_requested_architectures ${CMAKE_CUDA_ARCHITECTURES})
  elseif(DEFINED ENV{CUDAARCHS} AND NOT "$ENV{CUDAARCHS}" STREQUAL "")
    set(_generativeqc_cuda_requested_architectures "$ENV{CUDAARCHS}")
  endif()

  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT _generativeqc_cuda_requested_architectures)
      message(FATAL_ERROR
        "CUDA target architecture is required; set GENERATIVEQC_CUDA_ARCHITECTURES, "
        "CMAKE_CUDA_ARCHITECTURES, GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES, or CUDAARCHS")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${_generativeqc_cuda_requested_architectures})
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Preserve NVCC's real/virtual suffixes for compilation. A separate
    # numeric profile identity is derived below for manifest and cache paths.
    set(_generativeqc_cuda_compile_architectures ${CMAKE_CUDA_ARCHITECTURES})
    set(_generativeqc_aot_compile_pool generativeqc_cuda_compile)
    set(_generativeqc_cuda_job_pools
        generativeqc_cuda_compile=${GENERATIVEQC_CUDA_COMPILE_JOBS})
    if(NOT GENERATIVEQC_AOT_COMPILE_JOBS STREQUAL "")
      list(APPEND _generativeqc_cuda_job_pools
           generativeqc_aot_compile=${GENERATIVEQC_AOT_COMPILE_JOBS})
      set(_generativeqc_aot_compile_pool generativeqc_aot_compile)
    endif()
    set_property(GLOBAL PROPERTY JOB_POOLS ${_generativeqc_cuda_job_pools})
    # XsyevBatched is the capture-safe production eigensolver for realistic
    # AO buckets and is available in the supported CUDA 12.9 toolkit. Native
    # SDK builds link the toolkit providers normally. Python wheels instead
    # need only compiler-owned headers: hidden Implib.so trampolines resolve
    # the provider SONAMEs lazily at runtime.
    if(GENERATIVEQC_PYTHON_WHEEL)
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.9)
        message(FATAL_ERROR "GenerativeQC CUDA wheels require CUDA 12.9 or newer")
      endif()
      set(CMAKE_CUDA_RUNTIME_LIBRARY None)
      enable_language(C ASM)
      set(GENERATIVEQC_CUDA_TOOLKIT_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
      foreach(_generativeqc_cuda_header IN ITEMS cuda_runtime_api.h library_types.h)
        set(_generativeqc_cuda_header_found OFF)
        foreach(_generativeqc_cuda_include IN LISTS GENERATIVEQC_CUDA_TOOLKIT_INCLUDE_DIRS)
          if(EXISTS "${_generativeqc_cuda_include}/${_generativeqc_cuda_header}")
            set(_generativeqc_cuda_header_found ON)
            break()
          endif()
        endforeach()
        if(NOT _generativeqc_cuda_header_found)
          message(FATAL_ERROR "CUDA compiler include set lacks ${_generativeqc_cuda_header}")
        endif()
      endforeach()
    else()
      if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT GENERATIVEQC_CUDA_PROVIDER STREQUAL "cumetal")
        # Native GFN2 resolves its driver metadata calls lazily as well. Enable
        # import-source languages here, not inside a function-local scope.
        enable_language(C ASM)
      endif()
      find_package(CUDAToolkit 12.9 REQUIRED)
      set(GENERATIVEQC_CUDA_TOOLKIT_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    endif()
  else()
    message(WARNING "CUDA compiler not found; building the CPU reference backend only")
    set(GENERATIVEQC_ENABLE_CUDA OFF)
  endif()
endif()

# Scoped GenerativeQC-owned GFN2 CPU runtime is registered below.
add_library(generativeqc SHARED)
target_link_libraries(generativeqc PRIVATE Threads::Threads)
generativeqc_configure_cpu_linalg(generativeqc)
generativeqc_add_gfn2_runtime(generativeqc)
generativeqc_add_component_sources(generativeqc)
generativeqc_register_host_generated_sources(generativeqc)

# A local tuned library may replace this build only when its scientific source
# and generator are identical. Paths and Git metadata are deliberately excluded
# so the same installed build can be reused across homogeneous cluster nodes.
#
# Keep inventory discovery at configure time so additions/removals regenerate
# the build graph, but hash file contents only when the generated identity
# header is actually out of date. Ordinary source edits therefore do not force
# a CMake reconfigure before compilation can resume.
generativeqc_collect_source_identity_inputs(_generativeqc_identity_inputs)
# A removed dependency cannot make the remaining inputs newer than the header.
# Persist membership separately so deletions, renames and backdated additions
# invalidate identity too. file(GENERATE) preserves mtime when the list is unchanged.
set(_generativeqc_identity_inventory
    "${CMAKE_CURRENT_BINARY_DIR}/generated/build_identity_inputs.txt")
file(GENERATE OUTPUT "${_generativeqc_identity_inventory}"
     CONTENT "${_generativeqc_identity_inputs}\n")
set(_generativeqc_identity_dependencies)
foreach(_generativeqc_identity_path IN LISTS _generativeqc_identity_inputs)
  list(APPEND _generativeqc_identity_dependencies
       "${CMAKE_CURRENT_SOURCE_DIR}/${_generativeqc_identity_path}")
endforeach()
set(GENERATIVEQC_TUNING_RELEASE_BUILD OFF)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(GENERATIVEQC_TUNING_RELEASE_BUILD ON)
endif()
set(GENERATIVEQC_BUILD_IDENTITY_HEADER
    "${CMAKE_CURRENT_BINARY_DIR}/generated/build_identity.hpp")
generativeqc_register_generated_sources(
  NAME generativeqc_build_identity_codegen
  TARGET generativeqc
  GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_build_identity.py"
  OUTPUTS "${GENERATIVEQC_BUILD_IDENTITY_HEADER}"
  DEPENDS
    "${_generativeqc_identity_inventory}"
    ${_generativeqc_identity_dependencies}
    "${CMAKE_CURRENT_SOURCE_DIR}/src/api/build_identity.hpp.in"
  ARGS
    --source-root "${CMAKE_CURRENT_SOURCE_DIR}"
    --manifest "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerativeQCSourceIdentity.json"
    --template "${CMAKE_CURRENT_SOURCE_DIR}/src/api/build_identity.hpp.in"
    --output "${GENERATIVEQC_BUILD_IDENTITY_HEADER}"
    --cuda-fast-compile "${GENERATIVEQC_CUDA_FAST_COMPILE}"
    --release-build "${GENERATIVEQC_TUNING_RELEASE_BUILD}"
  COMMENT "Generating GenerativeQC build identity")
target_include_directories(generativeqc PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

target_include_directories(generativeqc
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(GENERATIVEQC_ENABLE_CXX_PCH)
  target_precompile_headers(generativeqc PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/src/pch.hpp>")
endif()

target_compile_definitions(generativeqc PRIVATE GENERATIVEQC_BUILDING_LIBRARY=1)
if(GENERATIVEQC_CUDA_PROVIDER STREQUAL "cumetal")
  target_compile_definitions(generativeqc PRIVATE GENERATIVEQC_CUDA_PROVIDER_CUMETAL=1)
else()
  target_compile_definitions(generativeqc PRIVATE GENERATIVEQC_CUDA_PROVIDER_CUMETAL=0)
endif()
target_compile_options(generativeqc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)

if(GENERATIVEQC_ENABLE_CUDA)
  # A single real architecture may use that architecture's measured host-side
  # candidate schedule. Fat binaries and virtual/PTX-only targets keep host
  # scheduling portable: their active runtime device is not a compile-time
  # property of shared host code and may differ from the PTX target.
  list(LENGTH _generativeqc_cuda_compile_architectures _generativeqc_cuda_architecture_count)
  if(_generativeqc_cuda_architecture_count EQUAL 1)
    list(GET _generativeqc_cuda_compile_architectures 0 _generativeqc_cuda_profile_architecture)
    if(NOT _generativeqc_cuda_profile_architecture MATCHES "-virtual$")
      string(REGEX REPLACE "-real$" "" _generativeqc_cuda_profile_architecture
             "${_generativeqc_cuda_profile_architecture}")
      target_compile_definitions(
        generativeqc PRIVATE GENERATIVEQC_CUDA_PROFILE_ARCHITECTURE=${_generativeqc_cuda_profile_architecture})
    endif()
  endif()
  generativeqc_register_cuda_generated_sources(generativeqc)
  generativeqc_configure_cuda_backend(generativeqc)
else()
  target_compile_definitions(generativeqc PUBLIC GENERATIVEQC_HAS_CUDA=0)
endif()

set_target_properties(generativeqc PROPERTIES OUTPUT_NAME generativeqc)
if(NOT GENERATIVEQC_PYTHON_WHEEL)
  # Native SDK installs retain the normal SONAME chain. Wheel archives often
  # dereference those symlinks and would otherwise store the large CUDA shared
  # object multiple times, so Python wheels install one unversioned library.
  set_target_properties(generativeqc PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
  )
endif()

if(GENERATIVEQC_BUILD_CLI)
  add_executable(generativeqc_cli src/cli/main.cpp)
  target_link_libraries(generativeqc_cli PRIVATE generativeqc)
  target_include_directories(generativeqc_cli PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_compile_definitions(generativeqc_cli PRIVATE GENERATIVEQC_CLI_VERSION="${PROJECT_VERSION}")
  target_compile_options(generativeqc_cli PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
  set_target_properties(generativeqc_cli PROPERTIES OUTPUT_NAME generativeqc)
  if(APPLE)
    set_target_properties(generativeqc_cli PROPERTIES
      INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
  elseif(UNIX)
    set_target_properties(generativeqc_cli PROPERTIES
      INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
  endif()
endif()

if(GENERATIVEQC_BUILD_TESTS)
  generativeqc_add_native_tests()
  if(GENERATIVEQC_BUILD_CLI)
    add_test(NAME generativeqc_cli_methods
             COMMAND generativeqc_cli methods --json)
    set_tests_properties(generativeqc_cli_methods PROPERTIES
                         PASS_REGULAR_EXPRESSION "gfn2-xtb")
    if(GENERATIVEQC_XTB_CPU_LINALG_LIBRARY OR GENERATIVEQC_BUNDLE_XTB_OPENBLAS)
      add_test(NAME generativeqc_cli_gfn2_cpu
               COMMAND generativeqc_cli run
                       "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/gfn2_h3_plus.xyz"
                       --method gfn2-xtb --backend cpu --charge 1 --units bohr
                       --forces --json)
      set_tests_properties(generativeqc_cli_gfn2_cpu PROPERTIES
                           PASS_REGULAR_EXPRESSION "energy_hartree")
    endif()
  endif()
endif()

generativeqc_add_codegen_pilot()

install(TARGETS generativeqc
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
if(GENERATIVEQC_BUILD_CLI)
  install(TARGETS generativeqc_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
