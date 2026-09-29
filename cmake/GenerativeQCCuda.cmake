include_guard(GLOBAL)

macro(generativeqc_configure_cuda_backend target)
  foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(_arch STREQUAL "native" OR _arch STREQUAL "all" OR _arch STREQUAL "all-major")
      message(FATAL_ERROR
        "GENERATIVEQC MP2 AOT generation requires numeric CMAKE_CUDA_ARCHITECTURES; "
        "special value '${_arch}' is unsupported. Configure with numeric architectures, "
        "for example 75;90.")
    endif()
  endforeach()

  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/dft"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor")

  # These consumers share substantial retained numerical functions. Resolve
  # only this archive's device symbols so NVCC can coalesce those definitions
  # without coupling generated kernels or unrelated SCF/DF owners to its link.
  set(GENERATIVEQC_DIRECT_NATIVE_SOURCES
    src/scf/cuda/direct_jk_kernels.cu
    src/scf/cuda/weighted_eri_kernels.cu
    src/scf/cuda/direct_cached_tensor_kernels.cu
    src/scf/cuda/direct_schwarz_kernels.cu
    src/scf/cuda/direct_packed_fock_kernels.cu
    src/scf/cuda/direct_angular_fock.cu
    src/scf/cuda/direct_reference_force.cu
    src/scf/cuda/direct_bounded_dddd.cu
    src/scf/cuda/direct_bounded_exact_force.cu
    src/scf/cuda/direct_bounded_fallback.cu
  )
  # The resident angular-force kernels have launch-bound register ceilings.
  # NVCC 12.9 needs whole-program compilation to propagate those ceilings into
  # retained callees, including for PTX JIT. Keep this owner outside device
  # linking even when the caller enables whole-library separable compilation.
  add_library(generativeqc_direct_angular_force OBJECT
    src/scf/cuda/direct_angular_force.cu
    "${GENERATIVEQC_WEIGHTED_ERI_HEADER}"
    "${GENERATIVEQC_DIRECT_RESIDENT_PSSS_SCHEDULE_HEADER}"
    "${GENERATIVEQC_DIRECT_HIGH_ORDER_PAIR_GRADIENT_HEADER}"
    "${GENERATIVEQC_DIRECT_SOURCE_CONTRACTION_HEADER}"
    ${GENERATIVEQC_DIRECT_RECURRENCE_HEADERS}
    ${GENERATIVEQC_DIRECT_PAIR_SUPPORT_HEADERS}
    "${GENERATIVEQC_DIRECT_ORDER2_SHELL_HEADER}"
    ${GENERATIVEQC_DIRECT_CARTESIAN_CONTRACTION_HEADERS}
    "${GENERATIVEQC_DIRECT_FOCK_ACCUMULATION_HEADER}")
  target_include_directories(generativeqc_direct_angular_force PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
  target_compile_definitions(generativeqc_direct_angular_force PRIVATE
    GENERATIVEQC_BUILDING_LIBRARY=1 GENERATIVEQC_HAS_CUDA=1)
  if(NOT GENERATIVEQC_PYTHON_WHEEL)
    target_link_libraries(generativeqc_direct_angular_force PRIVATE CUDA::cudart)
  endif()
  set_target_properties(generativeqc_direct_angular_force PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "${_generativeqc_cuda_compile_architectures}"
    JOB_POOL_COMPILE generativeqc_cuda_compile)
  if(GENERATIVEQC_CUDA_FAST_COMPILE)
    target_compile_options(generativeqc_direct_angular_force PRIVATE --Ofast-compile=max)
  endif()
  target_sources(${target} PRIVATE $<TARGET_OBJECTS:generativeqc_direct_angular_force>)
  # The CuMetal toolkit shim on Apple hosts does not provide NVIDIA's device
  # linker. Keep independently compiled launch owners for that backend and for
  # other CUDA compilers; callers can also select this path explicitly.
  if(GENERATIVEQC_CUDA_DIRECT_DEVICE_LINK AND
     CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND NOT APPLE)
    add_library(generativeqc_direct_native STATIC ${GENERATIVEQC_DIRECT_NATIVE_SOURCES}
                "${GENERATIVEQC_WEIGHTED_ERI_HEADER}"
                "${GENERATIVEQC_DIRECT_HIGH_ORDER_PAIR_GRADIENT_HEADER}"
    "${GENERATIVEQC_DIRECT_SOURCE_CONTRACTION_HEADER}"
    ${GENERATIVEQC_DIRECT_RECURRENCE_HEADERS}
    ${GENERATIVEQC_DIRECT_PAIR_SUPPORT_HEADERS}
    "${GENERATIVEQC_DIRECT_ORDER2_SHELL_HEADER}"
    ${GENERATIVEQC_DIRECT_CARTESIAN_CONTRACTION_HEADERS}
    "${GENERATIVEQC_DIRECT_FOCK_ACCUMULATION_HEADER}")
    target_include_directories(generativeqc_direct_native PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_compile_definitions(generativeqc_direct_native PRIVATE
      GENERATIVEQC_BUILDING_LIBRARY=1 GENERATIVEQC_HAS_CUDA=1)
    if(NOT GENERATIVEQC_PYTHON_WHEEL)
      target_link_libraries(generativeqc_direct_native PRIVATE CUDA::cudart)
    endif()
    set_target_properties(generativeqc_direct_native PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${_generativeqc_cuda_compile_architectures}"
      JOB_POOL_COMPILE generativeqc_cuda_compile)
    if(GENERATIVEQC_CUDA_FAST_COMPILE)
      target_compile_options(generativeqc_direct_native PRIVATE --Ofast-compile=max)
    endif()
    target_link_libraries(${target} PRIVATE generativeqc_direct_native)
  else()
    target_sources(${target} PRIVATE ${GENERATIVEQC_DIRECT_NATIVE_SOURCES})
  endif()
  set_property(TARGET ${target} PROPERTY JOB_POOL_COMPILE generativeqc_cuda_compile)
  if(GENERATIVEQC_CUDA_FAST_COMPILE)
    # Explicitly opt-in: this mode is for iteration speed and must not be used
    # for release performance/resource measurements.
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--Ofast-compile=max>)
    # NVCC 12.9 fast mode over-allocates shared storage for the derivative
    # entrypoints sharing contract_shell_task (even sss exceeds 48 KiB).
    # Compile the independent class units with the normal optimizer so scratch
    # remains within the launch limit. Release builds already use this mode.
    set_property(SOURCE ${GENERATIVEQC_DF_SHELL_SOURCES}
                 APPEND PROPERTY COMPILE_OPTIONS "--Ofast-compile=0")
  endif()
  if(NOT GENERATIVEQC_CUDA_SPLIT_COMPILE_THREADS STREQUAL "1")
    # Split compilation is useful for syntax/resource experiments on native
    # direct kernels, but NVCC may make different optimization choices.
    # Keep it opt-in so production benchmark binaries remain comparable.
    set_property(SOURCE ${GENERATIVEQC_DIRECT_NATIVE_SOURCES} src/scf/cuda/direct_angular_force.cu
                 APPEND PROPERTY COMPILE_OPTIONS
                 "--split-compile=${GENERATIVEQC_CUDA_SPLIT_COMPILE_THREADS}")
  endif()
  if(GENERATIVEQC_ENABLE_AOT_SHELLS)
    if(NOT Python3_Interpreter_FOUND)
      find_package(Python3 COMPONENTS Interpreter REQUIRED)
    endif()
    set(GENERATIVEQC_AOT_SHELL_MANIFEST
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/production_shell_classes.json"
        CACHE FILEPATH "Accepted generated shell classes for production CUDA")
    set(GENERATIVEQC_AOT_GENERATED_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/generated/production_shell_kernels")

    # Class-mode development builds need a stable output list at configure
    # time.  Read names from the manifest itself rather than relying on its
    # ordering; profile resolution still decides which units contain code.
    set(GENERATIVEQC_AOT_CLASS_NAMES)
    file(READ "${GENERATIVEQC_AOT_SHELL_MANIFEST}" _generativeqc_manifest_text)
    string(REGEX MATCHALL
           "\"shell_class\"[ \t\r\n]*:[ \t\r\n]*\"[a-z]+\""
           _generativeqc_manifest_class_matches "${_generativeqc_manifest_text}")
    foreach(_generativeqc_class_match IN LISTS _generativeqc_manifest_class_matches)
      string(REGEX REPLACE
             ".*\"shell_class\"[ \t\r\n]*:[ \t\r\n]*\"([a-z]+)\".*"
             "\\1" _generativeqc_class_name "${_generativeqc_class_match}")
      list(APPEND GENERATIVEQC_AOT_CLASS_NAMES "${_generativeqc_class_name}")
    endforeach()
    list(REMOVE_DUPLICATES GENERATIVEQC_AOT_CLASS_NAMES)
    list(SORT GENERATIVEQC_AOT_CLASS_NAMES)

    # Normalize and sort the compile targets so generated behavior and source
    # assignment do not depend on CMAKE_CUDA_ARCHITECTURES list order.
    set(GENERATIVEQC_AOT_TARGET_ARCHITECTURES)
    foreach(compile_architecture IN LISTS _generativeqc_cuda_compile_architectures)
      string(REGEX REPLACE "-(real|virtual)$" "" architecture
             "${compile_architecture}")
      if(NOT architecture MATCHES "^[0-9]+$")
        message(FATAL_ERROR
                "GenerativeQC AOT requires numeric CUDA architectures, got ${architecture}")
      endif()
      list(APPEND GENERATIVEQC_AOT_TARGET_ARCHITECTURES "${architecture}")
      set(_generativeqc_aot_compile_architecture_${architecture}
          "${compile_architecture}")
    endforeach()
    list(REMOVE_DUPLICATES GENERATIVEQC_AOT_TARGET_ARCHITECTURES)
    list(SORT GENERATIVEQC_AOT_TARGET_ARCHITECTURES COMPARE NATURAL)

    set(GENERATIVEQC_AOT_GENERATOR_ARGUMENTS)
    set(_generativeqc_aot_unit_arguments
        --unit-mode "${GENERATIVEQC_AOT_UNIT_MODE}")
    if(GENERATIVEQC_AOT_UNIT_MODE STREQUAL "class")
      list(APPEND _generativeqc_aot_unit_arguments --all-class-units)
    endif()
    set(GENERATIVEQC_AOT_GENERATED_PROFILE_SOURCES)
    foreach(architecture IN LISTS GENERATIVEQC_AOT_TARGET_ARCHITECTURES)
      set(profile_architecture "sm_${architecture}")
      list(APPEND GENERATIVEQC_AOT_GENERATOR_ARGUMENTS
           --target-architecture "${profile_architecture}")
      if(GENERATIVEQC_AOT_UNIT_MODE STREQUAL "class")
        foreach(shell_class IN LISTS GENERATIVEQC_AOT_CLASS_NAMES)
          list(APPEND GENERATIVEQC_AOT_GENERATED_PROFILE_SOURCES
               "${GENERATIVEQC_AOT_GENERATED_DIRECTORY}/${profile_architecture}/generativeqc_generated_shell_sm${architecture}_${shell_class}.cu")
        endforeach()
      else()
        foreach(shard RANGE 0 ${GENERATIVEQC_AOT_LAST_SHARD})
          list(APPEND GENERATIVEQC_AOT_GENERATED_PROFILE_SOURCES
               "${GENERATIVEQC_AOT_GENERATED_DIRECTORY}/${profile_architecture}/generativeqc_generated_shell_sm${architecture}_shard_${shard}.cu")
        endforeach()
      endif()
    endforeach()
    if(NOT GENERATIVEQC_AOT_PROFILE STREQUAL "auto")
      list(APPEND GENERATIVEQC_AOT_GENERATOR_ARGUMENTS
           --profile "${GENERATIVEQC_AOT_PROFILE}")
    endif()
    foreach(profile IN LISTS GENERATIVEQC_AOT_PROFILES)
      if(profile MATCHES "^sm_([0-9]+)$")
        set(profile_architecture "${CMAKE_MATCH_1}")
        if(NOT profile_architecture IN_LIST GENERATIVEQC_AOT_TARGET_ARCHITECTURES)
          message(FATAL_ERROR
                  "AOT profile ${profile} has no matching CUDA compile target")
        endif()
        list(APPEND GENERATIVEQC_AOT_GENERATOR_ARGUMENTS
             --profile-map "${profile}=${profile}")
      elseif(NOT profile STREQUAL "")
        list(LENGTH GENERATIVEQC_AOT_TARGET_ARCHITECTURES target_count)
        if(NOT target_count EQUAL 1)
          message(FATAL_ERROR
                  "Named/portable GENERATIVEQC_AOT_PROFILES require one CUDA target; use GENERATIVEQC_AOT_PROFILE to apply one profile to all targets")
        endif()
        list(GET GENERATIVEQC_AOT_TARGET_ARCHITECTURES 0 profile_architecture)
        list(APPEND GENERATIVEQC_AOT_GENERATOR_ARGUMENTS
             --profile-map "sm_${profile_architecture}=${profile}")
      endif()
    endforeach()

    set(GENERATIVEQC_AOT_GENERATED_REGISTRY_SOURCE
        "${GENERATIVEQC_AOT_GENERATED_DIRECTORY}/generativeqc_generated_shell_registry.cu")
    set(GENERATIVEQC_AOT_GENERATED_HEADER
        "${GENERATIVEQC_AOT_GENERATED_DIRECTORY}/generativeqc_generated_shell_registry.hpp")
    generativeqc_register_generated_sources(
      GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
      OUTPUTS
        ${GENERATIVEQC_AOT_GENERATED_PROFILE_SOURCES}
        "${GENERATIVEQC_AOT_GENERATED_REGISTRY_SOURCE}"
        "${GENERATIVEQC_AOT_GENERATED_HEADER}"
      DEPENDS
        "${GENERATIVEQC_AOT_SHELL_MANIFEST}"
      ARGS
        --production-manifest "${GENERATIVEQC_AOT_SHELL_MANIFEST}"
        --output-directory "${GENERATIVEQC_AOT_GENERATED_DIRECTORY}"
        --shards "${GENERATIVEQC_AOT_SHARDS}"
        ${_generativeqc_aot_unit_arguments}
        ${GENERATIVEQC_AOT_GENERATOR_ARGUMENTS}
      COMMENT "Generating target-specific fused shell-class CUDA bundles")

    # Each object target compiles only the schedule selected for that SM. The
    # handwritten CUDA sources and host registry still use the full fatbin set.
    foreach(architecture IN LISTS GENERATIVEQC_AOT_TARGET_ARCHITECTURES)
      set(profile_architecture "sm_${architecture}")
      set(profile_sources)
      if(GENERATIVEQC_AOT_UNIT_MODE STREQUAL "class")
        foreach(shell_class IN LISTS GENERATIVEQC_AOT_CLASS_NAMES)
          list(APPEND profile_sources
               "${GENERATIVEQC_AOT_GENERATED_DIRECTORY}/${profile_architecture}/generativeqc_generated_shell_sm${architecture}_${shell_class}.cu")
        endforeach()
      else()
        foreach(shard RANGE 0 ${GENERATIVEQC_AOT_LAST_SHARD})
          list(APPEND profile_sources
               "${GENERATIVEQC_AOT_GENERATED_DIRECTORY}/${profile_architecture}/generativeqc_generated_shell_sm${architecture}_shard_${shard}.cu")
        endforeach()
      endif()
      if(GENERATIVEQC_AOT_UNIT_MODE STREQUAL "class")
        # Give development builds a directly addressable object target per
        # shell class.  A dddd edit can therefore be built with
        # ``--target generativeqc_aot_sm_120_dddd`` without compiling neighboring
        # generated classes.
        foreach(shell_class IN LISTS GENERATIVEQC_AOT_CLASS_NAMES)
          set(class_source
              "${GENERATIVEQC_AOT_GENERATED_DIRECTORY}/${profile_architecture}/generativeqc_generated_shell_sm${architecture}_${shell_class}.cu")
          set(class_target "generativeqc_aot_${profile_architecture}_${shell_class}")
          add_library(${class_target} OBJECT "${class_source}")
          target_include_directories(${class_target} PRIVATE
              "${CMAKE_CURRENT_SOURCE_DIR}/src"
              "${GENERATIVEQC_AOT_GENERATED_DIRECTORY}")
          target_compile_definitions(${class_target} PRIVATE GENERATIVEQC_HAS_CUDA=1)
          set_target_properties(${class_target} PROPERTIES
              CUDA_ARCHITECTURES
              "${_generativeqc_aot_compile_architecture_${architecture}}"
              CUDA_STANDARD 20
              CUDA_STANDARD_REQUIRED ON
              POSITION_INDEPENDENT_CODE ON
              JOB_POOL_COMPILE ${_generativeqc_aot_compile_pool})
          if(GENERATIVEQC_CUDA_FAST_COMPILE)
            target_compile_options(${class_target} PRIVATE
              $<$<COMPILE_LANGUAGE:CUDA>:--Ofast-compile=max>)
          endif()
          if(NOT GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS STREQUAL "1")
            target_compile_options(${class_target} PRIVATE
              $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--split-compile=${GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS}>)
          endif()
          target_sources(${target} PRIVATE $<TARGET_OBJECTS:${class_target}>)
        endforeach()
      else()
        add_library(generativeqc_aot_${profile_architecture} OBJECT ${profile_sources})
        target_include_directories(generativeqc_aot_${profile_architecture} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${GENERATIVEQC_AOT_GENERATED_DIRECTORY}")
        target_compile_definitions(generativeqc_aot_${profile_architecture}
                                   PRIVATE GENERATIVEQC_HAS_CUDA=1)
        set_target_properties(generativeqc_aot_${profile_architecture} PROPERTIES
            CUDA_ARCHITECTURES
            "${_generativeqc_aot_compile_architecture_${architecture}}"
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
            JOB_POOL_COMPILE ${_generativeqc_aot_compile_pool})
        if(GENERATIVEQC_CUDA_FAST_COMPILE)
          target_compile_options(generativeqc_aot_${profile_architecture} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--Ofast-compile=max>)
        endif()
        if(NOT GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS STREQUAL "1")
          target_compile_options(generativeqc_aot_${profile_architecture} PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--split-compile=${GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS}>)
        endif()
        target_sources(${target} PRIVATE
            $<TARGET_OBJECTS:generativeqc_aot_${profile_architecture}>)
      endif()
    endforeach()
    target_sources(${target} PRIVATE "${GENERATIVEQC_AOT_GENERATED_REGISTRY_SOURCE}")
    target_include_directories(${target} PRIVATE
        "${GENERATIVEQC_AOT_GENERATED_DIRECTORY}")
  else()
    target_sources(${target} PRIVATE src/scf/aot_shell_registry_stub.cpp)
  endif()
  if(GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT)
    # Preserve hashed source/asset identity separately from generator imports.
    set(_generativeqc_stationary_contract_assets
      "src/dft/stationary_gradient_cuda.cuh"
      "src/dft/grid_task_view.cuh"
      "src/dft/xc_point.hpp"
      "src/integrals/eri_geometry.hpp"
      "src/integrals/range_moments.hpp"
      "src/tensor/cuda_runtime.cuh"
      "src/runtime/bounded_workspace.hpp"
      "src/runtime/cuda_resources.cuh"
      "src/runtime/resource_cuda.cuh"
      "src/runtime/resource_ledger.hpp"
      "src/tensor/cuda_error.hpp"
      "src/tensor/metrics.hpp"
      "src/runtime/allocation_measurement.hpp"
    )
    set(_generativeqc_stationary_contract_inputs)
    foreach(_input IN LISTS _generativeqc_identity_inputs)
      if(_input MATCHES "^python/generativeqc_compiler/(common|integral|xc|dft)/.*\\.(py|json)$" OR
         _input STREQUAL "python/generativeqc_compiler/__init__.py" OR
         _input IN_LIST _generativeqc_stationary_contract_assets)
        list(APPEND _generativeqc_stationary_contract_inputs "${CMAKE_CURRENT_SOURCE_DIR}/${_input}")
      endif()
    endforeach()
    set(GENERATIVEQC_STATIONARY_AOT_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/generated/stationary_force")
    set(_generativeqc_stationary_architectures)
    foreach(_generativeqc_stationary_arch IN LISTS _generativeqc_cuda_compile_architectures)
      string(REGEX REPLACE "-(real|virtual)$" "" _generativeqc_stationary_arch
             "${_generativeqc_stationary_arch}")
      list(APPEND _generativeqc_stationary_architectures
           "sm_${_generativeqc_stationary_arch}")
    endforeach()
    list(REMOVE_DUPLICATES _generativeqc_stationary_architectures)
    set(_generativeqc_stationary_architecture_args)
    foreach(_generativeqc_stationary_arch IN LISTS _generativeqc_stationary_architectures)
      list(APPEND _generativeqc_stationary_architecture_args
           --architecture "${_generativeqc_stationary_arch}")
    endforeach()
    set(_generativeqc_stationary_compile_architecture_args)
    foreach(_generativeqc_stationary_arch IN LISTS _generativeqc_cuda_compile_architectures)
      list(APPEND _generativeqc_stationary_compile_architecture_args
           --compile-architecture "${_generativeqc_stationary_arch}")
    endforeach()
    # Component-expanded s/p/d derivatives are shared compiler output: generate
    # the bounded primitive inventory once, then device-link it into each
    # method/spin wrapper. The 23 x 16 layout is part of the versioned v3
    # artifact contract and is checked again by the Python manifest writer.
    set(_generativeqc_stationary_spd_primitive_sources)
    set(_generativeqc_stationary_spd_primitive_args)
    foreach(_generativeqc_stationary_shard RANGE 0 22)
      set(_generativeqc_stationary_shard_source
          "${GENERATIVEQC_STATIONARY_AOT_DIRECTORY}/generativeqc_stationary_spd_primitive_${_generativeqc_stationary_shard}.cu")
      list(APPEND _generativeqc_stationary_spd_primitive_sources
           "${_generativeqc_stationary_shard_source}")
      list(APPEND _generativeqc_stationary_spd_primitive_args
           --primitive-source "${_generativeqc_stationary_shard_source}")
    endforeach()
    # One command owns every shard: separate Python processes cannot reuse the
    # compiler's in-process inventory cache and would repeat all lowering work.
    generativeqc_register_generated_sources(
      NAME generativeqc_stationary_spd_primitives_codegen
      GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_stationary_force_aot.py"
      OUTPUTS ${_generativeqc_stationary_spd_primitive_sources}
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/stationary_gradient_cuda.cuh"
      ARGS
        --output "${GENERATIVEQC_STATIONARY_AOT_DIRECTORY}"
        --component-domain spd
        --all-shards
      COMMENT "Generating stationary CUDA s/p/d primitive inventory")
    add_library(generativeqc_stationary_spd_primitives OBJECT
                ${_generativeqc_stationary_spd_primitive_sources})
    add_dependencies(generativeqc_stationary_spd_primitives
                     generativeqc_stationary_spd_primitives_codegen)
    target_include_directories(generativeqc_stationary_spd_primitives PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(generativeqc_stationary_spd_primitives PRIVATE
        GENERATIVEQC_HAS_CUDA=1)
    target_compile_options(generativeqc_stationary_spd_primitives PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(generativeqc_stationary_spd_primitives PROPERTIES
        CUDA_ARCHITECTURES "${_generativeqc_cuda_compile_architectures}"
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        JOB_POOL_COMPILE generativeqc_cuda_compile)

    # Generated source weights are exact-plan-bound after #665/#689. Package
    # semilocal and admitted global-hybrid plans through one profile catalog;
    # runtime selection still verifies the exact StationaryGradientPlan identity.
    set(_generativeqc_stationary_names
        lda_rks lda_uks
        pbe_rks pbe_uks
        r2scan_rks r2scan_uks
        pbe0_rks pbe0_uks
        b3lyp_rks b3lyp_uks)
    foreach(_generativeqc_stationary_name IN LISTS _generativeqc_stationary_names)
      set(_generativeqc_stationary_source
          "${GENERATIVEQC_STATIONARY_AOT_DIRECTORY}/generativeqc_stationary_${_generativeqc_stationary_name}.cu")
      set(_generativeqc_stationary_manifest
          "${CMAKE_CURRENT_BINARY_DIR}/generativeqc_stationary_${_generativeqc_stationary_name}.json")
      generativeqc_register_generated_sources(
        NAME "generativeqc_stationary_${_generativeqc_stationary_name}_codegen"
        GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_stationary_force_aot.py"
        OUTPUTS "${_generativeqc_stationary_source}"
        DEPENDS
          "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/stationary_gradient_cuda.cuh"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/stationary_cuda.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/stationary_gradient.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/spec.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/geometry_cuda.py"
        ARGS
          --output "${_generativeqc_stationary_source}"
          --profile "${_generativeqc_stationary_name}"
          --iterations 3
        COMMENT
          "Generating ${_generativeqc_stationary_name} stationary CUDA AOT source")
      set(_generativeqc_stationary_target
          "generativeqc_stationary_${_generativeqc_stationary_name}")
      add_library(${_generativeqc_stationary_target} SHARED
                  "${_generativeqc_stationary_source}")
      add_dependencies(${_generativeqc_stationary_target}
                       "generativeqc_stationary_${_generativeqc_stationary_name}_codegen")
      target_include_directories(${_generativeqc_stationary_target} PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_compile_definitions(${_generativeqc_stationary_target} PRIVATE
          GENERATIVEQC_HAS_CUDA=1)
      target_compile_options(${_generativeqc_stationary_target} PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
          $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
      if(NOT GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS STREQUAL "1")
        target_compile_options(${_generativeqc_stationary_target} PRIVATE
          $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--split-compile=${GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS}>)
      endif()
      set_target_properties(${_generativeqc_stationary_target} PROPERTIES
          CUDA_ARCHITECTURES "${_generativeqc_cuda_compile_architectures}"
          CUDA_STANDARD 20
          CUDA_STANDARD_REQUIRED ON
          POSITION_INDEPENDENT_CODE ON
          JOB_POOL_COMPILE ${_generativeqc_aot_compile_pool}
          OUTPUT_NAME "generativeqc_stationary_${_generativeqc_stationary_name}")
      if(GENERATIVEQC_PYTHON_WHEEL)
        generativeqc_attach_cuda_implib(${_generativeqc_stationary_target})
      else()
        target_link_libraries(${_generativeqc_stationary_target} PRIVATE
                              CUDA::cudart CUDA::cublas)
      endif()
      generativeqc_register_generated_sources(
        OUTPUTS "${_generativeqc_stationary_manifest}"
        GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/write_stationary_aot_manifest.py"
        ARGS
                --library "$<TARGET_FILE:${_generativeqc_stationary_target}>"
                --source "${_generativeqc_stationary_source}"
                --output "${_generativeqc_stationary_manifest}"
                --profile "${_generativeqc_stationary_name}"
                --iterations 3
                ${_generativeqc_stationary_architecture_args}
                ${_generativeqc_stationary_compile_architecture_args}
        DEPENDS
          ${_generativeqc_stationary_target}
          "${_generativeqc_stationary_source}"
          ${_generativeqc_stationary_contract_inputs}
        COMMENT
          "Recording ${_generativeqc_stationary_name} stationary CUDA AOT identity")
      add_custom_target(
        "${_generativeqc_stationary_target}_manifest" ALL
        DEPENDS "${_generativeqc_stationary_manifest}")
      install(TARGETS ${_generativeqc_stationary_target}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
      install(FILES "${_generativeqc_stationary_manifest}"
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
      set(_generativeqc_stationary_spd_source
          "${GENERATIVEQC_STATIONARY_AOT_DIRECTORY}/generativeqc_stationary_${_generativeqc_stationary_name}_spd.cu")
      set(_generativeqc_stationary_spd_manifest
          "${CMAKE_CURRENT_BINARY_DIR}/generativeqc_stationary_${_generativeqc_stationary_name}_spd.json")
      set(_generativeqc_stationary_spd_codegen
          "generativeqc_stationary_${_generativeqc_stationary_name}_spd_codegen")
      generativeqc_register_generated_sources(
        NAME "${_generativeqc_stationary_spd_codegen}"
        GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_stationary_force_aot.py"
        OUTPUTS "${_generativeqc_stationary_spd_source}"
        DEPENDS
          "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/stationary_gradient_cuda.cuh"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/stationary_cuda.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/stationary_gradient.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/spec.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/geometry_cuda.py"
        ARGS
          --output "${_generativeqc_stationary_spd_source}"
          --profile "${_generativeqc_stationary_name}"
          --iterations 3
          --component-domain spd
        COMMENT
          "Generating ${_generativeqc_stationary_name} stationary CUDA s/p/d wrapper")
      set(_generativeqc_stationary_spd_target
          "generativeqc_stationary_${_generativeqc_stationary_name}_spd")
      add_library(${_generativeqc_stationary_spd_target} SHARED
                  "${_generativeqc_stationary_spd_source}"
                  $<TARGET_OBJECTS:generativeqc_stationary_spd_primitives>)
      add_dependencies(${_generativeqc_stationary_spd_target}
                       "${_generativeqc_stationary_spd_codegen}"
                       generativeqc_stationary_spd_primitives)
      target_include_directories(${_generativeqc_stationary_spd_target} PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_compile_definitions(${_generativeqc_stationary_spd_target} PRIVATE
          GENERATIVEQC_HAS_CUDA=1)
      target_compile_options(${_generativeqc_stationary_spd_target} PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
          $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
      set_target_properties(${_generativeqc_stationary_spd_target} PROPERTIES
          CUDA_ARCHITECTURES "${_generativeqc_cuda_compile_architectures}"
          CUDA_STANDARD 20
          CUDA_STANDARD_REQUIRED ON
          CUDA_SEPARABLE_COMPILATION ON
          CUDA_RESOLVE_DEVICE_SYMBOLS ON
          POSITION_INDEPENDENT_CODE ON
          JOB_POOL_COMPILE generativeqc_cuda_compile
          OUTPUT_NAME "generativeqc_stationary_${_generativeqc_stationary_name}_spd")
      if(GENERATIVEQC_PYTHON_WHEEL)
        generativeqc_attach_cuda_implib(${_generativeqc_stationary_spd_target})
      else()
        target_link_libraries(${_generativeqc_stationary_spd_target} PRIVATE
                              CUDA::cudart CUDA::cublas)
      endif()
      generativeqc_register_generated_sources(
        OUTPUTS "${_generativeqc_stationary_spd_manifest}"
        GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/write_stationary_aot_manifest.py"
        ARGS
                --library "$<TARGET_FILE:${_generativeqc_stationary_spd_target}>"
                --source "${_generativeqc_stationary_spd_source}"
                --output "${_generativeqc_stationary_spd_manifest}"
                --profile "${_generativeqc_stationary_name}"
                --iterations 3
                --component-domain spd
                ${_generativeqc_stationary_spd_primitive_args}
                ${_generativeqc_stationary_architecture_args}
                ${_generativeqc_stationary_compile_architecture_args}
        DEPENDS
          ${_generativeqc_stationary_spd_target}
          "${_generativeqc_stationary_spd_source}"
          ${_generativeqc_stationary_spd_primitive_sources}
          ${_generativeqc_stationary_contract_inputs}
        COMMENT
          "Recording ${_generativeqc_stationary_name} stationary CUDA s/p/d AOT identity")
      add_custom_target(
        "${_generativeqc_stationary_spd_target}_manifest" ALL
        DEPENDS "${_generativeqc_stationary_spd_manifest}")
      install(TARGETS ${_generativeqc_stationary_spd_target}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
      install(FILES "${_generativeqc_stationary_spd_manifest}"
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endforeach()
  endif()

  if(GENERATIVEQC_PYTHON_WHEEL)
    generativeqc_attach_cuda_implib(${target})
  else()
    target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
  endif()
  target_compile_definitions(${target} PUBLIC GENERATIVEQC_HAS_CUDA=1)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ${GENERATIVEQC_CUDA_SEPARABLE_COMPILATION})
endmacro()
