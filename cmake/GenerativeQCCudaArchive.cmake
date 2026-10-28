include_guard(GLOBAL)

# Return a host-link copy of an archive whose device symbols are already closed.
# Keep the original target/file intact for incremental or downstream device
# links. nvprune rewrites CUDA registration wrappers as well as image payloads;
# deleting __nv_relfatbin with a linker script/objcopy is not equivalent.
function(generativeqc_cuda_host_archive target output)
  set(${output} "${target}" PARENT_SCOPE)
  if(NOT GENERATIVEQC_PRUNE_CUDA_ARCHIVES OR
     NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" OR
     NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    return()
  endif()
  get_target_property(kind ${target} TYPE)
  get_target_property(resolved ${target} CUDA_RESOLVE_DEVICE_SYMBOLS)
  if(NOT kind STREQUAL "STATIC_LIBRARY" OR NOT resolved)
    message(FATAL_ERROR "CUDA host archive pruning requires a device-linked static library: ${target}")
  endif()
  if(TARGET ${target}_host)
    set(${output} "${target}_host" PARENT_SCOPE)
    return()
  endif()

  # Select the tool from the active compiler's toolkit, not another CUDA on PATH.
  get_filename_component(compiler_directory "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  find_program(GENERATIVEQC_NVPRUNE nvprune HINTS "${compiler_directory}" NO_DEFAULT_PATH)
  if(NOT GENERATIVEQC_NVPRUNE)
    message(STATUS "nvprune unavailable; retaining complete CUDA archive ${target}")
    return()
  endif()
  execute_process(COMMAND "${GENERATIVEQC_NVPRUNE}" --version
                  RESULT_VARIABLE status OUTPUT_QUIET ERROR_QUIET TIMEOUT 10)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Cannot execute selected nvprune: ${GENERATIVEQC_NVPRUNE}")
  endif()

  # Preserve every requested executable cubin and PTX image, including the
  # distinction between real-only, virtual-only and combined architectures.
  get_target_property(architectures ${target} CUDA_ARCHITECTURES)
  set(codes)
  foreach(architecture IN LISTS architectures)
    if(NOT architecture MATCHES "^([0-9]+[af]?)(-(real|virtual))?$")
      message(STATUS "Retaining ${target}: nvprune requires explicit numeric CUDA architectures")
      return()
    endif()
    set(number "${CMAKE_MATCH_1}")
    set(mode "${CMAKE_MATCH_3}")
    if(NOT mode STREQUAL "virtual")
      list(APPEND codes "arch=compute_${number},code=sm_${number}")
    endif()
    if(NOT mode STREQUAL "real")
      list(APPEND codes "arch=compute_${number},code=compute_${number}")
    endif()
  endforeach()
  if(NOT codes)
    message(STATUS "Retaining ${target}: no explicit CUDA architecture images")
    return()
  endif()
  list(REMOVE_DUPLICATES codes)
  set(arguments)
  foreach(code IN LISTS codes)
    list(APPEND arguments --generate-code "${code}")
  endforeach()

  set(archive "${CMAKE_CURRENT_BINARY_DIR}/cuda-host/$<CONFIG>/${target}.a")
  add_custom_command(OUTPUT "${archive}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/cuda-host/$<CONFIG>"
    COMMAND "${GENERATIVEQC_NVPRUNE}" --no-relocatable-elf ${arguments}
            "$<TARGET_FILE:${target}>" --output-file "${archive}.tmp"
    COMMAND "${CMAKE_COMMAND}" -E rename "${archive}.tmp" "${archive}"
    DEPENDS ${target} "$<TARGET_FILE:${target}>" "${GENERATIVEQC_NVPRUNE}"
    COMMENT "Removing resolved device-link inputs from ${target} host archive"
    VERBATIM)
  add_custom_target(${target}_host_artifact DEPENDS "${archive}")
  add_library(${target}_host STATIC IMPORTED GLOBAL)
  set_target_properties(${target}_host PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    INTERFACE_LINK_LIBRARIES "$<TARGET_GENEX_EVAL:${target},$<TARGET_PROPERTY:${target},INTERFACE_LINK_LIBRARIES>>")
  # IMPORTED_LOCATION does not evaluate generator expressions. Resolve each
  # configuration explicitly, retaining a normal path for single-config builds.
  if(CMAKE_CONFIGURATION_TYPES)
    foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
      string(TOUPPER "${config}" upper_config)
      set_property(TARGET ${target}_host PROPERTY IMPORTED_LOCATION_${upper_config}
        "${CMAKE_CURRENT_BINARY_DIR}/cuda-host/${config}/${target}.a")
    endforeach()
  else()
    set_property(TARGET ${target}_host PROPERTY IMPORTED_LOCATION
      "${CMAKE_CURRENT_BINARY_DIR}/cuda-host/${CMAKE_BUILD_TYPE}/${target}.a")
  endif()
  add_dependencies(${target}_host ${target}_host_artifact)
  set(${output} "${target}_host" PARENT_SCOPE)
endfunction()

# Keep a shared CUDA artifact's public filename and host exports while giving
# its device link a separate owner. nvprune accepts archives, not shared ELF
# libraries. Whole-archive retention preserves every registration/export even
# when no executable in this build references the plugin's public entry points.
# Callers configure the returned owner with sources' includes/options/providers.
function(generativeqc_add_cuda_shared_library target owner)
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(device_target "${target}_device")
    add_library(${device_target} STATIC ${ARGN})
    set_target_properties(${device_target} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
      POSITION_INDEPENDENT_CODE ON)
    generativeqc_cuda_host_archive(${device_target} host_archive)
    # A host-only link keeps the already resolved device image intact. This TU
    # supplies a link language without adding a public symbol or device work.
    set(anchor "${CMAKE_CURRENT_BINARY_DIR}/cuda-host/${target}_link.cpp")
    file(GENERATE OUTPUT "${anchor}" CONTENT "// Host link for a device-closed CUDA artifact.\n")
    add_library(${target} SHARED "${anchor}")
    set_target_properties(${target} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
    target_link_libraries(${target} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${host_archive}>")
    set(${owner} "${device_target}" PARENT_SCOPE)
  else()
    add_library(${target} SHARED ${ARGN})
    set(${owner} "${target}" PARENT_SCOPE)
  endif()
endfunction()
