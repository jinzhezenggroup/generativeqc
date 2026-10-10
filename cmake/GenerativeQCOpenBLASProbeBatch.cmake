# Private hook for a source-signature try_compile. CMake owns importing the
# provider target graph and setting the compiler/platform/required-check flags.
# At directory completion, split the remaining checks into independent targets;
# combining their objects in one executable would change static-archive lookup.
# Defer an include rather than a function: the generated project sets its
# caller's policies after project(), and each target must see those same policies.
if(NOT _generativeqc_openblas_batch_deferred)
  set(_generativeqc_openblas_batch_deferred TRUE)
  cmake_language(EVAL CODE
    "cmake_language(DEFER CALL include [==[${CMAKE_CURRENT_LIST_FILE}]==])")
  return()
endif()

get_property(_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
list(LENGTH _targets _target_count)
if(NOT _target_count EQUAL 1)
  return()
endif()
list(GET _targets 0 _first_target)
get_target_property(_type "${_first_target}" TYPE)
if(NOT _type STREQUAL "EXECUTABLE" AND NOT _type STREQUAL "STATIC_LIBRARY")
  return()
endif()
get_target_property(_base_definitions "${_first_target}" COMPILE_DEFINITIONS)
list(GET GENERATIVEQC_OPENBLAS_BATCH_CAPABILITIES 0 _first_capability)
foreach(_capability IN LISTS GENERATIVEQC_OPENBLAS_BATCH_CAPABILITIES)
  if(_capability STREQUAL _first_capability)
    set(_target "${_first_target}")
  else()
    set(_target "generativeqc_check_${_capability}")
    set(_source "${GENERATIVEQC_OPENBLAS_BATCH_DIR}/${_capability}.cpp")
    if(_type STREQUAL "EXECUTABLE")
      add_executable("${_target}" "${_source}")
    else()
      add_library("${_target}" STATIC "${_source}")
    endif()
    foreach(_property IN ITEMS
        COMPILE_OPTIONS COMPILE_FEATURES INCLUDE_DIRECTORIES
        LINK_LIBRARIES LINK_OPTIONS LINK_DIRECTORIES LINK_FLAGS STATIC_LIBRARY_OPTIONS
        CXX_STANDARD CXX_STANDARD_REQUIRED CXX_EXTENSIONS
        POSITION_INDEPENDENT_CODE MSVC_RUNTIME_LIBRARY ENABLE_EXPORTS)
      get_property(_set TARGET "${_first_target}" PROPERTY "${_property}" SET)
      if(_set)
        get_target_property(_value "${_first_target}" "${_property}")
        set_property(TARGET "${_target}" PROPERTY "${_property}" "${_value}")
      endif()
    endforeach()
    if(_base_definitions)
      set_property(TARGET "${_target}" PROPERTY COMPILE_DEFINITIONS "${_base_definitions}")
    endif()
    # try_compile uses Make's /fast target, which skips add_dependencies.
    # A post-build command is part of the actual rule on every generator.
    add_custom_command(TARGET "${_first_target}" POST_BUILD
      COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}"
              --target "${_target}" --config "$<CONFIG>"
      VERBATIM)
  endif()
  # Match CheckCXXSourceCompiles without leaking result definitions from one
  # capability into another probe's translation unit.
  target_compile_definitions("${_target}" PRIVATE
    "GENERATIVEQC_OPENBLAS_HAS_${_capability}")
endforeach()
file(WRITE "${GENERATIVEQC_OPENBLAS_BATCH_DIR}/validated.txt"
   "${GENERATIVEQC_OPENBLAS_BATCH_CAPABILITIES}")
