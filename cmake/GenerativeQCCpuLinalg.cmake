include_guard(GLOBAL)

set(GENERATIVEQC_CPU_LINALG_PROVIDER "auto" CACHE STRING
    "CPU dense-linear-algebra provider: auto, scalar, or openblas")
set_property(CACHE GENERATIVEQC_CPU_LINALG_PROVIDER PROPERTY STRINGS auto scalar openblas)

function(_generativeqc_normalize_pkgconfig_libraries prefix usable)
  set(${usable} TRUE PARENT_SCOPE)
  set(_pkg_target PkgConfig::${prefix})
  get_target_property(_options ${_pkg_target} INTERFACE_LINK_OPTIONS)
  set(_absolute_libraries "")
  foreach(_option IN LISTS _options)
    if(IS_ABSOLUTE "${_option}" AND _option MATCHES "\\.(a|so(\\.[0-9]+)*|dylib|lib)$")
      list(APPEND _absolute_libraries "${_option}")
    endif()
  endforeach()
  if(NOT _absolute_libraries)
    return()
  endif()

  # FindPkgConfig puts bare absolute libraries in LINK_OPTIONS, before objects.
  # Rebuild only the supported library sequence, retaining resolved -l operands
  # and their order. Never move operands across unknown/positional link flags.
  get_target_property(_resolved_libraries ${_pkg_target} INTERFACE_LINK_LIBRARIES)
  if(NOT _resolved_libraries)
    set(_resolved_libraries "")
  endif()
  set(_ordered_libraries "")
  foreach(_flag IN LISTS ${prefix}_LDFLAGS)
    if(_flag IN_LIST _absolute_libraries)
      list(APPEND _ordered_libraries "${_flag}")
    elseif(_flag MATCHES "^-l.+" AND _resolved_libraries)
      list(POP_FRONT _resolved_libraries _library)
      list(APPEND _ordered_libraries "${_library}")
    elseif(_flag MATCHES "^-L.+" OR _flag STREQUAL "-pthread"
           OR _flag MATCHES "^-Wl,-rpath[,=][^,]+$")
      # Search paths are already resolved by FindPkgConfig; these options do
      # not change archive extraction or the scope of a library operand.
    else()
      message(WARNING
        "Ignoring ${prefix} pkg-config metadata with unsupported linker argument '${_flag}'; trying OpenBLAS CMake metadata")
      set(${usable} FALSE PARENT_SCOPE)
      return()
    endif()
  endforeach()
  if(_resolved_libraries)
    set(${usable} FALSE PARENT_SCOPE)
    return()
  endif()
  list(REMOVE_ITEM _options ${_absolute_libraries})
  set_property(TARGET ${_pkg_target} PROPERTY INTERFACE_LINK_LIBRARIES "${_ordered_libraries}")
  set_property(TARGET ${_pkg_target} PROPERTY INTERFACE_LINK_OPTIONS "${_options}")
endfunction()

function(generativeqc_configure_cpu_linalg target)
  if(NOT GENERATIVEQC_CPU_LINALG_PROVIDER STREQUAL "auto"
     AND NOT GENERATIVEQC_CPU_LINALG_PROVIDER STREQUAL "scalar"
     AND NOT GENERATIVEQC_CPU_LINALG_PROVIDER STREQUAL "openblas")
    message(FATAL_ERROR
            "GENERATIVEQC_CPU_LINALG_PROVIDER must be auto, scalar, or openblas")
  endif()

  set(_provider_libraries "")
  set(_scipy_prefix 0)
  set(_include_dirs "")
  if(NOT GENERATIVEQC_CPU_LINALG_PROVIDER STREQUAL "scalar")
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(GENERATIVEQC_OPENBLAS QUIET IMPORTED_TARGET openblas)
      if(TARGET PkgConfig::GENERATIVEQC_OPENBLAS)
        set(_pkg_prefix GENERATIVEQC_OPENBLAS)
        set(_provider_libraries PkgConfig::GENERATIVEQC_OPENBLAS)
        set(_include_dirs ${GENERATIVEQC_OPENBLAS_INCLUDE_DIRS})
      else()
        pkg_check_modules(GENERATIVEQC_SCIPY_OPENBLAS QUIET IMPORTED_TARGET scipy-openblas)
        if(TARGET PkgConfig::GENERATIVEQC_SCIPY_OPENBLAS)
          set(_pkg_prefix GENERATIVEQC_SCIPY_OPENBLAS)
          set(_provider_libraries PkgConfig::GENERATIVEQC_SCIPY_OPENBLAS)
          set(_include_dirs ${GENERATIVEQC_SCIPY_OPENBLAS_INCLUDE_DIRS})
          set(_scipy_prefix 1)
        endif()
      endif()
      if(_provider_libraries)
        _generativeqc_normalize_pkgconfig_libraries(${_pkg_prefix} _pkg_usable)
        if(NOT _pkg_usable)
          set(_provider_libraries "")
          set(_include_dirs "")
          set(_scipy_prefix 0)
        endif()
      endif()
    endif()

    # OpenBLAS also ships a CMake package config. This path matters on minimal
    # build hosts without pkg-config and for SciPy's redistributable OpenBLAS.
    if(NOT _provider_libraries)
      find_package(OpenBLAS CONFIG QUIET)
      if(OpenBLAS_FOUND AND OpenBLAS_LIBRARIES)
        set(_provider_libraries ${OpenBLAS_LIBRARIES})
        set(_include_dirs ${OpenBLAS_INCLUDE_DIRS})
        foreach(_library IN LISTS OpenBLAS_LIBRARIES)
          if(_library MATCHES "scipy_openblas")
            set(_scipy_prefix 1)
          endif()
        endforeach()
      endif()
    endif()
  endif()

  if(_provider_libraries)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${_include_dirs})
    set(CMAKE_REQUIRED_LIBRARIES ${_provider_libraries})
    if(_scipy_prefix)
      set(_thread_probe
          "#include <cblas.h>\nint main(){return scipy_openblas_set_num_threads_local(1);}")
      set(_global_thread_probe
          "#include <cblas.h>\nint main(){int n=scipy_openblas_get_num_threads();scipy_openblas_set_num_threads(n);return 0;}")
      set(_lapack_probe
          "#include <lapacke.h>\nint main(){double a[1]={1},w[1];int x=scipy_LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',1,a,1);return x+scipy_LAPACKE_dsyevd(LAPACK_ROW_MAJOR,'V','L',1,a,1,w);}")
    else()
      set(_thread_probe
          "#include <cblas.h>\nint main(){return openblas_set_num_threads_local(1);}")
      set(_global_thread_probe
          "#include <cblas.h>\nint main(){int n=openblas_get_num_threads();openblas_set_num_threads(n);return 0;}")
      set(_lapack_probe
          "#include <lapacke.h>\nint main(){double a[1]={1},w[1];int x=LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',1,a,1);return x+LAPACKE_dsyevd(LAPACK_ROW_MAJOR,'V','L',1,a,1,w);}")
    endif()
    # Cached successes are hints only: validate their actual compile/link inputs
    # together before reusing them. A dependency fingerprint cannot in general
    # cover transitive target properties, new include files, or replaced archives.
    # Cached failures are always re-probed, so newly available APIs are detected.
    unset(GENERATIVEQC_OPENBLAS_PROBE_CACHE_KEY CACHE)
    set(_capabilities LOCAL_THREADS GLOBAL_THREADS LAPACKE)
    set(_probe_variables _thread_probe _global_thread_probe _lapack_probe)
    set(_positive_capabilities "")
    foreach(_capability IN LISTS _capabilities)
      if(GENERATIVEQC_OPENBLAS_HAS_${_capability})
        list(APPEND _positive_capabilities "${_capability}")
      endif()
    endforeach()
    list(LENGTH _positive_capabilities _positive_count)
    set(_batch_valid FALSE)
    # Keep unusual project hooks/toolchains on CMake's ordinary check path.
    # The batch hook must exclusively own the generated check project setup.
    set(_batch_eligible TRUE)
    foreach(_variable IN ITEMS CMAKE_TOOLCHAIN_FILE CMAKE_PROJECT_INCLUDE
        CMAKE_PROJECT_INCLUDE_BEFORE CMAKE_PROJECT_TOP_LEVEL_INCLUDES
        CMAKE_PROJECT_CMAKE_TRY_COMPILE_INCLUDE
        CMAKE_PROJECT_CMAKE_TRY_COMPILE_INCLUDE_BEFORE
        CMAKE_USER_MAKE_RULES_OVERRIDE CMAKE_USER_MAKE_RULES_OVERRIDE_CXX)
      if(${_variable})
        set(_batch_eligible FALSE)
      endif()
    endforeach()
    if(NOT CMAKE_GENERATOR STREQUAL "Unix Makefiles"
       AND NOT CMAKE_GENERATOR STREQUAL "Ninja"
       AND NOT CMAKE_GENERATOR STREQUAL "Ninja Multi-Config")
      set(_batch_eligible FALSE)
    endif()
    # Semicolon-containing required flags have version-dependent legacy parsing.
    if(_positive_count GREATER 1 AND _batch_eligible
       AND NOT CMAKE_REQUIRED_FLAGS MATCHES ";")
      set(_probe_dir "${CMAKE_BINARY_DIR}/CMakeFiles/GenerativeQCOpenBLASProbe")
      file(MAKE_DIRECTORY "${_probe_dir}")
      foreach(_capability IN LISTS _positive_capabilities)
        list(FIND _capabilities "${_capability}" _index)
        list(GET _probe_variables ${_index} _probe_variable)
        file(WRITE "${_probe_dir}/${_capability}.cpp" "${${_probe_variable}}\n")
      endforeach()
      list(GET _positive_capabilities 0 _first_capability)
      set(_link_options "")
      if(CMAKE_REQUIRED_LINK_OPTIONS)
        set(_link_options LINK_OPTIONS ${CMAKE_REQUIRED_LINK_OPTIONS})
      endif()
      set(_link_directories "")
      if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.31 AND CMAKE_REQUIRED_LINK_DIRECTORIES)
        set(_link_directories
            "-DLINK_DIRECTORIES:STRING=${CMAKE_REQUIRED_LINK_DIRECTORIES}")
      endif()
      if(NOT CMAKE_REQUIRED_QUIET)
        message(CHECK_START "Validating cached OpenBLAS capabilities")
      endif()
      # Do not let retained native build outputs bypass any compilation/link.
      # Compiler launchers may still safely reuse their content-addressed cache.
      file(REMOVE_RECURSE "${_probe_dir}/build")
      file(REMOVE "${_probe_dir}/validated.txt")
      unset(_batch_result)
      unset(_batch_result CACHE)
      try_compile(_batch_result "${_probe_dir}/build"
        SOURCES "${_probe_dir}/${_first_capability}.cpp"
        COMPILE_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS}
        ${_link_options}
        LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES}
        CMAKE_FLAGS "-DCOMPILE_DEFINITIONS:STRING=${CMAKE_REQUIRED_FLAGS}"
                    "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
                    ${_link_directories}
                    "-DCMAKE_PROJECT_INCLUDE:FILEPATH=${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GenerativeQCOpenBLASProbeBatch.cmake"
                    "-DGENERATIVEQC_OPENBLAS_BATCH_CAPABILITIES:STRING=${_positive_capabilities}"
                    "-DGENERATIVEQC_OPENBLAS_BATCH_DIR:PATH=${_probe_dir}"
        OUTPUT_VARIABLE _batch_output)
      # A project/toolchain hook may intercept CMAKE_PROJECT_INCLUDE. Require
      # proof that every independent target was attached before trusting it.
      if(_batch_result AND EXISTS "${_probe_dir}/validated.txt")
        file(READ "${_probe_dir}/validated.txt" _validated)
        if(_validated STREQUAL _positive_capabilities)
          set(_batch_valid TRUE)
        endif()
      endif()
      if(NOT CMAKE_REQUIRED_QUIET)
        if(_batch_valid)
          message(CHECK_PASS "Success")
        else()
          message(CHECK_FAIL "Failed; re-probing individually")
        endif()
      endif()
      # try_compile always runs; do not retain its own implementation result.
      unset(_batch_result CACHE)
    endif()
    foreach(_capability IN LISTS _capabilities)
      if(NOT _batch_valid OR NOT _capability IN_LIST _positive_capabilities)
        list(FIND _capabilities "${_capability}" _index)
        list(GET _probe_variables ${_index} _probe_variable)
        unset(GENERATIVEQC_OPENBLAS_HAS_${_capability})
        unset(GENERATIVEQC_OPENBLAS_HAS_${_capability} CACHE)
        check_cxx_source_compiles("${${_probe_variable}}"
          GENERATIVEQC_OPENBLAS_HAS_${_capability})
      endif()
    endforeach()
    unset(CMAKE_REQUIRED_INCLUDES)
    unset(CMAKE_REQUIRED_LIBRARIES)

    target_link_libraries(${target} PRIVATE ${_provider_libraries})
    target_include_directories(${target} PRIVATE ${_include_dirs})
    target_compile_definitions(${target} PRIVATE
      GENERATIVEQC_HAS_OPENBLAS=1
      GENERATIVEQC_OPENBLAS_SCIPY_PREFIX=${_scipy_prefix}
      GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS=$<BOOL:${GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS}>
      GENERATIVEQC_OPENBLAS_HAS_GLOBAL_THREADS=$<BOOL:${GENERATIVEQC_OPENBLAS_HAS_GLOBAL_THREADS}>
      GENERATIVEQC_OPENBLAS_HAS_LAPACKE=$<BOOL:${GENERATIVEQC_OPENBLAS_HAS_LAPACKE}>)
    message(STATUS
      "GenerativeQC CPU linear algebra: OpenBLAS (local threads=${GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS}, global threads=${GENERATIVEQC_OPENBLAS_HAS_GLOBAL_THREADS}, LAPACKE=${GENERATIVEQC_OPENBLAS_HAS_LAPACKE})")
  else()
    if(GENERATIVEQC_CPU_LINALG_PROVIDER STREQUAL "openblas")
      message(FATAL_ERROR
              "GENERATIVEQC_CPU_LINALG_PROVIDER=openblas requested but no OpenBLAS package metadata was found")
    endif()
    target_compile_definitions(${target} PRIVATE
      GENERATIVEQC_HAS_OPENBLAS=0
      GENERATIVEQC_OPENBLAS_SCIPY_PREFIX=0
      GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS=0
      GENERATIVEQC_OPENBLAS_HAS_GLOBAL_THREADS=0
      GENERATIVEQC_OPENBLAS_HAS_LAPACKE=0)
    message(STATUS "GenerativeQC CPU linear algebra: scalar fallback")
  endif()
endfunction()
