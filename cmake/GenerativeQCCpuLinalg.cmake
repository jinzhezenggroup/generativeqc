include_guard(GLOBAL)

set(GENERATIVEQC_CPU_LINALG_PROVIDER "auto" CACHE STRING
    "CPU dense-linear-algebra provider: auto, scalar, or openblas")
set_property(CACHE GENERATIVEQC_CPU_LINALG_PROVIDER PROPERTY STRINGS auto scalar openblas)

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
        set(_provider_libraries PkgConfig::GENERATIVEQC_OPENBLAS)
        set(_include_dirs ${GENERATIVEQC_OPENBLAS_INCLUDE_DIRS})
      else()
        pkg_check_modules(GENERATIVEQC_SCIPY_OPENBLAS QUIET IMPORTED_TARGET scipy-openblas)
        if(TARGET PkgConfig::GENERATIVEQC_SCIPY_OPENBLAS)
          set(_provider_libraries PkgConfig::GENERATIVEQC_SCIPY_OPENBLAS)
          set(_include_dirs ${GENERATIVEQC_SCIPY_OPENBLAS_INCLUDE_DIRS})
          set(_scipy_prefix 1)
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
    # Only invalidate the expensive try-compiles if their effective provider or
    # toolchain inputs changed. An imported target can keep its name when its
    # include paths, link libraries, or options change in the same build tree.
    set(_generativeqc_openblas_probe_inputs
        "libraries=${_provider_libraries}\nincludes=${_include_dirs}\n"
        "scipy_prefix=${_scipy_prefix}\nlocal=${_thread_probe}\n"
        "global=${_global_thread_probe}\nlapacke=${_lapack_probe}\n")
    foreach(_variable IN ITEMS
        GENERATIVEQC_CPU_LINALG_PROVIDER
        OpenBLAS_DIR OpenBLAS_VERSION OpenBLAS_LIBRARIES OpenBLAS_INCLUDE_DIRS
        GENERATIVEQC_OPENBLAS_VERSION GENERATIVEQC_OPENBLAS_LINK_LIBRARIES
        GENERATIVEQC_OPENBLAS_LDFLAGS GENERATIVEQC_OPENBLAS_CFLAGS
        GENERATIVEQC_SCIPY_OPENBLAS_VERSION GENERATIVEQC_SCIPY_OPENBLAS_LINK_LIBRARIES
        GENERATIVEQC_SCIPY_OPENBLAS_LDFLAGS GENERATIVEQC_SCIPY_OPENBLAS_CFLAGS
        CMAKE_CXX_COMPILER CMAKE_CXX_COMPILER_ID CMAKE_CXX_COMPILER_VERSION
        CMAKE_CXX_COMPILER_TARGET CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN
        CMAKE_TOOLCHAIN_FILE CMAKE_SYSROOT CMAKE_BUILD_TYPE
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_EXE_LINKER_FLAGS CMAKE_REQUIRED_FLAGS CMAKE_REQUIRED_DEFINITIONS
        CMAKE_REQUIRED_LINK_OPTIONS CMAKE_REQUIRED_LINK_DIRECTORIES
        CMAKE_TRY_COMPILE_TARGET_TYPE)
      string(APPEND _generativeqc_openblas_probe_inputs
             "${_variable}=${${_variable}}\n")
    endforeach()
    foreach(_library IN LISTS _provider_libraries)
      if(TARGET "${_library}")
        foreach(_property IN ITEMS
            IMPORTED_LOCATION IMPORTED_IMPLIB INTERFACE_INCLUDE_DIRECTORIES
            INTERFACE_COMPILE_OPTIONS INTERFACE_COMPILE_DEFINITIONS
            INTERFACE_LINK_LIBRARIES INTERFACE_LINK_OPTIONS INTERFACE_LINK_DIRECTORIES)
          get_target_property(_value "${_library}" "${_property}")
          string(APPEND _generativeqc_openblas_probe_inputs
                 "${_library}.${_property}=${_value}\n")
        endforeach()
      endif()
    endforeach()
    # A provider can be replaced in place without changing its CMake/package
    # version. Include the probed header content in the cache identity.
    foreach(_include_dir IN LISTS _include_dirs)
      foreach(_header IN ITEMS cblas.h lapacke.h)
        if(EXISTS "${_include_dir}/${_header}")
          file(SHA256 "${_include_dir}/${_header}" _header_sha256)
          string(APPEND _generativeqc_openblas_probe_inputs
                 "${_include_dir}/${_header}=${_header_sha256}\n")
        endif()
      endforeach()
    endforeach()
    string(SHA256 _generativeqc_openblas_probe_key
           "${_generativeqc_openblas_probe_inputs}")
    if(NOT "${GENERATIVEQC_OPENBLAS_PROBE_CACHE_KEY}" STREQUAL
           "${_generativeqc_openblas_probe_key}")
      unset(GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS CACHE)
      unset(GENERATIVEQC_OPENBLAS_HAS_GLOBAL_THREADS CACHE)
      unset(GENERATIVEQC_OPENBLAS_HAS_LAPACKE CACHE)
    endif()
    check_cxx_source_compiles("${_thread_probe}" GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS)
    check_cxx_source_compiles("${_global_thread_probe}" GENERATIVEQC_OPENBLAS_HAS_GLOBAL_THREADS)
    check_cxx_source_compiles("${_lapack_probe}" GENERATIVEQC_OPENBLAS_HAS_LAPACKE)
    set(GENERATIVEQC_OPENBLAS_PROBE_CACHE_KEY
        "${_generativeqc_openblas_probe_key}" CACHE INTERNAL
        "Inputs to GenerativeQC's OpenBLAS capability checks" FORCE)
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
