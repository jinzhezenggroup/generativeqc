include_guard(GLOBAL)

# This is a bounded build inventory, not a public scientific admission list.
# The compiler catalog remains authoritative for exact-plan runtime selection.
set(_generativeqc_stationary_profile_catalog
    lda_rks lda_uks pbe_rks pbe_uks r2scan_rks r2scan_uks
    pbe0_rks pbe0_uks b3lyp_rks b3lyp_uks)
set(GENERATIVEQC_STATIONARY_AOT_PROFILES
    "${_generativeqc_stationary_profile_catalog}" CACHE STRING
    "Stationary CUDA AOT profiles to package; empty disables this inventory")

function(generativeqc_select_stationary_aot_profiles output)
  set(selected ${GENERATIVEQC_STATIONARY_AOT_PROFILES})
  foreach(profile IN LISTS selected)
    if(NOT profile IN_LIST _generativeqc_stationary_profile_catalog)
      message(FATAL_ERROR "Unknown stationary CUDA AOT profile: ${profile}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES selected)
  list(SORT selected)
  set(${output} "${selected}" PARENT_SCOPE)
endfunction()
