# Keep installed distribution labels tied to resolved compiler inputs, not the
# GPU on the packaging host or the defaults of a particular CI job.
function(generativeqc_write_native_cli_metadata output)
  string(REPLACE ";" "\", \"" _generativeqc_cli_architectures
         "${CMAKE_CUDA_ARCHITECTURES}")
  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/native_cli_build.json.in"
    "${output}" @ONLY)
endfunction()
