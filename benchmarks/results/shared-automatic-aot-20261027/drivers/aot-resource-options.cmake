# Qualification-only diagnostics; no optimization or architecture change.
cmake_language(DEFER CALL target_compile_options generativeqc_native_stationary PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>")
cmake_language(DEFER CALL target_compile_options generativeqc_libxc_semilocal_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>")
