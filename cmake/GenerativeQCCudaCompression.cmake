include_guard(GLOBAL)

# Configure image storage before creating CUDA targets, including optional AOT
# libraries and object libraries. Preserve NVCC's executable instructions and
# every requested cubin/PTX architecture. The legacy fatbinary compressor avoids
# the CUDA-12.4 driver requirement of the newer --compress-mode=size format.
function(generativeqc_configure_cuda_compression)
  if(GENERATIVEQC_COMPRESS_CUDA_FATBINS AND
     CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin=-compress-all>")
    # RDC archives create a new fatbinary during device linking; compiling their
    # individual objects with compression alone does not cover that final image.
    add_link_options("$<DEVICE_LINK:-Xfatbin=-compress-all>")
  endif()
endfunction()
