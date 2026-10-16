#pragma once

/* Only declaration carriers needed by the private NVIDIA host ABI. */
typedef enum { CUDA_R_64F = 1 } cudaDataType;
typedef enum { MAJOR_VERSION = 0, MINOR_VERSION = 1, PATCH_LEVEL = 2 } libraryPropertyType;
