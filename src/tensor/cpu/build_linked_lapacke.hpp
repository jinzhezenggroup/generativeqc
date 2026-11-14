#pragma once

// Some SciPy LP64 distributions ship conventional LAPACKE declarations while
// exporting prefixed symbols. Preserve the provider's argument types and rename
// only the two entrypoints used by the build-linked adapter during inclusion.
// CMake uses this same header and still requires both actual link symbols.
#if defined(GENERATIVEQC_OPENBLAS_SCIPY_PREFIX) && GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
#define LAPACKE_dpotrf scipy_LAPACKE_dpotrf
#define LAPACKE_dsyevd scipy_LAPACKE_dsyevd
#endif

#include <lapacke.h>

#if defined(GENERATIVEQC_OPENBLAS_SCIPY_PREFIX) && GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
#undef LAPACKE_dpotrf
#undef LAPACKE_dsyevd
#endif
