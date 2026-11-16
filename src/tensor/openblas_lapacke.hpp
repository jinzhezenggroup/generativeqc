#ifndef GENERATIVEQC_TENSOR_OPENBLAS_LAPACKE_HPP
#define GENERATIVEQC_TENSOR_OPENBLAS_LAPACKE_HPP

#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
// Some SciPy wheels declare ordinary LAPACKE names but export prefixed symbols.
// Rename only the declarations we consume; already-prefixed headers also work.
// CMake uses this same adapter so its link probe checks the production ABI.
#define LAPACKE_dpotrf scipy_LAPACKE_dpotrf
#define LAPACKE_dsyevd scipy_LAPACKE_dsyevd
#endif

#include <lapacke.h>

#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
#undef LAPACKE_dpotrf
#undef LAPACKE_dsyevd
#endif

#endif
