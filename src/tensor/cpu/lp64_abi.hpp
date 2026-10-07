#pragma once

#include <cstdint>

namespace generativeqc::tensor::cpu {
/** Borrowed LP64 CPU ABI only. It confers no provider admission or ownership. */
using LapackInt = std::int32_t;

using LapackDpotrfWork = LapackInt (*)(LapackInt matrix_layout, char uplo, LapackInt n,
                                       double* matrix, LapackInt leading_dimension);
using LapackDpoconWork = LapackInt (*)(LapackInt matrix_layout, char uplo, LapackInt n,
                                       const double* factor, LapackInt leading_dimension,
                                       double matrix_one_norm, double* reciprocal_condition,
                                       double* work, LapackInt* integer_work);
using LapackDsyevdWork = LapackInt (*)(LapackInt matrix_layout, char job_vectors, char uplo,
                                       LapackInt n, double* matrix, LapackInt leading_dimension,
                                       double* eigenvalues, double* work, LapackInt work_count,
                                       LapackInt* integer_work, LapackInt integer_work_count);
using CblasDtrsm = void (*)(int layout, int side, int triangle, int transpose, int diagonal,
                            LapackInt rows, LapackInt columns, double alpha,
                            const double* triangular_matrix, LapackInt leading_triangular,
                            double* right_hand_side, LapackInt leading_rhs);
using CblasDgemm = void (*)(int layout, int transpose_left, int transpose_right, LapackInt rows,
                            LapackInt columns, LapackInt inner, double alpha, const double* left,
                            LapackInt leading_left, const double* right, LapackInt leading_right,
                            double beta, double* result, LapackInt leading_result);
using BlasSetNumThreadsLocal = int (*)(int threads);
using BlasThreadCleanup = void (*)();

}  // namespace generativeqc::tensor::cpu
