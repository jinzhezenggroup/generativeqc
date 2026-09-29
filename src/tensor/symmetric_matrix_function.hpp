#ifndef GENERATIVEQC_TENSOR_SYMMETRIC_MATRIX_FUNCTION_HPP
#define GENERATIVEQC_TENSOR_SYMMETRIC_MATRIX_FUNCTION_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace generativeqc::tensor {

enum class SymmetricMatrixFunction : std::uint8_t { inverse_sqrt, pseudoinverse };

/** Apply a fixed-branch full-Frobenius VJP from a checked symmetric eigensystem.
 *
 * eigenvectors are row-major Q[row,column]. ``retained`` is an owner-validated
 * fixed spectral branch; branch selection/eigensolver gauge are deliberately
 * outside this primitive. Cross retained/discarded response is included.
 */
[[nodiscard]] std::vector<double> symmetric_matrix_function_vjp(
    std::span<const double> eigenvalues, std::span<const double> eigenvectors,
    std::span<const std::uint8_t> retained, std::span<const double> response,
    SymmetricMatrixFunction function, double resolution);

}  // namespace generativeqc::tensor
#endif
