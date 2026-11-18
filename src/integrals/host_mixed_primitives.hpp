#pragma once

namespace generativeqc::integrals::host_mixed {

/** Unnormalized Cartesian primitive mixed jets in independent nuclear slots.
 * exponents has two positive values, powers has two xyz triples through f.
 * centers/left/right have two xyz slots for S/T and three for attraction.
 * Outputs are [value,left,right,mixed], with S then T in the eight-field call.
 * Attraction includes the physical minus sign for a unit positive charge.
 * Normalization, contraction coefficients, physical atom binding, density and
 * overlap-constraint weights belong to the enclosing consumer. Neither an
 * electronic response nor a stationary Hessian is supplied by these helpers.
 * A common-origin frame preserves exact rigid-translation zeros. All fields
 * publish together; malformed or nonfinite inputs/arithmetic return false
 * without changing output. No heap allocation, JIT, oracle or displaced work.
 */
bool overlap_kinetic(const double* exponents, const double* centers, const unsigned* powers,
                     const double* left, const double* right, double* output) noexcept;
bool attraction(const double* exponents, const double* centers, const unsigned* powers,
                const double* left, const double* right, double* output) noexcept;

}  // namespace generativeqc::integrals::host_mixed
