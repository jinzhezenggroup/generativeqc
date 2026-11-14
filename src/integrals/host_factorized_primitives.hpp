#pragma once

namespace generativeqc::integrals {

/** Canonical full-range Cartesian s--f scalar value, with the same input
 * packing and external scaling as the first/mixed operations. Output has one
 * element. A physical consumer must justify omitting independent partials;
 * common numeric coordinates alone do not make nuclear derivatives vanish.
 */
bool full_range_primitive_value_cpu(const unsigned* angular, unsigned component,
                                    const double* record, double* output);

/** Explicit canonical full-range Cartesian s--f value/gradient operation.
 * Record packing is four exponents and twelve independent center coordinates.
 * Scale/contraction stay with the consumer; a caller's trailing scale field is
 * not consumed here. Output has thirteen elements: value then twelve partials.
 * The generated first-order specialization contains no gradient-JVP work.
 */
bool full_range_primitive_first_cpu(const unsigned* angular, unsigned component,
                                    const double* record, double* output);

/** Canonical full-range Cartesian s--f value and physical-owner matrix.
 * Uses the same record/component packing and external scaling as first/mixed.
 * incidence maps four mathematical slots onto one to three non-origin owners
 * and the fixed origin label `owners`; slot D must carry that origin label.
 * Non-origin labels follow first occurrence. Same-owner coordinates must agree,
 * but coincident distinct owners remain distinct. With dimension=3*owners,
 * output has 1+dimension+dimension^2 fields: value, physical-owner partials,
 * then ordered raw mixed partials. Constant incidence is applied to canonical
 * factors before product AD. No mixed field is copied from a transpose. The
 * consumer scatters each returned owner and its negative common-origin lane.
 */
bool full_range_primitive_matrix_cpu(const unsigned* angular, unsigned component,
                                     const double* record, const unsigned* incidence,
                                     unsigned owners, double* output);

}  // namespace generativeqc::integrals
