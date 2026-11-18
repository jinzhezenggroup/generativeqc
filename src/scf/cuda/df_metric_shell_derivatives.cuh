#ifndef GENERATIVEQC_SCF_CUDA_DF_METRIC_SHELL_DERIVATIVES_CUH
#define GENERATIVEQC_SCF_CUDA_DF_METRIC_SHELL_DERIVATIVES_CUH
#include <cuda_runtime.h>

#include "core/types.hpp"
#include "scf/cuda/df_shell_derivatives.cuh"

namespace generativeqc::scf {

/** Auxiliary AO polynomials as top-order Hermite Gaussians, by shell degree.
 *
 * A homogeneous harmonic polynomial S of degree l satisfies
 * S(r-A) exp(-a|r-A|^2) = (2a)^-l S(d/dA) exp(-a|r-A|^2) (Hobson), so each AO
 * is an exact combination of the degree-l Hermite Gaussians alone. Row p of
 * class l holds AO p's coefficients on the degree-l monomials in
 * df_metric_monomial order. Classes s through g, both representations for s/p.
 */
struct DfMetricHermiteTable {
  static constexpr unsigned kCoefficients = 1 + 3 * 3 + 5 * 6 + 7 * 10 + 9 * 15;
  double coefficients[kCoefficients]{};
};

/** Fill the table from the same normalized expansions as the derivative view.
 * Returns false, leaving the caller on the per-AO traversal, when a shell is
 * above g or any AO polynomial is not homogeneous harmonic of its degree
 * (Cartesian d and above). */
bool prepare_df_metric_hermite_table(const core::System& auxiliary, DfMetricHermiteTable& table);

/** Add sum_PQ M'[P,Q] (W[P,Q] + W[Q,P]) for a complete row-major naux x naux W.
 *
 * One thread owns an ordered auxiliary shell pair and differentiates only its
 * first center; the ordered sum over both orientations is the complete
 * translation-invariant gradient. Same-atom pairs contribute exactly zero to it
 * and are skipped. Weights may be nonsymmetric. Coulomb only (omega == 0).
 * The caller's view must cover every auxiliary shell; this launch allocates
 * nothing and adds into gradient on stream. */
cudaError_t launch_df_metric_shell_derivative(const DfShellBasisView& auxiliary,
                                              const DfMetricHermiteTable& table,
                                              const double* positions, const double* weights,
                                              double* gradient, cudaStream_t stream);
}  // namespace generativeqc::scf
#endif
