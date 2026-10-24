# Decision: bound pure-Coulomb DF response source work

Status: implemented
Date: 2026-10-13

## Problem

The semilocal 96-atom endpoint in #2411 streamed dense DF values even when one
physical symmetric packed fitted tensor could fit. Its general J/K response
also reread fitted panels for the exchange metric Gram although exact exchange
was absent. Bounded storage alone did not bound that repeated source work.

## Decision and invariants

- A generated physical singleton restricted J-only Fock owner may admit one
  packed fitted value with rank zero. This does not infer occupied factors.
  Dense residency retains priority; batch, unrestricted, and unqualified
  arbitrary-density planners keep their former automatic policy.
- Only live automatic budgets can rebalance value/response capacity. Preserve
  the total envelope and conservative response floor. Explicit caps, explicit
  layouts, and failed resource probes retain their original split/fallback.
- With a full-rank metric and an admitted fitted-panel reader, pure Coulomb
  uses p=(A M^-1):D, bar_A=sum_t c_t D_t p_t, and
  bar_M=-sum_t c_t p_t p_t^T/2. Each fitted panel is read once. A generated
  source may still regenerate all raw A for each fitted panel; this is linear
  in output panels, not the general exchange path's ordered panel pairs.
- Nonsymmetric densities and signed term coefficients keep their full dense
  contraction convention. All work uses existing charged workspace and the
  same producer/consumer stream. The fitted reader may reuse only its declared
  metric scratch and must not overwrite the retained metric adjoint/potentials.
- Truncated metrics keep the spectral pseudoinverse VJP, including finite
  discarded-eigenspace motion. Only exchange-only raw reloads are skipped.
  Borrowed, occupied, retained-value, scalar, and exchange routes remain bounded
  alternatives under their original admission conditions.

## Evidence and limits

Host qualification uses independent central finite differences, compiled native
budget negative controls, and executable production response adapters with
host CUDA/cuBLAS doubles. Those doubles do not establish CUDA compilation,
asynchronous execution, device performance, or complete endpoint acceptance.
Native device-gated tests compare source/resident/packed physical derivatives
with the independent CPU analytic oracle and inspect semantic raw source work.

The independent 1e-8 Eh energy and 1e-7 Eh/Bohr force gates remain unchanged.
No new GPU timing or accepted 48/96-atom endpoint is implied by these source
changes. Revisit planning and measured end-to-end work after source-matched
device qualification; do not close #2411 on host/source evidence alone.

References: #2411, #2425; tests/python/test_df_preparation_budget.py,
tests/python/test_df_streamed_coulomb_only.py,
tests/python/test_df_streamed_coulomb_host.py; tests/native/test_density_fitting.cpp
