# Decision: retain long-range DF as an explicit Hamiltonian opt-in

Status: implemented
Date: 2026-10-16

## Problem

PR #2526 adds an independently qualified fitted LR Hamiltonian, but its proposed
default promotion conflicts with retained complete-endpoint negative evidence.
At measured source `e7e7654930e8c41a349920d18aef41bbe9d3ebf9`, warm energy-only
replay is 1.0166% slower for SVP and 3.8904% slower for TZVP. An earlier campaign
also records 0.9267% and 3.4627% slowdowns. Three sequential repeats do not meet
the five-interleaved-pair promotion requirement, and no tolerated-regression
policy was declared. Force-endpoint gains do not waive that gate.

## Decision

Keep the pre-existing mixed Hamiltonian when
`GENERATIVEQC_DF_RANGE_EXCHANGE` is unset or `direct`. Full-range J/K remain
fitted; LR K remains exact. Only explicit `df` selection enables the independent
operator-matched fitted LR owner. Unsupported selector values still fail.

Selection is frozen at preparation on both CPU and CUDA. Energy-only, forces,
warm replay and changed geometry must use the same selected Hamiltonian, even
if the environment changes after preparation. The native token-bound proof,
not the live environment or requested properties, chooses derivative ownership.
An explicitly requested fitted owner still fails closed on missing capability
or resource admission; this is not a new runtime Direct fallback.

## Rejected alternatives

- Promoting fitted LR because force endpoints improve would disregard repeated
  energy-only regressions and incomplete promotion sampling.
- Selecting Direct for energies and fitted LR for forces would differentiate
  a different Hamiltonian than the reported energy.
- Loosening numerical or performance gates, discarding negative evidence, or
  treating the historical receipts as fresh-head qualification is not acceptable.

## Evidence and invariants

- The new public endpoint regression fails against the prior default by
  observing a fitted native proof for an unset selector. It covers default,
  explicit Direct and explicit DF, RKS/UKS, energy-to-force replay, geometry
  rebinding, proof revocation and independently reconverged libcint energies
  and force finite differences. CUDA cases require a finite Slurm allocation.
- Native API tests retain all 18 method/spin/selector combinations and their
  independent Hamiltonian-specific references; default WB97M-V uses Direct LR.
- Existing fitted-LR energy/force tests now opt in explicitly; their oracle
  models and acceptance gates are unchanged. No numerical kernels change.
- Historical raw evidence, source identities, measured regressions and build
  boundaries remain intact. No new speedup or GPU qualification is claimed.
- Local validation passes 61 tests in `test_range_separated_df.py` and
  `test_mixed_rsh_df_forces.py`, with 28 CUDA cases skipped. The separate host
  contracts pass 89 tests, capacity/stationarity contracts pass 419 tests, and
  related native CTest passes 11/11. Strict Sphinx, native API/ownership checks
  and all 84 lossless historical evidence members also verify successfully.

## Revisit when

A subsequent [scalar-layout optimization](../performance/2026-10-16-range-df-scalar-layout.md)
reduces CPU replay overhead without changing this selection policy. Its new
measurements are separate from the historical receipts above; neither reducing
a regression nor numerical equivalence alone promotes the default.

A separately qualified default-promotion campaign satisfies the repository's
complete cold/warm/geometry endpoint and interleaved comparison gates, or a
predeclared reviewed policy explicitly accepts the per-workload tradeoff.

## References

- [Original numerical decision](../numerics/2026-10-15-range-separated-density-fitting.md)
- [Retained endpoint evidence](../../../../benchmarks/results/range-separated-df-stable-20261015/README.md)
- [Current public selection contract](../../../../docs/user/dft_density_fitting.md)
- [Performance acceptance](../../../../docs/maintainer/performance_engineering.md)
