# Decision: bounded binary-scaled Becke first derivatives

Status: implemented
Date: 2026-10-12

## Problem

The ordinary phased geometry adjoint repeatedly evaluates logarithms and
exponentials for ordered normalized Becke products. Earlier V1–V11 experiments
had genuine device savings but insufficient complete-endpoint stability for
promotion. Issue #2288 preserves those unsuccessful directions. The user later
removed the 2% gain floor, but explicitly prohibits additional SCF iterations;
the scientific, input-identity and work gates are not relaxed.

## Decision

Represent ordered products by a mantissa, integer exponent and exact-zero class.
Retain the canonical switch/ratio AD, four-word pair panel, seven phase boundaries
and geometric gather. Exponent-budgeted products avoid premature underflow;
packed relative exponents reuse the existing private scratch word. Pullbacks use
correctly rounded division, not an approximate reciprocal. Exact two-zero
pruning is licensed only after complete primal validation; single-zero excluded
derivatives remain live. Invalid inputs must not publish forces.

Automatic selection is deliberately limited to 96–128 atoms, ordinary admitted
phase storage, center geometry and no primitive-adjoint alias. The lower bound
is a conservative measured crossover, not a universal performance prediction.
The upper bound is the lowering's independently tested representation limit.
Explicit `GENERATIVEQC_STATIONARY_BECKE_SCALED_PRODUCTS=1` also permits smaller
owners under those same safety guards; `0` retains logarithmic products.
Capture the selector at owner creation. Unsupported owners acquire no extra
storage and retain their incumbent route. This changes first derivatives only,
not SCF quadrature, convergence policy, Hessians or response.

## Evidence

Source base: master `6f3840ad7` (including #2274 and #2298). The measured image
contains only Becke changes, without the unpromoted larger-XC/scatter work.
Job 7645 uses two independent cold native owners and five fixed, interleaved
complete E/F pairs per geometry, retaining all preparations, samples and failures.

| Atoms | Endpoint | Baseline median s | Candidate median s | Reduction | Robust floor |
|---:|---|---:|---:|---:|---:|
| 96 | warm | 12.965453807 | 12.540530946 | 3.27735% | 0.35562% |
| 96 | moved-warm | 12.963502206 | 12.486285910 | 3.68123% | 0.71481% |
| 48 | warm | 4.517080698 | 4.443798706 | 1.62233% | 1.29186% |
| 48 | moved-warm | 4.497400131 | 4.445111461 | 1.16264% | 1.57429% |

The 48-atom portfolio **failed** its moved-geometry noise gate and stays out of
automatic promotion. Do not pool earlier, differently built images or relabel
this failed record as qualified. Small positive medians alone are insufficient.

All 40 timed calls have exactly 1 SCF iteration / 1 Fock build. Cold counts are
17/18 baseline versus 17/17 candidate at 96 atoms, and 19/20 versus 19/20 at 48.
Moved preparation is 12/12 on both sides at both sizes. This is an observed
no-extra-work result, not a cold-speedup claim: optional ERI-source admission can
differ across independently resident owners and is unrelated to Becke geometry.
No checkpoint import, hidden refinement, trimmed sample or adaptive retry occurs.

The native cross-arm density gate remains `2e-11`: 96-atom original/moved drifts
are `7.26916829952174e-12` / `1.5543122344752192e-12`. The maximum independent
force error is `3.805367132514448e-11` Eh/Bohr at 96 atoms and
`2.479416671974377e-11` at 48. Native export/ownership and frozen-seed checks pass.

The broad numerical image passes 2165 focused host tests and 9 device tests.
The final change only narrows automatic host admission; it does not alter any
device equation, scratch layout or the selected 96-atom launch route. The final
policy passes 32 executable host contracts and 18 native CUDA tests covering
explicit/default selection, small-owner fallback, 128 atoms, normalized-adjoint
alias refusal, serial normalization, implicit/external sources and invalid
publication. Both source-matched builds use verified ccache, and GPU jobs use
finite Slurm allocations without changing device visibility.

The compact evidence bundle retains both system records, including the failure,
the measured source patch, final admission delta, driver/input bundle and build
provenance under `benchmarks/results/becke-binary-scaled-20261012/`.

## Rejected alternatives

- Promoting every small positive median: the current 48-atom moved portfolio
  does not clear its unchanged robust noise floor.
- Approximate reciprocals, amplitude cutoffs, relaxed SCF tolerances or additional
  preparation iterations: none is part of this lowering or its qualification.
- Combining primitive normalized adjoints with scaled exponents: both borrow
  the same private scratch field, so selection must remain mutually exclusive.
- Repeating identical source until a favorable noisy sample appears: retain the
  failed 48 record and narrow admission instead.

## Revisit when

Expand the automatic crossover only with a materially justified mechanism or
source-matched, complete cold/warm/moved qualification that preserves the
no-extra-SCF/input/force/resource gates. Performance at 128 atoms is not claimed
by the 96-atom endpoint measurement; 128 has independent numerical/routing
coverage and remains the conservative representation ceiling.

## References

- Issues #1895, #1965 and #2288.
- `docs/developer/dft_grid.md`.
- `benchmarks/results/becke-binary-scaled-20261012/README.md`.
