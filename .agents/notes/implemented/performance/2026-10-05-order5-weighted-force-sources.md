# Decision: qualify precontracted order-five force sources

Status: implemented; composed endpoint and master transplant qualified
Date: 2026-10-05

## Problem

The 96-atom angular diagnostic identifies order five as an expensive consumer,
although its separate-pass timings are not production component shares. The
ordinary fallback differentiates each AO quartet before density contraction.

## Candidate and invariants

Extend the order-four weighted-root candidate to dppp, dpds and ddps. Reuse
WeightedIntegralIR and its scalar AD, the existing normalized Cartesian weight
adapter, primitive-pair geometry/Boys preparation and atom scatter. The 162/108
component classes use additive partitions of at most 64 components; every
partition keeps the full class weight indices. Recover the last center through
the existing compiler-owned translation contract.

Only full-range independent J/K source tasks in these exact classes change.
The scalar lane drains each accepted shell task once and the warp fallback
skips exactly those classes. f-containing classes, combined HF and range-separated
sources retain their old consumers. All queue, shell/AO screening, spin and
precision contracts remain unchanged. No persistent memory or scientific
threshold is added. This code qualification does not promote a provider/profile registry.

## Acceptance

Independent displaced-value Hermite gradients must pass 2e-8*(1+abs(reference));
the retained AO adapter must pass 2e-10*(1+abs(reference)). Cover all components,
partition tails, signed weights, both spins, repeated atoms, symmetry and zero
sources. Real CUDA through-f, memcheck/initcheck and whole-source/complete E+F
qualification remain mandatory. Retain cold, warm and moved SCF histories.

Do not infer an endpoint speedup from fewer scalar roots or smaller static
resource reservations. A losing candidate stays local and is not promoted.

## References

#1892, #1895; tests/python/test_direct_order5_sources.py.

## Complete composed result (Slurm5905)

The control is order-four candidate `b3acd80f7`; the candidate is `8f90f0ac3`.
Both include the composed #1830/#1833/#1847/indexed-policy baseline. These
are incremental order-five measurements, not unmodified-master timings.

| Atoms / regime | Order-four seconds | Order-five seconds | Native iterations |
| --- | ---: | ---: | --- |
| 48 cold | 124.331621 | 116.433140 | 25 / 23 |
| 48 warm | 9.987770 | 9.286528 | all 1 |
| 48 moved | 54.055510 | 53.350329 | 12 / 12 |
| 48 moved-warm | 10.006721 | 9.200909 | all 1 |
| 96 cold | 283.948644 | 244.975738 | 30 / 25 |
| 96 warm | 28.913819 | 26.989124 | all 1 |
| 96 moved | 133.021434 | 123.166182 | 13 / 12 |
| 96 moved-warm | 28.777184 | 26.014840 | all 1 |

The 96-atom warm derivative component is 7.6324 s and grid response 8.7953 s.
The complete warm improvement is 6.66%, with 9.60% at moved-warm. Preserve
cold/moved iteration differences without attributing them to force code.
All 288 same-geometry independent E/F pairings pass; maximum errors are
1.04e-10 energy and 3.18e-11 force. Native through-f, memcheck and initcheck
pass. Matched GPU4PySCF warm/moved-warm medians are 13.629942/10.086490 s,
so the candidate remains far from #1895's ten-second goal.

Full samples, histories, source/binary receipts and the decision summary remain
under `/data/jzzeng/qc-1895-order5-weighted-forces-20261005/.artifacts/` in
`endpoint-5905/`, `gpu-5905/` and `endpoint-decision.json`. Current-master source
and binary qualification must complete before extending the shipping change.


## Shipping qualification (Slurm5922)

The master-based shipping code was frozen at
`e828d51bbb9e9b39e4a4c388e6b4c9c6641cfe0d`, source identity
`c084c16a2d87344a4914c13ae148792d94be461a3ceb50e17bec58bc0101f247`.
All seven scientific/dispatch files match the composed candidate byte-for-byte.
The native build and all six stationary AOT modules use verified ccache launchers;
27 focused host tests, native through-f values/derivatives, memcheck and initcheck
pass. Both sanitizers report zero errors. The host coverage comprises 54,864
independent Hermite displaced-value coordinate checks and 746,496 retained-adapter
comparisons; do not describe the combined count as entirely independent.

`benchmarks/results/pbe0-weighted-order5-20261005/` retains every endpoint sample,
actual SCF history, source reconstruction patches, binary/source receipts,
shipping transplant and a source-matched intrusive warm profile from Slurm5917.
Both reconstructed source identities match the original measured arms. The
verifier enforces a complete geometry/phase/repeat inventory before pairing.
Shipping-master full endpoints were not measured: retain that boundary rather
than assigning composed timings to master. The next-step engineering plan is
`../../proposed/2026-10-05-pbe0-post-order5-roadmap.md`.
