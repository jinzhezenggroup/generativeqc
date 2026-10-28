# Decision: prepare XC coefficients and share KTBM workers by default

Status: implemented
Date: 2026-10-28

## Problem

The [first shared-worker implementation](../architecture/2026-10-27-xc-parameter-worker-sharing.md)
removed duplicated KTBM executables but made point evaluation about 3% slower.
Keeping mathematical coefficients symbolic through AD lost coefficient-only
folding, while indexing a device constant table introduced dynamic address loads.
The restricted worker grew from 116/122 to 140/148 registers.

## Decision

Prepare products of direct parameter leaves and literal factors with the existing
Graph algebra, after AD and before equal-spin projection. Preserve input-dependent
operations, domain guards and lazy branches. Both spin paths use the same ordered
parameter row. Pass that immutable row by value from a host constexpr table into
CUDA launch parameter space, with fixed offsets in the device worker.

The current audited family produces 60 launch values from 12 registration
coefficients. That count is an implementation result, not a public contract.
No runtime host scientific evaluation, mutable device coefficient storage,
per-point functional switch or cross-unit device link is introduced. Canonical
coefficient-bound CPU/CUDA admission identities remain separate from this
alternative execution lowering; worker identity binds its body and complete table.

Sharing defaults on inside enabled automatic CUDA AOT packages. Automatic AOT
itself remains opt-in. CMake `GENERATIVEQC_SHARE_AUTOMATIC_XC_CUDA_PARAMETERS=OFF`
and generator `--no-share-parameters` retain literal specialization. Singletons,
unmatched contracts and other functional families retain their existing emitters.

## Rejected alternatives

- Limiting registers with launch bounds did not remove the arithmetic overhead;
  combining the limit with by-value parameters could increase stack use.
- By-value parameters alone reduced the slowdown to approximately 2% but left
  coefficient-only multiplications in each point evaluation.
- `__grid_constant__` did not improve the measured candidate and adds an
  unnecessary architecture/toolchain constraint.
- A generated-text folding prototype reduced the regression to below 1%, but
  production uses Graph operations, including nested product reconstruction.
- Branch-region coalescing was compiled as an exploratory variant but was not
  selected or runtime-qualified. The narrower prepared-product change sufficed.

## Invariants and acceptance

Regrouping FP64 products is not a bitwise-equivalence promise. Independent Libxc
energy/derivative and PySCF complete-KS oracles remain required. Never hoist a
reciprocal, transcendental or input-dependent predicate out of a lazy branch.
Preserve registration authentication, ordinary/batched layouts, physical spin
derivatives, explicit equal-spin fallback and per-launch coefficient ownership.

Promotion requires point median and every spin/layout median at most 1.0 of the
literal baseline, no case above 1.01, and isolated complete-endpoint phase medians
within 2%, with unchanged semantic work. These replace the previous 5% tradeoff
gate; they do not imply that every device has been qualified.

## Evidence

[Source-matched receipts and reduction](../../../../benchmarks/results/xc-prepared-parameters-20261028/README.md)
retain all pilots, paired point samples, isolated endpoints, independent oracles,
compiler/cache records and source reconstruction.

- NVCC 12.9.1, SM120, Slurm PRO6000 on node2.
- All 26 registrations, both spin paths, ordinary/batched layouts and two point
  counts: 208 cases, two randomized paired rounds, 50 samples per arm/round.
  Median ratio 0.989173; maximum case ratio 0.997337.
- Registers 110/118 for both paths, stack 96 bytes and no additional local/shared
  memory. General ordinary SASS has 365 DMUL instructions versus 563 for the
  original shared worker and 410 for the literal KTBM_0 worker. These are static
  instruction counts, not dynamic endpoint work counts.
- Seven complete SCF cohorts, five alternating process pairs, 350 calls. All
  iteration, Fock and precision/work ledgers match. Maximum energy difference
  2.274e-13 Eh; phase median ratios range from 0.993306 to 1.003332.
- All 234 real-GPU independent qualification cases pass. Host tests exercise
  original and prepared parameter rows, including directional derivatives.
- A separate concurrent endpoint pilot had noisy cold/moved timing. It remains
  in the receipt and is not pooled with the isolated qualification campaign.

The follow-up also refreshes the importer/admission provenance missed in the
original PR. Regenerating the pinned bulk catalog leaves both generated outputs
unchanged; it updates their actual compiler-input digests. Synthetic importer
test doubles now accept the parameter-binding API. Integrity and reachable-input
rejection tests remain executable.

## Revisit when

Requalify on other architectures/toolchains, after changing point lowering or
coefficient preparation, or before admitting another parameter family. Preserve
the bounded literal fallback. Do not extrapolate KTBM kernel or package-size
ratios to unrelated mathematics, all functional families or an entire release.

## Catalog discovery follow-up

The [structural discovery decision](../architecture/2026-10-28-xc-structural-parameter-discovery.md)
extends candidate discovery to other formulas while retaining this qualified
KTBM default. Its negative small-batch evidence prevents blanket promotion.
