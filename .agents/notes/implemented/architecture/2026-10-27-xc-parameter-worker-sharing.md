# Decision: share audited automatic XC parameter-family workers

The opt-in runtime tradeoff below is superseded by
[prepared launch coefficients](../performance/2026-10-28-xc-prepared-parameters.md).
This note retains the original measurements and rationale.

Status: implemented as an opt-in artifact-size tradeoff
Date: 2026-10-27

## Problem

Distinct functional registrations can bind the same Maple formula to different
scalar parameters. Literal specialization duplicates E/vxc arithmetic and spin
traversal. KTBM provides 26 registrations with 12 nonzero scalar coefficients;
registration count is not a mathematical or executable-ABI census.

## Decision

Lift explicitly requested scalar `params_a_*` bindings into the existing Graph
before differentiation, retaining their registered values and coefficient-bound
identity. Physical-feature derivatives treat these coefficients as constants.
The normal literal path is unchanged. Worker thresholds, array parameters and
preprocessor controls cannot be lifted through this interface.

The build option `GENERATIVEQC_SHARE_AUTOMATIC_XC_CUDA_PARAMETERS` defaults to
OFF. Enable it explicitly to trade a measured small point-evaluation overhead
for a smaller package. Literal specialization remains the default because the
user requested no default runtime regression. This is not a free performance
optimization or a default promotion. Initially only the audited KTBM entry
proposes sharing groups. Both polarized
and equal-spin emitted bodies, ordered coefficients, layouts, ingredients and
work-domain/version must match exactly after removing only the emitter-owned
function identifier. Registration entry membership alone cannot establish a
shared executable. A singleton or any nonmatching registration retains ordinary
specialization; no registration or product is removed from admission.

One worker owns four existing point traversals (ordinary/batched and
polarized/equal-spin), using an immutable constant-memory table with one uniform
row per launch. Per-registration host wrappers preserve functional-code,
scale, count, layout, response and equal-spin validation. Only host launchers
cross shard boundaries; there is no device link or runtime compiler dependency.
Registration-only units avoid the tensor runtime header, which would otherwise
emit unrelated conversion kernels into every CUDA translation unit.

The native descriptor retains canonical literal scientific identities because
CPU admission compares the registered mathematics, not an alternative execution
lowering. Shared worker identity hashes its complete source and ordered table;
each wrapper source then binds that worker and its immutable row. Private
visibility is scoped after runtime includes to avoid hiding runtime resource
ledger/TLS entities. Selected package order and duplicate requests cannot alter
rows or symbols. Different subsets have distinct worker identities.

## Rejected alternatives

- Aliasing by functional name, common ingredients or source file alone cannot
  establish the derivative and execution contract.
- A generic per-point functional interpreter would add dispatch and discard
  specialization without evidence of acceptable endpoint performance.
- Replacing canonical CPU science hashes with symbolic parameter graph hashes
  breaks CPU/CUDA admission even when the bound equations are equivalent.
- Applying the measured KTBM savings to PBE or unrelated functional families
  ignores different folding opportunities, predicates and derivative ABIs.
- Hiding runtime includes to make all symbols private changes resource ownership;
  only generated private definitions receive hidden visibility.

## Evidence

On NVCC 12.9.1 / SM120, the selected 26-registration package has four point
kernels instead of 104. Registration-only includes also stop duplicating tensor
conversion kernels. Registry objects shrink from 17,416,384 to 1,093,560 bytes
(93.72%); the same configured complete library shrinks by 2.24%. One controlled
compile trial with verified sccache forced recaches (cache contents retained)
takes 22.15 versus 10.17 seconds. Code generation grows from 9.33 to 11.63 seconds;
combined time improves by 30.7%. This is not a full-catalog size ratio or a
multiarchitecture/timing-distribution claim.

Sharing is measurably slower: 208 paired point cases show a median 3.09% and
maximum 4.18% overhead. Registers increase, especially for equal-spin workers.
Seven complete SCF cohorts, three fresh-process pairs each, have at most 1.51%
median overhead, identical semantic work and energy differences below 1.71e-13.
They pass the explicit 5% qualification bound. The user's stronger requirement
of preserving default performance therefore wins: sharing remains opt-in, and
literal specialization remains the default rather than calling this zero-cost.

Validation passes 635 host tests (34 skips), 15 provider contracts, 234 real-GPU
cases and two actual-provider receipt-export checks. Independent pinned Libxc
and PySCF gates cover every KTBM row, physical coefficients/directional
derivatives, work-domain boundaries, layout/error handling, AO/XC/Fock and full
RKS/UKS cold/warm/displaced SCF. Compiled descriptors retain canonical CPU science.
The explicit option produces all 34 qualified generated files unchanged.

[Source-matched receipts and reproduction](../../../../benchmarks/results/xc-parameter-sharing-20261027/README.md)
retain timings, numerical/work data, resource growth, controlled compilation,
source reconstruction and failed/incomplete pilot limitations. Other families
remain outside this execution optimization without restricting their admission.

## Revisit when

Expand a family only after complete parameterized expression and ABI comparison,
independent numerical acceptance, measured object/compile savings and complete
endpoint plus point-worker performance qualification. Preserve specialization
for cases where lost constant folding costs runtime performance.

## Catalog discovery follow-up

The [structural discovery decision](2026-10-28-xc-structural-parameter-discovery.md)
adds catalog-wide candidate discovery after literal specialization. The measured
KTBM default remains separate from that explicit qualification policy.
