# Decision: prepared FP32 intervals for bounded Direct admission

Status: implemented (explicit opt-in)
Date: 2026-10-23

Supersedes [per-visit block rejection](2026-10-23-conservative-block-screening.md).

## Problem

The first FP32 experiment repeated conversion on every block visit and repeated
the incumbent FP64 predicate on every survivor. The molecular force census
explained its poor return: PBE0 water-96 rejected no block pages; r2SCAN-3c
rejected only 30,112 of 16,263,744. Block Schwarz maxima and the system-wide
density maximum are conservative but loose. Faster arithmetic alone cannot
reduce the unchanged integral work.

## Decision

Keep `GENERATIVEQC_DIRECT_FP32_BLOCK_SCREENING` default-off. Prepare one upward
FP32 bound per block and system alongside the existing FP64 reductions. Recover
the lower endpoint as the predecessor float. Check lower products first to
certify the common survivor, then upper products to certify rejection. Read the
original FP64 metadata only for overlapping intervals or unsupported inputs.
This removes repeated FP64 screening for proven survivors as well as repeated
conversion, while preserving the incumbent task set exactly.

The three bounded consumers share this helper. Exact shell predicates, the
24-byte shell-density ABI, Schwarz preparation, ERIs, J/K, energies and forces
keep their existing precision. Generated resident streams and class-page gates
remain outside this experiment. The standalone synthetic probe retains the
old per-visit and prepared rejection-only controls, and adds interval admission
as its seventh arm. INT8 and actual INT8 WMMA remain experiments without a
production backend selector.

## Numerical invariants

Each stored upper bound encloses its nonnegative FP64 source. Its predecessor
is a lower bound, including exact-representability cases. Positive subnormals
are floored to `FLT_MIN`; that floor has lower endpoint zero. Directed FP32
products enclose each sequential FP64 rounding boundary. Lower products must
meet upward thresholds for acceptance; upper products must lie strictly below
downward thresholds for rejection. Equality and overlapping intervals refine.

Upward thresholds need the same positive-underflow floor as bounds: in FTZ
builds, a subnormal upward threshold can otherwise compare as zero and falsely
accept. Invalid/overflowed prepared inputs use infinity sentinels and refine.
Compatibility compilers without qualified NVCC rounding semantics always
refine. The arithmetic fixture checks both outcomes against sequential host
FP64 and independently checks every rejection with extended precision.

## Storage and ownership

Opt-in owners charge four additional bytes per block and system, in addition
to the retained FP64 metadata. Preparation adds no launch or synchronization.
Geometry preparation rebuilds block bounds; every density reduction refreshes
the system cache. HF arena sizing and retained replay freeze the same selector.
Generated exchange owners allocate private caches within their charged budget;
allocation failure retains the existing bounded fallback policy.

Borrowed batch pointers must not propagate a writable cache into a new owner.
In combined J/K and range-separated prepared execution, K's cache represents
spin-aware exchange metadata, while J reduces total density into its own FP64
metadata. J must not overwrite K's cache: a cached bound paired with another
owner's FP64 value invalidates interval refinement. This is why the Coulomb
geometry constructor clears borrowed pointers, and the J density launch clears
the optional cache pointer on its local batch copy.

## Qualification and limitations

Both FTZ settings pass 3,061,731 cases with zero task mismatches and nonvacuous
accept/reject/refine coverage. The integrated run passes 35 Python tests,
including independent UHF forces, signed raw J/K against independent AO
integrals, full/SR/LR shared-owner matrices at original and moved geometries,
and resource identity/budget/replay coverage. Compute Sanitizer covers arithmetic, all seven probe arms,
and complete C7 cold/warm/moved E/F execution with the caches enabled.

The [current evidence bundle](../../../../benchmarks/results/direct-screening-20261023/README.md)
binds clean three-round PBE0/r2SCAN-3c water-96 and C7 RHF endpoints, independent
references, separate diagnostic profiles and repeated synthetic controls to
the compiled source. The initial rejection-only measurements remain recoverable
in commit `2fe65ffff7c92ef6b7c98057e21dd95f5b715315`. All accepted force/reference
arrays and geometries fit within the unchanged retention budget. The reader
recomputes independent and cross-arm errors, as well as work counts. Full raw
logs and profiler traces remain outside Git.

Checking lower products first favors the molecular all-survive block workload,
but adds work in a sparse synthetic workload dominated by rejection. Prepared
interval admission is therefore not a universal screening speedup. At 8192
bounds, the final all-survive force probe drops from 1.474 ms for per-visit FP32
to 0.473 ms for intervals (FP64: 0.566 ms), but sparse intervals take 0.619 ms
versus 0.553 ms for FP64. All molecular cold/warm median differences are below
0.7%, with no demonstrated endpoint gain; retain the default-off selector. Fused
screening/compaction/integral kernels do not provide isolated instruction timing.
C7 is Direct RHF/def2-SVP, not CCSD(T)/CBS, and its HF owner is outside the
existing DFT quartet census.

## Revisit when

Promote only after source-matched complete endpoint gains survive repeated
cold/warm/moved measurements and independent numerical/task-set gates. If the
remaining overhead is immaterial, improve bound selectivity or the dominant
integral/Schwarz/XC owners instead of accumulating more screening machinery.
A 12-byte shell-density ABI needs separate near-threshold and generated-consumer
qualification. Production INT8/WMMA also needs bounded staging and actual
compaction/integral consumers, with packing and lifetime costs included.
