# Decision: keep DF host replay outside CUDA admission headers

Status: implemented; whole-wheel and separate Direct endpoint gates remain open
Date: 2026-11-16

## Problem and ownership

The five generated DF action/core/hoisted/pair/AO headers combine shared runtime
shape, work and admission metadata with ordinary C++ evaluator bodies. Their
CUDA declarations include those headers, so NVCC parses unused host arithmetic
again in each generated owner and in handwritten consumers. This is the same
frontend ownership problem already addressed for UCCSD and RCCSD, not evidence
that DF scientific programs or admission queries are unnecessary.

## Decision and invariants

Keep exact capacities, equation identities, work/contraction queries, pair
bounds, output/input types, canonical denominators and CUDA declarations shared.
Guard only the 16 host evaluator definitions with `#ifndef __CUDACC__`.
Ordinary C++ retains every evaluator, including one-Q, batched, hoisted,
occupied-pair/norm and AO factor/pullback/block operations. No scientific
arithmetic, launches, provider branches, precision, reduction ordering, shape
domain, fallback, method or architecture changes.

Do not apply this globally to the shared evaluator emitter: other generators
intentionally retain host consumers in CUDA translation units, including the
RHF preconditioner. No new compatibility shim or monolithic generated unit is
needed. The handwritten DF consumers compile against their metadata-only view.

## Complete generated-owner evidence

Freeze before/after headers and byte-identical CUDA sources under one common
source path. Derive compilation arguments from the configured Release Ninja
commands, retaining `sm_80`, `sm_90`, `sm_120` and compute-80 PTX. NVCC 12.9.86
and verified sccache 0.16.0 compile all ten complete objects.

The actual `sm_80` NVCC preprocessed inputs, with line directives removed using
the same command in each variant, shrink as follows:

| Complete generated owner | Before bytes | After bytes | Reduction |
| --- | ---: | ---: | ---: |
| DF virtual/response actions | 4,317,431 | 4,091,700 | 5.23% |
| DF retained core | 5,914,368 | 5,526,401 | 6.56% |
| DF hoisted auxiliary | 6,213,958 | 5,738,607 | 7.65% |
| DF occupied spectator pairs | 5,974,579 | 5,454,497 | 8.70% |
| AO fitted-source maps | 3,691,958 | 3,683,499 | 0.23% |
| **Total** | **26,112,294** | **24,494,704** | **6.19%** |

All **five complete object files are byte-identical**. Separate extraction also
verifies all 20 device images: one PTX and three architecture-specific CUBINs per
owner. Ordinary C++ preprocessing of the shared header closure is byte-identical
at 2,760,002 bytes. These are one-off migration checks, not required-CI body or
generated-source fingerprints.

The handwritten `src/cc/cuda_solver.cu` and `src/cc/df_lambda_cuda.cu` also compile
with the retained three-architecture inventory. All seven candidate objects
pass one combined relocatable host link. That is not a complete native library,
manylinux repair, installed-wheel test or physical-GPU endpoint execution.

Before/after owned foreground cache statistics distinguish the work: baseline
has zero hits and 40 CUDA-stage misses; candidate has 15 CUBIN hits and 25 stage
misses. The shared cache is neither cleared nor weakened. Cache asymmetry and
overlapping host tests mean observed durations are not cold-build speedup
evidence. Claim reduced frontend input, not a measured whole-wheel speedup,
compiler-memory reduction, smaller packaged device code or faster GPU method.

## Executable validation

The new compiled host/CUDA-preprocessor interface probe compares actual
admission queries and checks overflow/short-arena refusals. Ordinary host AO
factor evaluation matches an independent NumPy contraction. This interface
probe does not execute CUDA. Existing DF action, solver, pair, factorization
and AO oracle suites pass **66 tests**; **85 existing GPU/explicit-qualification
cases skip** under their unchanged gates.

All 989 CI shard-ownership cases pass across the initial run and a focused
15-case recheck after staging the new test file. The selector intentionally
ignores untracked files; its initial failures were inventory prerequisites,
not scientific regressions. No shard policy or selection is changed.

The compiler structure check covers 604 modules with zero dependency errors;
Ruff lint/format checks and `git diff --check` pass.

The official Libxc catalog check passes without changing its retained snapshot
or scientific acceptance. Regeneration changes only the already-non-gating
whole-native-tree identity, so avoid unnecessary compressed-snapshot churn;
all actual scoped dependency audits and qualification requirements remain.

Local ignored evidence is under `/tmp/gqc-wheel-df-boundary/`: frozen closures,
exact compiler commands, source-equivalence audit, objects/device images,
compiler-cache statistics, host/interface/oracle tests and combined link.
`compiled-report.json` has SHA256
`6914477261d967897a98b6171d7ef787b183be1c17a4d520e774ce1229a7e10f`.

## Revisit when

Revisit if a CUDA owner acquires a deliberate host evaluator consumer. Make
that ownership explicit and retain complete compiled-consumer qualification;
do not silently hide metadata or globally re-expose all host arithmetic. Larger
compile-work targets require their own complete-owner and endpoint evidence.
