# Decision: compiler-owned CUDA lowering profiles for XC density and potential

Status: implemented
Date: 2026-11-07

## Problem

The common native lowering selector already understood candidates, precision,
resource limits, complete phase costs and qualified incumbents, but the KS method
selected indexed FP64 GEMM earlier using a full device query and local shape
heuristics. This duplicated capability ownership and prevented the prepared
provider from considering expected reuse before allocating optional resources.

## Decision

Extend the existing shared CUDA target record with FP32/FP64 throughput ratio,
core/memory clocks and memory bus width. Both selected attributes and the
authoritative property fallback publish those facts. They describe capability,
not a benchmark or an independently established Tensor Core arithmetic mode.

The compiler XC density and potential adapters declare their shared qualification
bounds as a `CudaLoweringProfile`. The generic runtime profile checks hardware and bounded
workload axes; the existing `select_native_lowering` remains the only candidate
cost/resource selector. The method supplies precision admission, optional
capacity and expected SCF reuse, not GPU facts or an algorithm flag. Internal
backend recipe flags remain for typed execution and explicit native test probes.

The prepared panel and symmetric products select before provider/cache acquisition
and reselect through the same mechanism after a resource refusal. Complete costs
use prepare plus expected replays of every execution phase. Unknown costs retain
the previously qualified profile explicitly, rather than becoming free phases.
No new timing constants, hardware-name branches, startup autotuning or persistent
device capability cache are introduced.

This follow-up is stacked on the unmerged #3041 density-GEMM implementation,
not a replacement for its endpoint qualification. Density keeps the ordinary
reservation distinct from the automatic optional quote; a hardware/workload miss
cannot silently expand the former. Private test hooks exercise small numerical
fixtures with the production FP64 math policy, replacing the public recipe bool.
The profile emitter participates in both portfolios' compilation identities;
scientific graphs and algorithm identities remain unchanged.

Additional consumers reuse the same facts/profile boundary: DF packed occupied
projection, occupied Gram capability admission and automatic DFT SCF/force tile
preparation. Projection retains its existing scalar-FP64 throughput qualification
and AO/auxiliary/rank/scratch-row bounds, including pre-Ampere eligibility. Gram
retains the stronger matrix capability gate independently of explicit projection
overrides. The DF plan freezes its facts before capture; no shared context/device
cache is introduced. Automatic tile policies and explicit tile choices are
unchanged. These are capability/admission integrations, not a claim that Gram or
tiling now has canonical complete-phase candidate costs.

The remaining property queries in COSX value/derivative setup, post-HF transforms
and DF triples mostly pass architecture into prepared contexts; RHF compaction
queries an authoritative launch-grid limit. They are not additional automatic
GEMM promotions and are deliberately not bulk-rewritten here. COSX contractions
and structured SCF products already expose common candidate bindings, but their
unqualified production offers need complete endpoint evidence before a shared
cost model may enable them. Direct Coulomb scheduling already consumes the
shared target record; its launch-policy selection is distinct from contraction
cost ranking. These are follow-up integration opportunities, not delivered
performance qualifications.

## Invariants

- Retain XC's existing large-local, strict-FP64 qualification, pre-Ampere and
  low/unknown-FP64 exclusions, and the 128 MiB optional KS ceiling. Do not impose
  that matrix gate on the separately qualified scalar-FP64 DF projection.
- Public numeric promises and explicit reference-memory budgets retain their
  bounded owner; the compiler does not bypass method resource/scientific gates.
- Preserve the original TensorIR algebra, algorithm domains, FP64 GEMM math
  mode, indexed publication, user tile choices and generated execution fallback.
- Do not reinterpret unrelated CUDA failures or failed live cleanup as a
  recoverable capability/resource miss.
- Do not infer performance gains or extend the evidence domain from peak clock,
  bus width, scalar FP64 ratio or host-double tests.

## Rejected alternatives

Keep per-method GPU predicates: duplicates hardware ownership and makes new
devices a method-by-method maintenance exercise. Relax the FP64 threshold for
consumer GPUs: unsupported by endpoint evidence. Invent a universal FLOP-rate
cost model or cold autotune: would promote unqualified schedules and add startup
work. Create a second selector: unnecessary because the native binding already
provides bounded admission, phase-cost ranking and fallback semantics.

## Evidence and remaining qualification

Executable host probes compile the generated adapter with the actual prepared
context/executor and explicit provider doubles. They cover hardware/workload
exclusions, bounded budgets, expected-reuse cost crossover, packing overhead,
unknown-cost retention, optional-query errors, allocation/ledger refusal and live
cleanup failures. Existing independent NumPy symmetric-product tests and the
native long-double/map/capture suites remain intact.

The final focused host regression passes 819 cases, with 128 optional/toolchain
skips. This includes 58 actual density/potential adapter admission/cost/fallback
cases. Compiler structure (588 modules) and provider-selection ownership
(14 files) report zero errors. Ruff and pinned clang-format 23.1.3 pass.
Compilation uses verified sccache 0.16.0 and retained shared caches. The successful
incremental build retains receipts for 130 cache hits in its earlier compilation
stage; subsequent link/format-only rebuilds reuse that same cache.
No cache is cleared, disabled or weakened to obtain hits.

The initial CI run exposed the pinned formatter's wrapping difference and a
stale live-catalog snapshot. The supported catalog generator refreshes exactly
the two current-input identity fields; capability summaries, retained campaign
bytes, previous baseline and `stale=True` are unchanged. No historical receipt
is promoted or relabeled. An additional 139 profile/panel/catalog tests pass.

The full non-fast Release library and standalone native DFT driver compile,
including production DF setup, occupied-exchange and DFT method units. The
standalone driver explicitly links the shared device-facts and component-trace
owners required by the generated XC unit. Stationary CUDA/CPU force AOT targets
are disabled in this validation build; this is not full force qualification.

Source-matched CUDA 12.9.1/sm_120 qualification on 2026-11-08 uses finite Slurm
allocations on allowed nodes; `node3` is not used. Both devices pass the
independent complete-property oracle, invalid-ordinal cleanup and preservation of
the scheduler-assigned current device. Both pass the native `--potential-lowering`,
`--indexed-potential`, `--matrix-schedule`, `--point-batches`, `--local-ao`,
`--density-provider`, `--indexed-density-provider` and
`--qualified-indexed-density-provider` gates: scalar/CPU E/V oracles,
variational checks, tails, maps, capture/replay, semantic work and bounded
provider/cache fallbacks.

| Device/node | Final facts/Gram/native XC/public/precision job |
| --- | --- |
| RTX 5090 / node5 | 2011 |
| RTX PRO 6000 Blackwell / node2 | 4113 |

The Gram fixtures cover `(n,k)` = `(17,79)`, `(33,32769)`, `(385,65539)` and
`(773,131075)`, signed weights and changed-input captured replay. They compare
every output with cuBLAS and selected entries with an independent long-double
oracle, with a `2e-10` gate. Maximum cuBLAS errors are `2.49e-13` on node5 and
`2.24e-13` on node2. This validates unchanged algebra, not a speed promotion.

On each device the public endpoint suite has 15 passes and one failure. Passing
cases include independent PySCF semilocal references and cold/warm/moved-geometry
DF PBE/PBE0 endpoints. The unchanged
`test_cuda_scaled_dft_force_request_remains_unqualified` expects `ValueError`
about unsupported force properties, but the existing admission in
`src/methods/dft_method_preparation.cpp` first rejects unqualified scaled/global
composition with `NotImplementedError`/status 3. That failure occurs before the
lowering changes and is retained, not weakened or counted as a pass. Endpoint
suite elapsed times are 264.62 s (node5) and 241.39 s (node2), not paired
performance comparisons. Both devices separately pass all 258 precision tests:
232 host policy/ABI tests and 26 independent pinned native scientific endpoints.

A supplementary pre-format node2 resource run (Slurm 4112, source identity
`35dedcc0b4b25d299e543680b5715f8ce28399de003b5159826e605eff953f3b`)
has 152 passes and nine failures.
Two CUDA named-force tests need the omitted packaged `pbe_rks` stationary-force
artifact. Seven CPU named-correction tests reach the existing compatibility
correction owner, which those tests explicitly forbid. Neither frontend owner
is changed here; these results do not qualify complete corrected-force endpoints.
The raw failures and all test coverage are retained rather than silently
deselecting tests or promoting an unavailable production force path.

Source identity:
`26f450c226dd16b4641629cf821d0f20fe5734051eba90424ead4704253b4a01`.
The official inventory, retained source receipt and actual library getter agree
on each device. Both use library SHA-256
`e5121ad29999c40cd37d0968b7d1f4f3de4eeaaecb14997dc9d843b35a42d3f6`
and native XC driver SHA-256
`748a312142113f14dfaca91cd07cc77f586d815ff7a624648667dee912e47d7f`.
Raw build, cache, Slurm and qualification receipts are retained under
`/data/jzzeng/compiler-lowering-20261107/` on the corresponding nodes, with
`stacked-` prefixes for the final qualification and separate earlier attempts.
Older master-based trials and the pre-format source-35 receipts are retained
separately, not relabeled as this source. Keep the
explicit toolkit ahead of the shared Conda environment's incomplete NVCC in
`PATH`; the build pins `/group/software/cuda-12.9.1/bin/nvcc` directly.

No accessible A100 allocation or new complete-endpoint performance comparison
is claimed. The host capability probes cover its existing qualification but
cannot promote a new hardware/workload cost model. Full SCF/force timing evidence
remains necessary before changing an incumbent or expanding the qualified domain.

## Revisit when

Source-matched complete A100/consumer-GPU endpoints support a workload/tile/reuse
cost model, or another contraction recipe needs additional capability gates.
Integrate that evidence into compiler-owned offers rather than adding method
branches or treating theoretical bandwidth as measured performance.

## References

- [Original indexed GEMM qualification](../performance/2026-11-01-capability-selected-xc-and-gram.md).
- [Current lowering contract](../../../../docs/developer/lowering_providers.md).
- [Endpoint qualification requirements](../../../../docs/maintainer/performance_engineering.md).
