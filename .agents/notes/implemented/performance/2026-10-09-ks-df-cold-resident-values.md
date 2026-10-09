# Decision: retain bounded method-owned DF values on cold restricted KS

Status: implemented
Date: 2026-10-09

## Problem

A frozen master `4e20f7a7bcfbff770fefd13fa2777f8ab309679c` cold PBE0-DF
calculation took 955.024 s versus GPU4PySCF 1.8.1's 75.220 s on node1's
RTX 5090. This was the 32-water/96-atom spherical def2-SVP fixture (768 AOs)
with cc-pVDZ-JKFIT (3712 auxiliaries), FP64, and an unpruned 48x16x32 grid
(2,359,296 points). Energy/density tolerances were 1e-12/1e-10. Native and
reference screening tolerances were 1e-12/1e-14. Preparation was 67.336 s;
SCF and energy publication were 887.688 s. Native used 24 iterations/Focks;
the reference used 42 iterations/43 Focks. The independent energy error was
1.36424e-11 Eh and native physical residual was 5.97671e-13.

A separate two-Fock Nsight/DF-ledger diagnosis found 19,704,840,192 raw
source auxiliary evaluations in the first general K (nine complete raw tensor
passes), then 2,189,426,688 per occupied K (one complete pass per iteration).
The occupied exchange path already existed: whitening/source regeneration,
not failure to choose occupied exchange, was the dominant wasted work.
Coordinate-major one-electron derivative export also consumed about 60 s
despite an energy-only endpoint.

## Decision

The validated singleton restricted KS occupation reservation can use the
same dense-to-packed-single capacity crossover as RHF. It does not acquire
RHF-owned SCF factors or density provenance. Fully resident dense remains the
preferred representation; packed B is considered only to replace a streamed
dense source. Optional full U remains charged and may be dropped. The actual
selected representation is diagnostic/cache identity, separate from the
requested selector, so automatic replay matches and explicit selector changes
invalidate correctly.

Only live automatic DF budgets may rebalance the existing total value/response
envelope for this method-owned crossover. Keep at least 20% for response and
a conservative reserve for four metric matrices, AO/density temporaries,
two all-Q occupied projections, a 64-Q W panel, and metadata. The native packed
planner must admit the resulting value allowance. Rebalancing is accepted only
when it removes streaming or adds complete method-owned U to a packed resident
B. Complete U preserves the existing final-state force lease and prevents
rebuilding an expensive fitted projection merely because the original split
dropped U while leaving unused response capacity. Otherwise preserve the
original split/streamed fallback. No total-cap or device-headroom increase is
permitted. Explicit budgets, absent probes and generic/UHF callers retain the
old policy. Native response allocation continues enforcing its own cap.

CUDA fitted energy preparation with retained future force capability no
longer exports all H'/S' matrices. The ordinary stationary CUDA force consumer
contracts the final token-checked D/W directly. If that optional consumer is
unavailable or exceeds its allowance, the prepared owner materializes H'/S'
with the same CUDA exporter and retains them for the exact host contraction.
CPU and explicitly requested derivative Fock preparations stay eager.

This supersedes only the automatic-layout boundary of
[the projection-reservation repair](2026-10-02-ks-packed-projection-reservation.md).
Its producer/consumer provenance and one-shot projection-lifetime guards remain.

## Rejected alternatives

- Selector-only promotion: a first real 96-atom candidate still streamed and
  regenerated all ten raw tensor passes across two Focks. Value/response
  partitioning, not just representation selection, must admit the owner.
- B-only admission: a converged intermediate reduced energy cold time to
  87.251 s (three-sample median), but ordinary force response still took
  228.227 s without complete final U. Independent force error was
  1.576e-10 Eh/bohr. Numerical parity and zero SCF regeneration do not alone
  qualify complete-endpoint force work.
- Inferring occupation authority from AO/electron dimensions in generic callers.
- Stealing the complete response allowance or raising an explicit budget.
- Disabling future force capability to improve an energy-only timing.
- Changing precision, grid, convergence tolerances, screening, or initial density.
- Calling diagnostic/unfinished runs clean converged endpoint measurements.

## Evidence and reproduction

`benchmarks/pbe0_df_cold.py` journals clean fresh-process endpoints for native
and independent GPU4PySCF. Run each invocation through a finite-time Slurm
allocation, preserving scheduler visibility. Set `GENERATIVEQC_BENCHMARK_SOURCE`
to the frozen source identity and use the same environment/library/cache for
all native repeats. It excludes imports, CUDA context initialization and native
Calculator construction, but includes preparation through synchronized public
energy or energy-plus-force return. Persistent disk caches are retained.
Owner closure, module unloading and process shutdown follow the measured
endpoint; these numbers are not whole-process or `Calculator.singlepoint`
teardown-inclusive wall times.

Host tests execute the real tile/resource planners for resident, constrained,
explicit, absent-probe, batched and arbitrary-density cases. The CUDA regression
uses an independent E/F oracle and a real 112-MiB dense-to-packed crossover,
cold/warm replay, and an artificially constrained 4-KiB one-electron response
allowance to exercise on-demand host fallback and one-time derivative caching.

The final Release sm_120 binary was qualified through Slurm on node1/RTX 5090.
Three alternating fresh-process pairs produced:

| Endpoint | Native | GPU4PySCF 1.8.1 |
| --- | --- | --- |
| Median energy cold | 75.470946 s | 65.740917 s |
| Energy cold range | 75.436222..75.620645 s | 65.736456..65.886152 s |
| Iterations / Focks in each repeat | 25 / 25 | 37 / 38 |
| Cold energy plus force (one sample) | 298.174049 s | 84.631802 s |
| Force phase (one sample) | 222.560308 s | 10.752989 s |

The original master comparison has one completed native baseline sample;
do not treat its interrupted second repeat as evidence. The energy endpoint
speedup is 12.6542x against that 955.023585-s sample; the remaining matched
GPU4PySCF energy-cold ratio is 1.1480x. No per-Fock normalization is applied.
Maximum energy error across the clean population was 2.18279e-11 Eh;
maximum clean force error was 1.57244e-10 Eh/bohr; maximum native physical
residual was 3.824e-13. Grid, precision, tolerances and basis identities are
unchanged. Force acceptance gates remain 1e-8 Eh and 3e-7 Eh/bohr.

A separate converged full E/F ledger recorded one 295,296-pair resident B
materialization, 50 J/K rows across 25 Focks, zero SCF raw source auxiliary
evaluations, and one completed final-U force hit with all 288 coordinates
returned. Its value-plan peak was 16,183,938,809 bytes under a
17,340,963,226-byte value allowance, with a 4,335,240,806-byte response
allowance; the unchanged resolved total was 21,676,204,032 bytes. These are
DF provider bounds, not whole-KS/CUDA-driver memory claims.

An additional 8-GiB synthetic device-pressure E/F run exercised the resident
B/zero-full-U fallback: total 15,233,753,088 bytes, value 12,187,002,471,
response 3,046,750,617, value-plan peak 12,176,281,337. This was the
intermediate B-only binary; the final driver's added optional-U admission
cannot fit U in this same envelope and preserves that resource plan. It is
not a clean timing sample. The positive 112-MiB CUDA regression separately
qualifies the final binary's zero-U fallback and on-demand one-electron export.

Final focused qualification: 34 host tests and 31 allocation-scoped
CPU/CUDA tests passed; one explicitly unsupported case skipped. An additional
RHF resource suite produced 11 passes and two failures, reproduced identically
on unmodified master. Those old assertions require streamed values at 24 MiB
and a non-source-backed force owner; both are incompatible with its already
present automatic packing/source policy. They are not repaired in this change.

Ignored local receipts, traces, summary and focused patch are retained in
`.artifacts/pbe0-df-fix/`; remote raw evidence is under
`/data/jzzeng/qc-pbe0-df-cold-20261009-4e20f7a7b/results/` on n1. Completed
final receipts use the `candidate-v4` prefix; failed/partial earlier candidates
are retained separately. Compiler caching used verified ccache 4.5.1.

After the last descriptive-variable-name cleanup, the rebuilt library's `.text`
and `.nv_fatbin` sections were byte-identical to the qualified binary; only
build/source identity metadata changed. The final library SHA-256 is
`ffcf2c40df74ff39061ff019bd94aaa175adeba63cfd4af0034289126b90fc5c`.
A fresh final-binary 96-atom energy smoke returned in 74.921026 s with 25 Focks,
energy -2441.5968865132486 Eh and residual 3.82255e-13. The same final-binary
allocation-scoped regression suite again passed 31 tests with one skip; the
11-test host budget/source subset also passed. These smoke results supplement,
rather than replace, the alternating three-pair qualification population.

Remaining force performance is explicitly not solved: the full ledger spent
about 0.60 s in paired one-electron response, 2.05 s in the final-U exchange
response, and 72.21 s in ordinary Coulomb response. Other stationary/grid work
accounts for most of the remainder of the 222.54-s traced force phase. A future
repair should profile the complete force endpoint and avoid repeatedly
inverse-projecting fitted panels for J-only response; final-U reuse alone does
not remove that work or the remaining stationary/grid cost.

## Invariants and revisit conditions

Keep full FP64 mathematics, auxiliary/metric/moving-grid force response and
independent numerical gates. Retain complete endpoint timing and semantic source
work counts, not just kernel speedups. The response reserve is a conservative
staged-work model; changes to response allocation, occupation provenance,
simultaneous owner count, or projection lifetime require revisiting it and
requalifying constrained complete endpoints. Further source-integral and XC
kernel improvements are separate from removing this repeated-work defect.
