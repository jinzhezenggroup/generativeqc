# Candidate: per-item budget-selected direct-HF derivatives

Status: numerically qualified candidate; material endpoint speedup still unproved
Date: 2026-10-14

## Ownership and arithmetic

The README HF endpoint has no XC consumer. Reusing the RKS qualification's grid
counts would fabricate evidence. Instead, the HF force owner consumes the v2
application request only after its existing final density/Fock/orbital token is
force-consumable. The candidate domain is restricted all-electron direct HF,
24--96 AOs, through d shells, 2--48 atoms. Tight budgets, DF, ECP, unrestricted
states, physical-reference exports and unqualified shapes retain their ordinary
arithmetic. The scientific integral and force schedules are not changed between
strict and candidate endpoints.

Existing compiler-owned weighted order-0..5 consumers select actual float
recurrences and local primitive contraction. Geometry, density weights, Boys
moments, unselected schedules, higher orders, one-electron/Pulay sources,
global accumulation and the final SCF remain double. This is not all-FP32 HF.
Nonfinite candidate tasks retry once in double before scattering any result;
the existing strict retry contract must remain unchanged.

## Evidence ownership

Three uint64 counters per physical item record task attempts, primitive/source
evaluations and strict task retries. The shared consumer derives the item index
from its physical shell pair. Standalone Coulomb plans still have one item;
fused HF cannot copy a bucket-wide total into every result. The arena reserves
and accounts for the bounded 24-byte/item storage through the shared checked
layout, including dry-run capacity queries. Only candidate force execution
clears, updates and downloads it; normal FP64 has no counter kernels or copies.

Private gradient v3 retains the six v2 entries and appends a collection bit plus
those three counters. An absent collection bit is unavailable evidence, not
zero work. Model ID 5 identifies RHF, never an XC qualification. Publication
requires a synchronized successful finite result, a valid final token and
positive actual task and primitive work. It certifies only this gradient, not
an otherwise incomplete SCF operator census. Energy-only and failed items must
not inherit a prior gradient record. A fully retried weighted-only result is
reported as FP64, not mixed. Explicit observable audits still use independent
strict calculator owners and charge their complete work.

## Validation in progress

The focused host parser/policy/provenance suite passed 97 checks. Expanded
transport/compiler checks passed 164 with 65 CUDA cases skipped on the host.
The full touched-file pre-commit run passed after registering the CUDA driver's
narrow dependency on the CUDA-free admission header. An attempted local HF
resource regression run failed two fixtures because this checkout has no local
native library; those suites are included in scheduled validation instead.

CUDA tests cover independent PySCF forces and energies at 24/96 AOs, Cartesian
and spherical water, ammonia, per-item versus fused work, energy-only replay,
frozen seeds, actual geometry changes, invalid-peer isolation and tight-budget
rejection. Complete timings include both public singlepoints and the private
native prepared/frozen/moved transport. The latter does not implement the
missing public PrecisionCalculator prepared-batch API by itself.

Slurm 8291 passed all four public independent HF cases but stopped on two new
fixture errors: public mean-field screening must be positive, and a prepared
call with no coordinates uses its original geometry rather than the preceding
call's override. The test now uses `1e-30` for the near-unscreened counter check
and supplies moved coordinates on every moved replay. No numerical threshold
was relaxed. The timing script follows the same explicit-coordinate contract.

Slurm 8293, library
`2e68f3290a578ac0fc8d3c82d77df89f22a5fd513eb350be92834200c2b0f519`, passed
116 tests including HF public, fused, frozen/moved and rejection tests, then
stopped at two older resource fixtures. Slurm 8294 reproduced both failures
unchanged with milestone library
`88571e9e2a80b43b00521cc161e57e0969077a8311497455288821c6a09d6dff` and its
matching Python, pre-HF library `89c9e3645fc38e1b42cac64cfa9dfc15899bb1a4a18a697dbf1dc8cf78100594`,
and the current candidate. One pins a recomputed-DF footprint 360 bytes below
all three implementations; the other uses a bare KS estimate as the complete
default-calculator budget, missing 17,251,800 host bytes. Do not change these
unrelated fixtures or claim the complete original suite passed. Subsequent
qualification explicitly deselects those two nodes, retaining all raw failures.

The new gradient-v3 ABI bounds/lifetime test and M06-2X candidate-only probes
were added after 8293 collected its tests; they require fresh scheduled runs.

## Final numerical evidence and negative performance evidence

Slurm 8296 passed 151 tests with four explicit deselections: the two reproduced
baseline resource failures and two separately scheduled M06-2X candidates. The
new gradient-v3 bounds/lifetime check passed. Subsequent M06-2X probes failed
before derivative execution because no native stationary profile is linked for
that complete scientific model; this is not a failed numerical tolerance gate.
The full script therefore exited nonzero despite successful HF qualification.

The first prepared timing script incorrectly chose SCF AUTO for the private
HF candidate, whereas the public facade already chooses strict SCF plus the
derivative budget for HF. That mismatch added one mixed and two strict warm
iterations instead of the baseline's single strict iteration. Preserve those
~0.45x warm ratios as a rejected protocol, not the public Fast result. No
production SCF policy or baseline was changed to repair the benchmark.

Slurm 8299 uses the actual strict-SCF HF policy for both public singlepoints and
private native prepared execution. All 12 HF checks passed, including all four
molecules/representations under both Hcore and default automatic Hückel seeds.
The same library as 8293/8296 produced these complete endpoint ratios:

| AOs | Singlepoint | Prepared cold | Frozen warm | Moved | Moved frozen warm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 0.9964 | 0.9877 | 1.0168 | 0.9906 | 1.0182 |
| 96 | 1.0018 | 1.0112 | 1.0212 | 1.0068 | 1.0132 |

These results do **not** establish useful general HF acceleration. In particular,
cold timing is neutral/noisy, not a completed performance gate. Do not describe
five completed README Fast methods or rebrand ordinary scheduling improvements.
Before widening arithmetic, profile the remaining FP64 work. The current shared
weight assembly and Gaussian/Boys construction still use double, as do orders
above five and unselected generated consumers. Any further FP32 candidate needs
new independent numerical gates and the same optimized strict comparator.

Retain `hf-timing-wrong-prepared-auto.json` separately from the corrected
`hf-timing.json`, with the job logs, XML and native/Python identity receipts.
Public prepared-batch policy integration and the full README domain remain open.

## Generated-consumer dispatch hole and rejected owner replacement

Slurm 8300 profiled three frozen 96-AO executions per mode with Nsight Systems.
The generated PPPS resident-bra, PPPP persistent and DPPP persistent force
consumers do not receive `DeviceBatch.direct_force_fp32`. Their strict/Fast
aggregate kernel times were respectively 38.881/39.024, 20.152/20.204 and
16.179/16.285 ms. Native PSSS and order-zero consumers actually received the
request and improved from 16.576 to 13.876 and 13.557 to 11.441 ms. Positive
weighted counters therefore prove some arithmetic, not full order-0..5 coverage.

A candidate routed all fixed-topology orders through five to the existing typed
shared weighted consumer. Slurm 8304, library
`15270b3812170d35d954b0b21e3b3e2dfbc13627e48e2cf604f964ae290415fa`, passed
all 12 HF tests, then compared incumbent strict, shared-owner strict and Fast.
The shared-strict arm used a sub-admission private budget solely to select the
same schedule without narrowing. The candidate was **rejected and reverted**:

| AOs | Endpoint | Incumbent strict s | Shared strict s | Shared Fast s |
| ---: | --- | ---: | ---: | ---: |
| 24 | Singlepoint | 0.456734 | 0.446359 | 0.439306 |
| 24 | Frozen warm | 0.073474 | 0.061526 | 0.055352 |
| 96 | Singlepoint | 0.796153 | 0.842365 | 0.823680 |
| 96 | Frozen warm | 0.123625 | 0.170548 | 0.149189 |

At 96 AOs, the replacement loses more schedule efficiency than FP32 recovers:
0.829x versus the incumbent frozen endpoint despite 1.143x versus shared FP64.
At 24 AOs the apparent 1.327x includes ordinary scheduling improvement; the
precision-only ratio is 1.112x. Do not promote only the Fast schedule or use the
slower shared-strict 96-AO comparator to claim a win. The original generated
force scheduling is restored for both modes. A future candidate should type
the efficient generated consumer itself and retain pre-scatter strict recovery.

Raw Nsight reports, the rejected binary and source tarball remain under
`n1:/data/jzzeng/readme-budget-fast-20261013/`; copied kernel CSVs, job logs, XML
and `hf-shared-owner-timing.json` are in the ignored matching local artifact
directory. No full-FP32 or useful general HF Fast claim follows from this work.

## References

- `2026-10-14-readme-budget-fast.md`
- `../implemented/architecture/2026-10-14-mean-field-application-budget-transport.md`
- `tests/python/test_hf_fast_cuda.py`
- `.artifacts/readme-budget-fast-20261013/qualify-hf.sh`

## Fitted residual profile before choosing an arithmetic owner

Slurm 8435 profiles the incumbent **strict HF DF** schedule at 24/96/384 AOs
(3/12/48 atoms), cc-pVDZ-JKFIT with 116/464/1856 auxiliary functions. Each case
primes its own converged density, freezes it and captures three complete E/F
replays. GPU4PySCF independently checks each energy and analytic fitted force,
including metric/auxiliary response; maximum force discrepancies are
1.67e-11/1.72e-11/1.51e-10 Eh/bohr. The native library is
`f993f2ba55c378b4c8102a682226ad8ed7193903d1dc674632445b3fec886bf6`.

These are intrusive Nsight plus component-event traces, **not clean endpoint
timings or a Fast result**. Across three calls, total recorded kernel times are
30.675/142.758/1893.484 ms. Generic derivative kernels contribute
21.108/10.841/129.796 ms; shell derivative kernels contribute
0/86.705/594.631 ms. The zero in the smallest case is a different incumbent
schedule, not absent three-center derivatives. BLAS kernels across the whole
endpoint contribute 3.663/18.162/1013.310 ms. At 384 AOs they include expensive
SCF/final-state work as well as response; do not attribute the total to response.

The nested component ledger isolates each complete force response:

| AOs | Complete DF response ms/call | Three-center derivative ms/call | Metric derivative ms/call | Occupied projection/weight/metric/pseudo-density products ms/call |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 8.298 | 6.628 | 0.416 | 0.079 / 0.063 / 0.011 / 0.803 |
| 96 | 36.600 | 29.689 | 3.617 | 0.431 / 0.392 / 0.168 / 0.430 |
| 384 | 402.504 | 200.824 | 43.268 | 40.146 / 52.672 / 25.844 / 33.103 |

At 96 AOs the response reports real final occupied-projection reuse, a bounded
64-auxiliary panel, 8 panels, 4,276,224 three-center weights and 215,296 metric
weights. The larger route retains its packet schedule. Typing only GEMMs will
not address the small-case dominant derivative cost; routing every shell to a
generic float consumer risks repeating the direct-HF rejected-schedule mistake.
Type the existing scalar graph and efficient shell consumer instead, retaining
strict metric/conditioning, finite pre-scatter recovery and actual precision
work. Larger BLAS/SCF narrowing is a distinct, potentially valuable domain.
Do not implement a second derivative formula or silently round the metric
inverse merely to make every buffer float.

Raw source scripts, per-call JSON/JSONL and kernel CSVs are retained locally in
`.artifacts/readme-budget-fast-20261013/hf-df-profile-*-8435*`; full Nsight files
remain on n1. HF DF public Fast still has no admitted mixed operator.

The optional typed polynomial lowering and independent host checks are in
`2026-10-14-df-typed-polynomial-candidate.md`. Its diffuse d/f metric rejection
requires conditioning/fallback work before native admission. No HF-DF speed
or full-gradient qualification follows from these scalar tests.
