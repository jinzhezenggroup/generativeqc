# Decision: explicit native SCF sampled-AO discovery

Status: implemented, experimental; composition 24/48 measured, default unchanged
Date: 2026-10-03

## Problem

The resident force caller can consume geometry-bound maps, but the native SCF
owner still supplies every AO to density and potential assembly on every tile.
The explicit compact-panel API alone does not change a complete SCF endpoint.

## Decision

An unused physical FP64 `CudaXcPlan` may explicitly discover sampled-jet maps
using its own immutable device basis/grid, the compiler's ordinary AO collocator,
and the same all-jet flag reducer used by resident discovery. Discovery performs
no density contractions or CPU AO work. It borrows the charged dense AO/work
panels and transfers only flags/errors; sorted host maps are copied once into
an appended, caller-owned device arena. A plan cannot change maps after any
physical or replay body has started, including unpublished replay work.

The SCF experiment is opt-in with `GENERATIVEQC_CUDA_KS_ACTIVE_AO=1`, admitted
only for device-fused FP64 WB97M-V, with an explicit sampled-jet cutoff of
1e-16. Unset or 0 retains dense SCF. Invalid switch values are rejected.
This is an experimental qualification control, not a promoted product default.
The native owner is rebuilt on changed basis/geometry/grid; density updates
reuse its immutable map. The force caller has a separate order-2 map and must
still pass complete energy/force gates when composed with this order-1 SCF map.

## Resources and fallback

Admission reserves one full global-AO map per tile, plus host indices, offsets,
and one tile's flags. The host setup peak is bounded by 64 MiB and conservatively
retained in the enclosing owner's numeric-capacity report. This fixed experiment
cap is not admission against the remaining public host budget. The current
Python KS resource planner rejects WB97M-V before preparation, so constrained
public host-budget support remains a promotion prerequisite. Mean selected AO
counts never reduce admission. Device allocations use the ordinary resource
allocator and any active device ledger.
Mandatory KS, VV10 and eigensolver allocations precede optional maps. A typed
device/budget OOM for the enlarged XC arena retries the original dense arena;
host registry OOM and unrelated errors propagate. Fixed host-cap misses select
the dense route without discovery. No external library or reference engine enters
production preparation.

The optional additive C diagnostic reports the current result's actual XC
submission count and its owner's immutable preparation work. Preparation time
is lifetime setup time, not a cost to add again on every warm sample. Counters
for point/AO and point/AO-square work describe one complete XC traversal.

## Qualification

Native tests derive independent expected masks from CPU AO values, then use
full-global-AO bilinears to validate selected rho/gradient, XC energy/electrons,
and the complete potential. Cartesian and spherical f cases span >32 AOs and
129/257-point tiles, with a small cutoff, partial maps, and all-empty maps.
Resource tests cover exact bounds, host/device misses, arena canaries and
post-evaluation refusal. The integration composition at `c06859af9` passed the
native discovery target, memcheck and synccheck (zero reported errors), and the
WB97M-V native target in n1 Slurm job 5571. That executable qualification does
not establish this standalone branch's complete SCF/force caller or a speedup.
The later standalone qualification is recorded below.

The complete integration composition `0132d7584` subsequently passed seven
independent WB97M-V energy/force, changed-geometry and stale-snapshot cases with
both native SCF and force maps enabled, and the same seven with force-cache
allowance zero (n1 RTX 5090, finite Slurm 5575; 182.27/182.45 s, no skips).
Each group reported 66 successful native calls and 467 actual XC submissions;
every successful native call selected its geometry-owned SCF map. The zero
allowance group asserts no force discovery and dense force contraction work.
Native discovery, memcheck, synccheck and the native WB97M target also passed.
Host resource/compiler checks with the completed library passed 180 cases
(seven device cases skipped in that separate host-only command).

The loaded library SHA-256 is
`b82e468613c5c90e076aa10082b2389c8dda74335f625a7b77641c1c085ef3f9`;
all 1351 manifest inputs independently reproduce source identity
`de8a1684f0afbb4ec8545022c46d24c1e6c85c3c7851df237af436897bf1d862`.
The actual master `d442177a6` integration changes no build-manifest inputs from
the built source `c06859af9`. This is composition evidence, not an exact-head
standalone library qualification. Completed 24/48 timings are recorded below;
the numerical tests do not establish a speedup or public host-budget support.

SCF discovery runs in prepared-owner construction. A complete cold comparison
must include synchronized preparation plus the first energy/force execution
for both engines. The historical comparator's execute-only `native_cold` field
alone excludes this preparation. Preserve that field and report an explicit
complete-cold total; lifetime discovery time must not be added again to warms.

## Completed composition endpoints

The unchanged integration source/library identified above has complete same-GPU
dense versus joint SCF/force-map results on n1 RTX 5090. These use spherical
def2-SVP and 48 x 16 x 32 unpruned points per atom, the matched full semilocal
and VV10 grids, three frozen-post-cold warm repeats and synchronized preparation.
Native and reference use their own converged density snapshots; neither receives
the other's density. Every priming/warm sample takes one SCF iteration.

| Atoms / AO | Native dense warm median | Native joint warm median | GPU4PySCF paired with joint |
| --- | ---: | ---: | ---: |
| 24 / 192 | 27.110692 s | 26.330918 s | 27.397887 s |
| 48 / 384 | 106.139498 s | 97.112717 s | 103.042334 s |

Joint/reference ratios are 0.961057 and 0.942455 respectively. All five E/F
pairs per variant pass the unchanged 1e-8 Eh / 1e-7 Eh/Bohr gates. The 48-atom
joint maximum errors are 1.001e-11 Eh and 6.307e-10 Eh/Bohr; actual XC backend
flags are all on-GPU. Native joint warm samples are 97.129844, 97.112717 and
97.014143 s; paired reference samples are 103.055296, 103.042334 and 103.035995 s.
Do not attribute the joint improvement solely to either the SCF or force map.

Complete cold includes preparation. At 48 atoms it is 1113.882289 s for dense,
962.211204 s for joint and 474.695895 s for the joint-paired reference: cold is
still slower than reference. Both native variants submit 21 cold XC evaluations
and one per warm. SCF discovery costs 0.649430 s once in preparation, with
14194184 numeric peak host bytes, 4608 tiles, 436 empty tiles and active-AO sum
621388. Actual SCF and force point/AO-square work fractions are 0.145202 and
0.187396. Reduced work alone was not used as a performance claim.

Slurm jobs are 5577 (24) and 5576 (48). Ignored integration artifacts under
`.artifacts/scf-active-ao/` retain raw samples, environment/source/library
receipts and `scripts/verify-matched.py`, which independently recomputes every
pair's numerical errors, verifies actual map selection, constant preparation
work, complete-cold addition and reference XC backend. The summaries are
`results/matched24-verified.json` and `results/matched48-verified.json`.
The 96-atom control and candidate remain pending. These composition endpoints
do not qualify this standalone PR head, promote default screening, or resolve
the public host-budget boundary above.

## Standalone qualification after both master claim barriers

The standalone source `c625f46b1` includes actual master `1a4acc519`, including
both bounded-force and generated-task claim-reader barriers. Its 1350 manifest
inputs reproduce source identity
`904db4bce8745a94d3e0c81ef9b8a9909e5534f7816e7e9429e9dc33dff3e5c0`;
the complete library SHA-256 is
`15de3f8a4ac9683ba90d0bf93021ef1a41e8b33853f8843a6d5a548ec04bc4bb`.
Verified ccache launchers remain enabled; this build's cache and compiler receipts
are retained separately from the earlier composition.

Finite n1 RTX 5090 Slurm job 5602 passes the native discovery and WB97M-V targets.
With the standalone branch's ordinary force caller, all seven independent
complete molecular/displaced-energy/stale-snapshot tests pass for default dense
SCF (203.43 s) and explicit SCF maps (181.96 s), without skipped cases. Each
mode records 66 successful native calls and 467 XC submissions; actual SCF
selection is checked for every successful call. These suite times are validation
receipts, not controlled performance comparisons. No force-map caller from the
broader integration is injected into this standalone qualification.

The native test executable is byte-identical to the sanitizer-qualified
`5956bd132175d701358722b8b7a204431a94962c84aec0011917af6a7f948c0e`;
its component memcheck/synccheck evidence remains applicable. The new complete
library receives the independent endpoint runs above. Earlier standalone Slurm
5599 also passed but predates the generated-task barrier and is retained as a
separate identity. Ignored `.artifacts/discovery-api-qualification/post-1783/`
contains the source archive, remote identity verification, all output/work logs,
compiler/cache receipts and independently checked `verified.json`.

This closes the standalone library qualification gap. It does not promote an
automatic AO cutoff, qualify public constrained-host-budget admission, or
replace the parent branch's documented pre-existing B3LYP full-regression failure.

## Revisit when

Promote only after independent full cold/warm/moved-geometry energy/force gates,
constrained-resource fallback checks, actual work reporting, and matched large
endpoint timing establish a useful workload domain. The preliminary force-only
24-atom experiment did not establish an advantage over GPU4PySCF.

## 2026-10-04 master integration and diagnostic invalidation

The current integration includes actual master `db6d7672` after #1774 merged.
The earlier B3LYP failure above is historical: #1794 subsequently corrected the
VWN spin-fraction rounding, with the original full native DFT regression passing
in its recorded Slurm 5625 composition. The adapter, provenance manifest and
boundary tests are preserved unchanged here. See the
[VWN correction record](../numerics/2026-10-04-vwn-spin-fractions.md).
This does not claim a new device run of the present restack.

The restack retains the original discovery implementation and all local-map
tests, plus master's newer moved-geometry/epoch regression. Independent review
found one host diagnostic-lifetime defect: batch execution could revoke its
method-owned result and then reject a null output before clearing cached KS
diagnostics. The new AO getter could therefore expose the previous solve's work
record. Cached host diagnostics now clear before method invalidation or argument
rejection can return. The production getter and admission prefix are executed
in a device-free C++ regression for null output, throwing invalidation and normal
preflight; the original prefix reproduces the stale-record failure.

Focused validation passes 322 host/compiler/oracle cases. Ninety-three real-device
cases and one NVCC resource probe are explicitly skipped; eleven native-library
resource cases are not part of this bounded local run. No AO formula, cutoff,
capacity policy, scientific tolerance or default is changed by the repair.
The fixed 64 MiB experiment cap remains distinct from public remaining-budget
admission, and unset/0 still selects dense execution. Historical qualification
and timing identities above remain separate from current integration checks.
