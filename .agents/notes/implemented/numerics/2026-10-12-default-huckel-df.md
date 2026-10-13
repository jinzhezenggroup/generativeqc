# Decision: default native Modified Hückel, including restricted DF

Status: implemented
Date: 2026-10-12

## Problem

The explicit native provider and CUDA RHF execution-owner integration established
independent numerical correctness and workload-specific complete-endpoint benefit.
The user subsequently explicitly requested promotion to default and a PR including
DF. This supersedes the opt-in/default decisions in the native Hückel and CUDA HF
admission notes; it does not retroactively change their measured evidence or claim
that every supported workload crosses the original strictly >5% performance gate.

## Decision

`initial_guess="auto"` prefers native Modified Hückel for FP64 all-electron H–Ar
restricted CPU/CUDA HF/KS, including DF. No atom-count profitability whitelist is
introduced. Explicit `None` restores Hcore. Unsupported spin, elements, ECP,
precision, higher derivatives and resource-planner domains remain unchanged.

The options ABI remains version 1. Published `AUTO_CPU=4` and its bit `1<<3`
retain MINAO semantics. Hückel uses kind 5, `AUTO_HUCKEL_CPU=6`, and new bits
`1<<4` (provider), `1<<5` (CUDA RHF), `1<<6` (DF), `1<<7` (per-item CPU).
Additive v2 automatic queries return only Hückel/Hcore; v1 queries still return
MINAO/Hcore for old consumers. Model and system policy remain native-owned.
Additive capability bits separately advertise
CUDA RHF and DF target integrations; a global Hückel provider bit from an older
library cannot imply those integrations. Older libraries retain the previous
MINAO default in its original advertised exact domains, otherwise Hcore.
Native CPU batches resolve and charge each supported item independently, preserving
upstream mixed-batch behavior. CUDA/legacy batches remain conservative when any
item has an unsupported element/ECP.

DF targets borrow their actual prepared overlap and eigen owner. Exact neutral
atomic RHF is explicitly native production preparation only: it never substitutes
the molecular DF operators, auxiliary basis, cutoff, backend, functional or
response. CUDA DF RHF publishes a strictly admitted density without warm
normalization. CUDA DF KS and CPU DF share their existing preliminary orchestration.
Warm/imported densities take precedence. Preparation is bounded and serialized;
allocation failures propagate. Failed seeded items receive at most one ordinary
DF Hcore retry, after releasing the failed CUDA HF arena; successful neighbors
are not rerun. Warm failure disables preliminary preparation on its core retry.

The atomic implementation alone may own the native exact atomic Fock plan, DIIS
and shared-scalar eigen adapter. The structure audit permits those dependencies
only for that implementation, not other projection or independent reference
modules. The CUDA seed adapter remains a separately audited execution-owner bridge.

## Evidence and limits

Historical exact CUDA RHF energy/force endpoints retain their full-precision
timing samples, scalar numerical gates, source reconstruction and work ledger in
`benchmarks/results/cuda-hf-huckel-20261012/`. Median wall reductions against
Hcore were 3.61%, 6.95% and 10.20% for 3/48/96 atoms. Atomic preparation is included.
The earlier exact PBE0 campaign retains its separate source identity under
`benchmarks/results/native-huckel-20261012/`; reductions against MINAO were 9.14%
and 2.31% for 48/96 atoms. Neither campaign is evidence of a DF speedup.

`tests/python/test_huckel_df.py` independently converges PySCF DF HF/PBE/PBE0 with
the exact orbital/auxiliary primitives and quadrature. Energy, analytic force,
final density and physical F(D) commutator gates do not compare only with Hcore.
It also protects batch/single execution, warm precedence, provider/domain gates,
preparation caps and failure, and bounded target retries. Existing native atomic
fixtures, CUDA HF tests and default/ABI tests remain independent qualifications.

The final source-bound CUDA qualification on finite Slurm job 7915 passes all
210 Python cases (CPU/CUDA, no failures/skips) and four native executables. Local
host checks pass 411 cases plus 22 subtests; two source-owner DF retry cases pass.
Real-device tests exposed a remaining common CUDA DF preflight guard that still
rejected every preliminary provider. It now admits only native Hückel, without
granting HF/LDA/MINAO DF support or changing the molecular DF schedule. Actual
compiler-cache receipts and final execution/JUnit counts are retained in
`benchmarks/results/default-huckel-df-20261012/validation.json.gz`.

The separate fresh-process CUDA DF RHF energy/analytic-force campaign on Slurm
job 7914 measures 18 independently accepted samples with spherical def2-SVP
orbital and auxiliary sets. Median Hcore→Hückel times for 3/48/96 atoms are
0.372555→0.379060, 2.242193→2.242727 and 14.500008→13.387533 seconds: wall
reductions of -1.75%, -0.02% and 7.67%. Target iterations fall 15→14, 21→18 and
25→18; preparation medians are 0.001652/0.110424/0.744293 seconds. All seeded
samples use seven atomic iterations/Fock builds and one target attempt. The
public DF target-Fock count is unavailable, not inferred from iterations. The
96-atom paired reductions are 7.85%, 7.58%, 7.71%; smaller cases do not clear 5%.
No DF KS performance claim is made, and no profitability whitelist is inferred.

The DF auxiliary choice is an explicit benchmark input within current CUDA
angular admission. Universal-JKFIT/Weigend oxygen g shells were correctly
rejected; production admission and requested auxiliary sets remain unchanged.
The 48/96 fixtures are the nested water structures owned by
`benchmarks.readme_hf_scaling`, not the historical exact PBE0 fixtures. Independent
GPU4PySCF gates bound energy/force/density/physical residual errors; the summary
retains every full-precision timing, gate and work sample and frozen source,
library, harness and assigned-device identities. This is evidence for the stated
DF RHF workloads only, not universal acceleration.

The current checkout retains compact scalar gates, exact timing/work samples and
production reconstruction patches. SHA-bound `recovery.json.gz` inventories bind
original full JSON records, source audits and reconstruction patches to the
preparatory commit in the branch's existing Git history, avoiding duplicate large
payloads in the 64 MiB checkout budget. Transient dense density exports and logs
remain in local ignored run storage. No external publication, Release, tag or
distribution workflow is created.

## Review correction: preserve terminal provider failures

The new bounded seed retry exposed legacy exception adapters that erased
`CUDA_ERROR` into an untyped numerical failure. Preserve native statuses from
seed admission, independent Fock/eigen providers, compact DF dispatch, final
validation, warm-state publication and bucket/Fleet adapters. Numerical rejection
still permits the documented Hcore retry; CUDA/OOM failures cannot start it.
The backend-neutral final-state selector retains an opaque original exception
alongside its failed selection so an execution owner can recover its type without
introducing a backend dependency or changing the selector's result contract.

The executable owner regression compiles the actual seed/bucket/retry functions
with injected provider failures. On the original PR source, all ten CUDA_ERROR
cases fail (five boundaries, owned/cached arenas); the corrected matrix passes.
It also covers terminal OOM, failed retries, successful-neighbor preservation and
retiring the failed arena before a per-item retry. Independent adapter and native
selector/preliminary tests protect exception identity end to end. This mechanical
status repair does not change numerical arithmetic or gates. The recorded GPU
campaign remains evidence for its frozen source; local CPU/owner regression
checks qualify this review delta rather than claiming a new GPU campaign.

## Rejected alternatives

- Simply deleting DF guards would silently leave CUDA DF HF on Hcore without
  diagnostics, or normalize a validated cold density as a warm input.
- Inferring support from the schema or provider-kind bit would break older
  libraries that implement Hückel but not the new target-domain owners.
- Atomic/minimal preparation using PySCF, or a CPU reference retry of failed CUDA
  admission, would introduce a hidden production reference dependency.
- Treating one favorable cohort as a universal speedup claim would omit negative
  cases and preparation cost. Default behavior is user-authorized and rollbackable.

## Revisit when

Revisit for stationary-root/state regressions, complete-endpoint regressions,
unbounded preparation/retries or genuinely broader scientific qualification.
Keep numerical/force gates and work accounting independent of performance claims.

## References

- `2026-10-12-native-modified-huckel.md`: native atomic construction and cost.
- `2026-10-12-cuda-seed-eigenpair-ordering.md`: paired presentation, unchanged gates.
- `2026-10-12-cuda-hf-huckel-admission.md`: execution and force-frame ownership.
- `docs/user/initial_guesses.md`: current behavior and explicit rollback.
