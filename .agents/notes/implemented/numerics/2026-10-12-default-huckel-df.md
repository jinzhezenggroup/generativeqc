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

Retained benchmark records omit transient dense density exports and build/test
logs from Git; the original unmodified payloads remain in the task's local ignored
run storage. Compact records retain scalar gates, exact samples and reproduction
inputs. No external publication, Release, tag or distribution workflow is created.

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
