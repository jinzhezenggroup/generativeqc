# Decision: Keep exact RHF canonical values within one reference phase

Status: implemented (opt-in; broad default promotion pending)
Date: 2026-10-10

The opt-in rollout is superseded by
[bounded automatic routing](2026-10-10-rhf-phase-values-auto.md). Historical
opt-in measurements below retain their original source/build identity.

## Problem

The ethane spherical aug-cc-pVTZ/aug-cc-pVTZ-RI cold DF-CCSD(T) E+F investigation
at `37227b5384fac74e6122f7befc1086bbbd877181` found thirteen exact RHF Fock
actions spending 92.03 seconds in the bounded shell-quartet fallback.
Graph-node tracing was essential: graph-level tracing concealed 85.52 seconds
of RHF device work. These are historical observations, not measurements of the
implementation base `4444d0376bb133f7981f44a263e2e409bfbfdd27`.

The response owner already demonstrated a 4,605,115,320-byte canonical value
source built in about 18 seconds. Its sixteen resident actions cost about
4.18 seconds. This suggested reducing repeated ERI construction, not changing
SCF acceptance, the Hamiltonian, precision or audit counts. Static register
counts alone did not establish occupancy/spill causality. Nuclear derivative
domain specialization is independent work and is deliberately not included.

## Decision

Reuse the common Direct provider and its compensated exact FP64 contraction.
The only new provider seam is explicit borrowing of an existing non-null
stream; ordinary creation still owns its stream. An RHF execution owns the
optional source, raw matrices, compensation and local graphs as one bounded
transaction. Source construction and finite audit precede graph capture.
Captured graphs are drained and retired before their borrowed storage dies.
No pointer-keyed, geometry-reused or process-global scientific cache is added.

Charge conservative host preparation and mandatory RHF resources before the
optional value inventory, with an 8-GiB ceiling and explicit exact fallback.
Keep transient phase admission separate from the reusable bucket's mandatory
admission: otherwise later references would pay for already retired values,
and correlation might mistake the transient source for retained storage.
Preparation scopes retain attempted-source and admission observations even
when a rejected attempt is discarded; completed work is reported separately.

The opt-in is `GENERATIVEQC_RHF_RESIDENT_VALUES=1`; default execution is unchanged.
The current size/iteration guards and memory legality are not a complete
amortization model for generated s/p/d shell schedules or one-iteration warm
references. Broad default promotion needs source-matched complete endpoint
evidence across those regimes, not just this ethane result.

## Rejected alternatives

- Retaining values through all E+F phases: the historical endpoint occupied
  roughly 31.1 GiB on a 32-GiB device; an extra 4.6 GB could reduce Lambda Q32
  and undo the RHF improvement.
- Building another ERI or RHF implementation: the common Direct owner already
  defines source identity, canonical ordering, compensation and projection.
- Reusing bucket graphs after releasing their value lease: graph pointers
  outlive the values even when geometry happens to compare equal.
- Calling fewer exact final/reference checks, substituting DF-HF or narrowing
  precision: those would change the correctness contract, not just scheduling.

## Acceptance and follow-up

Host probes cover numeric-boundary arithmetic, optional allocation/capability
refusal, CUDA/numerical failure propagation, census checks and graph-before-source
retirement. The allocated-device probe checks independent PySCF RHF energies,
cold/warm/changed/restored geometry, disabled and tight-budget fallback, and
retained bucket capacity. Complete ethane E+F must pass retained independent
energy/force-FD gates with the original Hamiltonian and response controls.
Clean interleaved endpoint samples and graph-node profiles have distinct scopes;
source/build/input hashes and raw failures must be retained.

## Source-matched qualification on 2026-10-10

The final opt-in implementation at base master
`4444d0376bb133f7981f44a263e2e409bfbfdd27` built library SHA256
`959aabeb55b953de5f4f1b6338a53acd80a29bfd40ddf5b58043145574fcb27e`.
Three RTX 5090 UUID strata ran fresh-process ABBA (Slurm jobs 7092–7094),
six complete ethane E+F samples per side. Median native time changed from
431.281191 to 352.292258 seconds, an 18.31495% reduction; RHF including the
native DF guess changed from 97.396944 to 18.343522 seconds. Work checks
retained RHF 12 iterations, CC 20/38 iterations/evaluations, Lambda 21/22
iterations/actions with Q32, Z 12/13 and two exact shell derivative passes.
The one-second total-device occupancy peaks were 31,149/31,151 MiB; these
are sampled NVML totals, not precise allocator or numeric-admission peaks.

All twelve clean results and final diagnostic endpoint passed retained
same-Hamiltonian independent PySCF energy and two-coordinate/two-step FD
gates, plus the retained qualified 24-force vector and existing response
residual gates. This is not a newly generated full ethane oracle. Three
allocated-device tests separately generated fresh independent water
aug-cc-pVTZ RHF original/displaced energies and covered lease/fallback and
geometry behavior. The focused host suite passed 61 tests with one skip.
The final checker registration explicitly bounds this owner to public
provider/assembly capabilities; its structural suite passed 157 tests.
This checker/test completion did not modify the measured production source.

Final graph-node profile (job 718 on n4, not pooled with clean n1 timing)
observed 575,639,415 source values, 13 resident physical Focks and 364
resident kernel launches totaling 2.976455 device seconds. After source
preparation there were no canonical recurrence or bounded direct Fock
kernel launches in the RHF action window. Full source preparation was
11.650857 seconds; complete RHF phase admission was 5,594,556,007 bytes.
Two-electron nuclear derivatives remained 89.508216 seconds and were not
modified. This supports the subsequent derivative-domain specialization,
not a precision reduction, fewer audits or automatic default promotion.

The complete local report, raw final profile, source patch, checksums and
per-sample gates are retained in the audit workspace as
`REPORT-RHF-PHASE-VALUES-20261010.md` and
`evidence-rhf-phase-values-20261010/`. Cancelled/superseded builds are
retained separately and not pooled. Master advanced after the measurement
freeze; these numbers do not claim qualification of a later remote tip.
