# Decision: preserve the native Hückel factor for the first DF exchange

Status: implemented
Date: 2026-10-23

## Problem

Native Hückel initialization already computes occupied molecular AO coefficients
and forms `D = 2 C C^T`. Discarding C forced the first restricted DF exchange to
contract dense D, although later iterations already used the occupied provider.
The preceding [diagnosis](../../proposed/2026-10-23-fast-df-endpoint-opportunities.md)
identified this work and independently rejected disabling Fast Direct residency.

## Decision

Move the existing coefficients through a temporary target-bound factor alongside
the admitted density. The CUDA owner checks target, dimensions, occupied rank,
finite coefficients and their scaled reconstruction against its actual normalized
D. A maximum reconstruction difference of `1e-12` is required. Original D remains
authoritative; the optimization changes only the first exchange representation.
The shared occupied provider rebuilds its own projection for the current solve.

The factor is accepted only for native cold seeds in restricted FP64 fitted
owners. Unset/1 enables `GENERATIVEQC_CUDA_KS_COLD_FACTOR`; other values disable it.
External/imported densities, different owners, unsupported precision/spin,
resource-ledger owners, unavailable bindings and malformed factors retain dense
K. The native batch drops host C after synchronous begin and on retry. Reset may
prepare a new factor; a moved warm density cannot inherit the old one. Existing
last-good warm storage and its independent provenance remain unchanged.

## Resources and failures

Retaining the already-built `nao * nocc` coefficients through strict validation
fits within the existing sixteen-matrix Hückel preflight reserve. Begin packs one
additional temporary `nao * nao` host matrix and uploads it into existing device
iteration scratch. Device retained capacity does not increase. Capacity samples
include prepared D, all live batch seeds/factors and neighboring owners. Ordinary
unobserved calls skip the added fleet capacity scans.

Public resource-ledger owners currently have no reservation for optional packing,
so they decline it. Packing allocation failure also keeps dense K. Device errors
remain errors and drain queued pageable staging before destruction; they do not
silently retry a scientific or driver failure as a successful factor admission.

## Alternatives

- Extending Direct delta updates to DF would add approximation and anchor policy
  while leaving this exact, already-produced low-rank representation unused.
- Factorizing arbitrary imported D would add eigensolver work and ownership rules.
  This change only reuses the coefficients from the native producer.
- Replacing D by reconstructed C-derived D could change initialization. Instead,
  reconstruction is an admission check and D is never replaced.
- Retaining another device matrix or old mutable U is unnecessary. Existing
  scratch and the ordinary occupied contraction suffice.

## Invariants and qualification

Independent native CPU DF checks final physical state; a diagnostic same-D dense
GPU K/full-Fock audit must remain below `2e-11`. Hückel's independent density and
simultaneous allocation probes also validate the retained coefficients. Native
fault cases cover target/shape/rank, truncated storage, perturbation, NaN/overflow,
missing/external seeds, normalization, repeated/reset/moved solves and the resource
ledger. Public Standard/Fast two-item batches verify cold/warm/reset energy/forces
and first-K routing. The PySCF fixture imports its DFT module explicitly, so it no
longer relies on another test's import order.

The initial public GPU run lacked full small-basis PBE/PBE0 force AOT artifacts;
they were built from current compiler inputs before rerunning the failed tests.
The matching existing SPD artifacts were reused only after contract verification.
No numerical gate was relaxed. Complete endpoint results and reproducible source,
reference, build and work identities belong in the accompanying benchmark receipt.

## Measured endpoints

The [qualification receipt](../../../../benchmarks/results/ks-df-cold-factor-20261023/README.md)
retains 56 native processes / 204 complete E/F endpoints. All numerical and
admission gates pass, with identical Fock counts in all 28 paired controls.
At 96 atoms, Standard cold latency falls 36.952 to 33.275 seconds (10.0%) and
Fast 36.868 to 33.381 seconds (9.5%); warm medians remain within 1.2% across
sampled sizes. The initial 12-atom Standard population was 5.18% slower, triggering
four additional full pairs for both owners. All original samples remain included;
the six-pair median satisfies the declared regression gate. Fast48's +0.22%
remains visible. These default-grid results are not a universal speedup claim.
The first cold K/Fock differences are at most 6.22e-14/7.8e-15, and the factor
reconstructs admitted D within 9e-16. No additional retained device matrix is used.

## Revisit when

Qualify a reserved resource-plan extension before admitting bounded owners.
Qualify mixed SCF or unrestricted factors independently. Revisit admission cost
for additional basis/size domains using complete cold and warm endpoints; this
change does not reduce DF source setup, metric projection or force-response work.

## Integration outcome

While this independent implementation was being qualified on `bfb1584b3`,
[PR #2692](https://github.com/jinzhezenggroup/generativeqc/pull/2692) merged a
separate native-cold algebraic density-factor recovery, resident DF validation
and prepared-force improvements. PR #2696 subsequently incorporated that master
at `9f008cff0631834d10c4520d33a9b86cfc162fa6`.

The current upstream already removes the same dense first K in its qualified
large-system domain. Our earlier comparison therefore cannot establish an
additional current-upstream gain. The tested branch and evidence are retained;
no duplicate production PR was opened. A future handoff experiment must compare
against that new default, reuse its shared admission policy and preserve its
explicit noncanonical lifetime (no warm frame or stationary projection lease).
The historical evidence above remains unchanged and must not be presented as a
new 10% improvement on top of #2692.
