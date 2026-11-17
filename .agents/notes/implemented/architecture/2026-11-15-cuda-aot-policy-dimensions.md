# Decision: separate CUDA AOT build intent and policy dimensions

Status: implemented
Date: 2026-11-15

## Problem

Issue #3086 P1b identifies profile classification as a proxy for several
unrelated decisions. In particular, portable classification suppresses exact
derivative packages and Rys-task default preferences despite distinct
mathematical capability, resource, qualification and packaging contracts.
Global profile selection also obscures mixed-target intent and binary images.

## Decision

Keep the existing production generator and runtime dispatcher. Resolve explicit
`compile_derivative_shells` and `prefer_rys_task_fock` booleans separately from
profile match/tuning metadata. Missing policy is false rather than inferring
admission from a tuned name. Migrate the repository manifest without changing
its effective defaults; custom manifests must request these policies explicitly.
`optional_fock_variants` continues to control optional module packaging only.

Preserve legacy build flags and add explicit `sm_XX=profile` overrides for
multiarch builds. Reject duplicate normalized targets instead of last-wins
behavior, in both primary and derivative CLI paths. Per-target intent overrides
the global default. Record actual supplied NVCC image inputs independently.

Write a versioned deterministic build-policy sidecar with per-target requested
and resolved intent, compiled per-consumer/route masks, preference masks and
packaging policy. This record is diagnostic provenance, not an additional
mathematical artifact or numerical qualification. Its paths, dates and host
state do not enter generated arithmetic. Preserve returned bundle compiler-input
paths and declare the sidecar as a CMake generated output.

Extend the existing device probe with optional route availability and default
preference masks. Preserve runtime consumer ownership: diagnostics do not select
algorithms, relax precision/resource checks or replace bounded fallback.
Selection diagnostics remain outside user-local hardware compatibility identity.

## Rejected alternatives

- Rename portable/tuned metadata but keep capability inference: conceals the
  same coupling and still cannot represent explicit research configurations.
- Copy tuned schedules to other GPUs or enable optional portable modules by
  default: changes performance/science policy without qualification.
- Rewrite production codegen or add a parallel runtime selection API: unnecessary
  for separating intent, available kernels and preference.
- Silently accept colliding per-target overrides: prevents reproducible intent.

## Validation boundaries

Executable selection and C ABI probes remain host-stub contract tests. Existing
retained source/device-bound numerical evidence stays historical. Byte-preserved
generated CUDA/registry/derivative artifacts provide a one-off migration receipt,
not a permanent required-CI source fingerprint. P1c qualification and P1d global
promotion gates remain distinct from this structural repair.

Validation on the implementation checkout passed 429 tests (one existing skip,
one full-native-library identity test deselected). The real C API host fixture
also executes nonzero optional-route inventories and multi-device restoration.
The stock SM80/SM90/SM120 auto bundles and forced-portable SM120 bundle, including
exact derivative headers, were byte-identical to parent
`8d2ad48175b79f2a3cb5749d3552413cdf5b7d4e` in a one-off comparison.
Compiler structure, public native API contracts/boundaries and default-promotion
inventory checks passed. Replace Rys preference Python-spelling ratchets with
manifest-policy audits and existing executable preference/resource tests; retain
the measured-domain policy ratchet without requiring a particular function body.

## Deliberate resource-ledger metadata requalification

The added explicit manifest dimensions change the full manifest SHA even though
generated kernels remain byte-identical. CI run 43694879552, job 132753210613,
actually rebuilt the qualified SM120 release shard with CUDA 12.9. Its original
report fails solely on that manifest identity. All other generated/preprocessed
source, header, ABI, compiler/image and release-flag identities match the prior
ledger exactly, as do every REG/STACK/SHARED/LOCAL field of all 102 kernels.

Retain the original failed report, original/new ledgers, raw CUOBJDump/compiler
receipts and run identity in
`benchmarks/results/cuda-aot-policy-20261115/resource-manifest-requalification.json.xz`.
The existing strict checker accepts the measured ledger after updating only the
manifest SHA. Do not remove the manifest integrity check, reset resources without
measurement, or claim numerical/default qualification from this metadata update.

## Global-disable integration follow-up

Separating portable metadata from packaging exposed a caller that had used
`--profile portable` as the global derivative-AOT off switch. That is insufficient
for a valid custom portable manifest with `compile_derivative_shells=true`.
The repair at `038aedeccf3964b58acec499cc9684f423a7a93f` carries an explicit
disabled-packages request from CMake to the generator, preserving the global
build contract without recoupling classification and capability.

Finite n1 job 17476 exercises actual CMake ON/OFF generation for SM80/SM90/SM120
under that custom opt-in and independently checks the resulting inventories.
It does not execute or qualify those derivative packages. New builds and
endpoint receipts retain the new source identity; the preceding measurements
remain historical. See the retained CUDA AOT evidence README for the receipts
and unchanged failed stock-force gates.

## Revisit when

New compiled routes need independently modeled packaging/admission, or a proposed
global default supplies complete source/device-matched cold/warm/moved numerical,
resource and package/first-use qualification. Local warm tuning alone is not that
global promotion contract.
