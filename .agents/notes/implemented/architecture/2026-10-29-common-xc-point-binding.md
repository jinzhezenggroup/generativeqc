# Decision: prepare all CUDA XC providers through one execution binding

Status: implemented
Date: 2026-10-29

## Problem

Curated, composed and automatic providers already shared physical feature
traversal, but ordinary launch, batch launch and automatic identity were resolved
separately. Native batching also used automatic-registry membership as its gate.
Moving curated methods into that route would therefore disable their qualified
batching before any change to the actual point kernel.

## Decision

Resolve a CudaXcPointBinding once during preparation. It owns the selected
ordinary callback, the qualified batch callback and optional borrowed automatic
scientific descriptor. The compiler emits both callbacks from the same provider
census. Delete the separate late batch resolver and consume the retained binding
in both execution schedules. A null batch callback means bounded tile execution.

Keep all existing kernels, thread choices, equal-spin selection, component
scales, precision paths and scientific domain identities. Automatic raw batch
entries remain available to their existing independent probes but are not
advertised as production-qualified. Response retains its unbatched contract.
No host/device allocation, feature repacking or per-point dispatch is added.

This is the execution-binding step toward provider unification. It does not
claim that curated numerical policies can already be replaced by raw Libxc
policies, or promote automatic mixed precision, graph replay or hybrid methods.

## Rejected alternatives

- Reclassifying curated functional codes as automatic would inherit narrower
  preparation and execution gates, and could substitute different low-density
  and empty-spin rules.
- Enabling every descriptor's raw batch callback would confuse executable
  availability with complete endpoint/resource qualification.
- A second per-point functional switch would add device work to a host-only
  preparation concern.

## Validation

Compiled host probes execute the emitted selector and actual LDA/PBE point
arithmetic against independent fixtures. They cover ordinary/batch pairing,
policy changes after binding, restricted/unrestricted and response execution,
split hybrids, automatic descriptor retention, missing entries and unqualified
automatic batching. A one-off comparison confirms the complete production
point program is byte-identical; no permanent source fingerprint is introduced.

GPU qualification and matched complete-endpoint measurements are recorded with
the source and library identities in the PR and ignored local qualification
artifacts. This refactor makes no compile-time or library-size speedup claim.

### Qualification evidence

Production source `51a8499b1` was compared with `ef759b52c` on a Slurm-assigned
PRO6000, CUDA 12.9.1, SM120. Subsequent provider commits only change tests and this
note. All 13,861 linked CUDA instruction sections have identical names, sizes and
SHA256 values. The 580,793,144-byte baseline and 580,792,832-byte candidate are
effectively the same size; fresh versus incremental build times are incomparable.

The independent device gates passed 100 automatic XC/KS/force cases and 28
curated complete-force cases. Nine native CUDA CTest cases passed; the broad KS
test retains a pre-existing WB97M-V force-rejection assertion that also fails
against the baseline. The baseline SONAME resolution was explicitly verified;
an earlier incorrectly linked attempt is retained as invalid evidence.

The first endpoint campaign retained all 960 energy/force calls: 12 cohorts,
eight balanced paired processes, cold/three warm/moved phases, H4/STO-3G,
24/8/16 grid, 37-point tiles and FP64. Energy differences were zero and maximum
force difference was 1.33e-15; SCF iterations and Fock counts matched in every
pair. Four phase medians failed the predeclared 1.02 timing gate: curated LDA
cold/moved (1.05939/1.04276) and automatic LDA RKS warm2/warm3 (1.02063/1.02313).
Device instruction equality alone therefore did not establish endpoint timing.

Fixed-CPU diagnostics retained another 320 calls for both LDA paths, with all
original gates passing. Automatic LDA warm ratios were 1.0121--1.0135. An 80-call
control using the same baseline library in both arms also gave warm ratios of
1.0116--1.0164, so this residual cannot be attributed to the binding change.
Do not respond to it by speculatively rearranging plan fields or changing point
mathematics. Loaded-library paths/hashes and profile selection are recorded in
the follow-up receipts. Original failed populations remain separate from any
fixed-CPU replication; acceptance thresholds are unchanged.

The complete fixed-CPU replication then passed all 960 calls and all 60 phase
gates. Paired median time ratios ranged from 0.99625 to 1.01352, within the
unchanged 1.02 limit; energy/force and equal-work gates also passed. This
qualifies the measured FP64 endpoints, not every Libxc functional, numerical
domain or production grid. The complete receipts and failed attempts are
retained in `qualification-evidence.tar.gz` under the ignored qualification
directory; PR #2860 records the measured source/library identities.
