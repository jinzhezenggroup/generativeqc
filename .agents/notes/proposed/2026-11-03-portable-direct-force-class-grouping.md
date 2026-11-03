# Decision: group bounded Direct force packets independently of GPU identity

Status: proposed, implementation under qualification
Date: 2026-11-03

## Problem

The initial class-grouping implementation in PR #2952 applies a compile-time
`__CUDA_ARCH__ == 800` condition. Its A100 endpoint benefit is measured, but its
5090 non-regression is obtained by leaving the 5090 implementation unchanged.
That demonstrates a qualified architecture-specific experiment, not a reason
that class grouping is intrinsically specific to that architecture. The user
correctly rejects using a device architecture as the scheduling policy.

Historical evidence for that version remains in the
[SM80-only decision](../implemented/performance/2026-11-03-sm80-direct-force-class-grouping.md).
Those timing and byte-preservation results are not relabelled as portable
qualification.

## Decision

Use the same class-grouped packet algorithm on every CUDA architecture. The
selection depends only on the existing spatial low-order owner and already
admitted task classes. There is no GPU name, compute-capability whitelist,
device-generation threshold, or new user tuning knob.

A single candidate scan screens and admits each quartet once. The 21-bin
histogram partitions the existing bounded packet into disjoint class spans.
It adds 84 logical shared bytes per CTA and two CTA fences, but does not build
an additional global quartet inventory or repeat the candidate scan. At most
128 descriptors enter the existing 256-descriptor queue in a packet. The
grouping reduces within-warp recurrence-class divergence, which is a SIMT
execution concern rather than a GPU model identity.

The scalar/warp consumer, generated-class ownership, order-six split, screening,
FP64 recurrences, source coefficients, AO/primitive work, grids, SCF controls,
resource fallbacks, and source-channel semantics remain unchanged. Task and
floating-point accumulation order may differ; observable accuracy gates still
apply.

## Qualification contract

- Baseline: merged master `6affc9f92b6781ffc2072d5fc8696d5ae54e3747`, native
  source identity `d66089ee1c670a134d182b660cf11c7cab1e5ba7412a2cf48fdf97affbc31bd9`.
- Portable candidate native source identity:
  `2c766f01f2b8a2d737628b5944c823f9b48e7f3e4505184924145f75ebccc127`.
- Fresh normal Release builds use fast compile OFF, tests OFF and stationary
  AOT `auto`, retaining verified compiler caches. Source, package and loaded
  ELF identities are checked independently in each native endpoint owner.
- The predeclared six-method protocol has three fresh cold owners and nine
  frozen-own-cold-density warm calls per arm/method: 144 clean complete
  energy/force calls on each GPU, alternating arm order between rounds.
- Every cold/warm median ratio must be at most 1.05; the two primary A100 warm
  ratios must be at most 0.98. Matched-phase and independent default-grid
  numerical tolerances are unchanged. Diagnostic and sanitizer calls are
  excluded from timing populations.
- Independent Libcint source-channel tests and complete force-work censuses run
  on both GPUs. The 5090 now executes changed device code and needs fresh
  memory/race checks; its previous binary-identity argument no longer applies.
- Existing independent GPU4PySCF reference receipts keep their original source
  identities and are reused only as independent observable references, not as
  newly measured cross-engine performance comparisons.

Raw inputs, commands and receipts belong to the ignored local directory
`.artifacts/direct-portable-20261103/`. LSF build **225094** and full A100
qualification **225097** own the ECNU work. The local 5090 experiment uses a
fresh finite Slurm allocation on `node1`, retaining assigned device visibility.
PR #2952 stays a draft while the revised candidate is being qualified.

## Rejected alternatives

- Replacing the SM80 condition with another model/capability boundary would
  preserve the same unsupported hardware assumption under a different spelling.
- Repeating admission in angular-specific kernels would change semantic work.
- Adding per-force timing-based policy selection would introduce extra kernels,
  synchronization and cold-endpoint cost without evidence that it is necessary.
  First qualify the simpler portable algorithm on both target GPUs.
- Claiming that the old SM120 image is still unchanged would be false: removing
  the architecture guard changes its actual kernel, so it must be measured.

## Acceptance status

The host suite passes **370 tests**. Removing the device guard exposes the
grouped admission state in the existing compiled host harness; that harness now
supplies the class histogram and admitted task, and verifies publication totals
through the production prefix-offset helper. It does not fake a CUDA
architecture or force the obsolete ungrouped branch to keep the test green.
The initial missing-checkout-import collection attempt and incomplete-harness
compile failure are retained alongside the corrected complete run.

Portable numerical, work, performance and safety qualification is in progress.
No portable 5090 non-regression claim is made until the full population passes.
Uniform implementation is not an all-workload or all-architecture performance
claim; untested devices remain unqualified.
