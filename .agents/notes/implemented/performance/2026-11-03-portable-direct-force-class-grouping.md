# Decision: group bounded Direct force packets independently of GPU identity

Status: implemented
Date: 2026-11-03

## Problem

The initial class-grouping implementation in PR #2952 applies a compile-time
`__CUDA_ARCH__ == 800` condition. Its A100 endpoint benefit is measured, but its
5090 non-regression is obtained by leaving the 5090 implementation unchanged.
That demonstrates a qualified architecture-specific experiment, not a reason
that class grouping is intrinsically specific to that architecture. The user
correctly rejects using a device architecture as the scheduling policy.

Historical evidence for that version remains in the
[SM80-only decision](2026-11-03-sm80-direct-force-class-grouping.md).
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
qualification **225097**, boundary/resource controls **225103**, and scoped
safety **225104** own the ECNU work. All finish successfully. Slurm **15752**
owns the local 5090 experiment on `node1`, retaining assigned device visibility:
`COMPLETED`, exit `0:0`, 50m30s under a finite one-hour limit. Accounting storage
is disabled, so the retained final scheduler receipt comes from `scontrol`, not
a fabricated `sacct` record. PR #2952 remains a draft until these receipts and
the final source identity are checked and the evidence update is submitted.

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

Both full populations pass: **144 clean complete E+F calls per GPU**, with three
cold and nine warm samples per image/method. All six methods improve in both
cold and warm median time on each GPU; this is not an all-workload or
all-architecture guarantee. Untested devices remain unqualified.

### A100 complete endpoints

| Method | Baseline cold / warm (s) | Portable cold / warm (s) | Warm reduction |
| --- | ---: | ---: | ---: |
| PBE0 | 14.879444 / 5.345750 | 14.640658 / 5.137212 | 3.90% |
| r2SCAN-3c | 19.162125 / 10.405899 | 18.735732 / 10.000248 | 3.90% |
| PBE | 7.921723 / 3.025791 | 7.864189 / 2.954184 | 2.37% |
| B3LYP | 14.978168 / 5.554661 | 14.755507 / 5.156371 | 7.17% |
| r2SCAN | 8.686865 / 3.313032 | 8.627733 / 3.245655 | 2.03% |
| M06-2x | 17.211794 / 5.567571 | 16.947227 / 5.387435 | 3.24% |

The fresh observable/work audit covers **54 complete owners, 27 matched pairs
and two moved-geometry pairs**. The 48-/129-atom, explicit 32-GiB resource,
moved/moved-repeat/restored same-phase, matched-phase and independent physical
gates pass. The known restored-versus-own-cold strict cross-trajectory
diagnostic remains distinct from those gates; the stock 192-atom baseline
nonconvergence is not reclassified as passing.

As an opt-in experiment-integrity comparison, the complete SM80 fatbin and core
host `.text` are byte-identical to the previous grouped candidate. Their SHA256
values are respectively `cad71e7a52a3fa294b51b50069ccb68ab1d00d1a28941952155eb86d06cd238b`
and `ca62bd0f866a29e128bf4db1a0c920b69c90a8bd0bb338d46ed8807153ea38d0`.
This is not a permanent source/body fingerprint CI test, and it does not replace
the fresh complete endpoint qualification above.

### RTX 5090 complete endpoints

| Method | Baseline cold / warm (s) | Portable cold / warm (s) | Warm reduction |
| --- | ---: | ---: | ---: |
| PBE0 | 19.499703 / 7.137964 | 19.273234 / 6.916340 | 3.10% |
| r2SCAN-3c | 31.688588 / 14.032246 | 31.343119 / 13.579458 | 3.23% |
| PBE | 13.766861 / 5.182291 | 13.630298 / 5.045775 | 2.63% |
| B3LYP | 19.829429 / 7.353679 | 19.622646 / 7.130794 | 3.03% |
| r2SCAN | 16.018543 / 6.243021 | 15.723539 / 6.110303 | 2.13% |
| M06-2x | 23.607902 / 7.755754 | 23.364111 / 7.538604 | 2.80% |

All **48 complete owners / 24 matched pairs** pass. The timing-only copy was
inspected after all 144 timed calls completed; every copied timed receipt is
byte-identical to its counterpart in the final complete archive. Diagnostics
subsequently confirm all six complete per-class force censuses are paired-equal.
No timing population or gate changes after seeing results.

The SM120 image now genuinely differs. Its twelve low-order entry points retain
255 linked registers/thread; tool-reported static shared storage changes from
4173 to 4257 bytes, matching the 84-byte logical histogram addition. These are
linked resource reservations, not measured occupancy or traffic counters. No
hardware-counter causal claim is made.

### Semantic work and independent checks

Per complete A100 force census, baseline equals portable candidate:

| Method | Candidate visits | Admitted tasks | Primitive-pair products |
| --- | ---: | ---: | ---: |
| PBE0 | 161862824 | 92223688 | 695763928 |
| r2SCAN-3c | 521098784 | 243123674 | 1444681762 |
| PBE | 161862824 | 91374768 | 688679280 |
| B3LYP | 161862824 | 92867712 | 700969904 |
| r2SCAN | 161862824 | 90171306 | 680137370 |
| M06-2x | 161862824 | 91819562 | 691502330 |

All available traversal, screening, per-class AO and primitive counters match
within each GPU's image pair. Aggregate executed-shell/AO counts with unknown
contributors stay null; the table does not substitute a partial sum for them.
SCF/Fock work is also equal. Fresh independent Libcint channel checks pass six
cases per image on each GPU, covering pure J, hybrid RKS/UKS and both AO
representations. The scoped pre-commit suite passes without changing the
qualified native source identity.

Both 5090 images pass **unfiltered** memcheck (zero errors) and racecheck (zero
hazards/errors/warnings), with no suppressions or kernel filters. Each of the
four probes independently passes its Libcint test and records three valid
complete force censuses with **2619 actual spatial pages**. Paired work records
are exactly equal.

Fresh A100 safety repeats the paired vendor-API control: exactly five
`cuCtxGetLimit` / `CUDA_ERROR_INVALID_VALUE` / result-1 reports in each unfiltered
image, from cuBLASLt initialization, and no other report kinds. Only after that
control, the experiment-specific XML exclusion matches that exact Api/API/error/
result. Both images then have zero unsuppressed memcheck errors and zero
racecheck hazards/errors/warnings, with equal positive work and passing oracle
tests. It excludes no memory/race finding, filters no kernel and is not generic
CI policy. The original parent/injection binaries remain unchanged; only the
interpreters of task-owned child-helper copies use the private loader. See the
historical decision for the compatibility rationale; the new unfiltered exit-99
receipts and exact exclusion are retained under this experiment's own identity.

### CI and retained limits

The advisory source-work workflow reports pre-existing stale producer/consumer
bindings in untouched `src/integrals/s_integrals.cpp` / `.hpp`. A local ratchet
reproduction reports baseline 9, candidate 9, delta/added/removed/changed all zero.
This change does not rewrite unrelated integrity acknowledgements to hide it;
the local acceptance claim is not a claim that all external CI is green.

The Libxc catalog refresh changes only current checked identities, preserving
historical measurements, qualification and `stale=True`. The failed initial
host environment/harness attempts and unavailable accounting query are retained.
No release, tag or externally hosted evidence asset is published.

## Raw receipt identities

All archives remain in the ignored experiment directory named above:

- Build `portable-build-225094.tar.gz`: `600d141cacef8d72f8253f04d13417ad9cf3cb70d8a322e12b0e46f95686aa05`.
- A100 `portable-qualify-225097.tar.gz`: `63f50d8e54a709d601a04a6d0f0bfdfde21a7cda5f2579cdf6332f7f77505612`.
- Boundary/resource `portable-controls-225103.tar.gz`: `867e940ad218b72e2bfc8f36c31e09b92f288165716a264a68579a020961e2a7`.
- A100 safety `portable-sanitizer-225104.tar.gz`: `460b3cacf625feb3f989d0c8fe9abd4d6522ecb20f0b9215c85fa977cdeadc7e`.
- 5090 `qualify-5090-15752.tar.gz`: `0cc59ef2bd7d09dcbfaa2508b64717c0aadde33a7d53c5252b37527ca882acac`.
