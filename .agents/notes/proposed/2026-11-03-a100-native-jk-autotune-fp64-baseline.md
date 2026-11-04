# Decision: use A100 as a high-throughput FP64 tuning reference

Status: proposed; bounded search exhausted its reference budget before timing
Date: 2026-11-03

## Problem

Follow-up: [GPU oracle and disposable tuning discovery](../implemented/compatibility/2026-11-03-gpu-oracle-and-disposable-tuning-discovery.md)
records the implemented correction to the reference/discovery workflow. The
failed experiment below remains historical evidence, not a completed search.

The remaining fully Direct performance gap is not explained by a small portable
force-queue improvement. Repeating broad qualification for speculative tuning
is costly. Master already provides an evidence-gated native J/K policy search;
reuse that owner before creating another selector or device-specific kernel.
The user additionally proposes A100 as the default reference for FP64 GPUs.

## Decision and boundaries

Use the current master implementation for one bounded A100 J/K search. Keep
the exact workload, scientific settings, native image and observed work fixed.
Do not substitute the unqualified wider-force-queue branch for master, and do
not compare a candidate with historical timings from another native consumer.

A100 is an appropriate **reference baseline for high scalar-FP64 throughput**,
not an identity to hard-code into Direct dispatch. FP64 support alone does not
identify that performance class. A future automatic family selector must use
actual hardware throughput/resource facts, preserve unknown-device fallback,
and distinguish scalar FP64 from optional matrix acceleration. The existing
matrix-admission predicate is not, by itself, a Direct performance qualification.

An A100 winner may seed candidates on another high-throughput FP64 device. It
does not transfer its timing, numerical qualification, architecture-specific
binary, or resource legality. Exact-device/workload validated selections take
precedence. A shared family default must remain explicitly untuned on unmeasured
targets and retain the portable incumbent until the relevant admission and
endpoint gates pass. In particular, do not replace RTX 5090 policy with an A100
recipe or silently install process-global environment controls.

The existing `--component jk` contract returns a device/source/workload-bound
`qualified_recipe`, not a global profile. Respect that boundary. A single
workload winner cannot establish an all-method or all-device default, and a
failed/no-winner search must not publish one. No new production default is
claimed merely by recording this reference-policy proposal.

## Frozen experiment

- Master revision: `8c6c6208bb1d68c96e62ed2d15ee561fcc784081`.
- Native source identity:
  `0c58960b68c71342f95bb4fe6f6a701e945957c0ac6c3eb0362bf4b71d65c25b`.
- Independent normal Release SM80 build, fast compile OFF, production tests OFF,
  stationary AOT `auto`; reuse the verified compiler cache and freeze the package.
- Stock water-96 geometry, PBE0-RKS, def2-SVP, spherical AOs, Treutler level 3,
  one system, charge zero and singlet. XYZ input uses Bohr explicitly.
- At most six native J/K policies, five interleaved endpoint pairs per candidate,
  1,800-second search budget, fresh incumbent holdout for any selected winner.
- The framework times complete frozen-density warm energy/analytic-force replay,
  excluding preparation, cold convergence and warmup. Its timings are not cold
  throughput or isolated force-kernel measurements.
- Preserve the existing independent PySCF/libxc reference, numerical, translation,
  convergence, measured-memory, work and statistical endpoint gates. No relaxed
  tolerance, proxy timing, incomplete census or manufactured measurement is used.
- CPU LSF build: 225184. Initial GPU bootstrap: 225185; corrected search: 225188,
  gpu22, one scheduler-assigned A100, finite 40-minute allocation. Preserve
  `CUDA_VISIBLE_DEVICES`. No real-GPU work runs on maintenance node3.

## Completed result and promotion decision

The master SM80 build succeeds. Its main library SHA256 is
`a40890d6ff56e85ff6bd93f11a330f76d88d64ea9702c6afc46bbba955a3deff`.
The retained build receipt verifies the expected native source identity and
normal Release settings; the existing compiler cache is reused, with hits and
misses recorded rather than claiming an entirely cached build.

The initial bootstrap fails before measurements because exporting the private
Python libc search path globally makes the system `date` executable load an
incompatible loader/libc pair (`_dl_audit_symbind_alt` / `GLIBC_PRIVATE`). The
corrected runner scopes that path to the Python process, leaving system tools,
the scheduler's device visibility and the native package untouched. This is a
launcher correction, not an algorithm change or a passing GPU qualification.

Corrected search 225188 starts at 2026-11-03 21:32:30 +08:00 and terminates with
exit status 1. The independent CPU PySCF/libxc reference subprocess consumes the
1,800-second budget and times out before producing `reference.json`. The saved
report has **zero candidate rows**, no native baseline endpoint, `installed=null`
and `qualified_recipe=null`. Consequently there is no timing population, no
accepted speedup, and no new A100 or cross-device default. A timeout is not a
measurement of A100 GPU speed and does not show that the candidate policies are
slower or ineffective.

The offline audit verifies the recorded device/source/workload binding and the
absence of an installed or qualified recipe. Host recipe/isolation/refusal tests
pass 19 cases, excluding the three explicit real-GPU cases; these synthetic
contracts are not substituted for missing endpoint evidence. No additional
search is submitted after the finite-budget result. No RTX 5090 profile or
shared runtime J/K default is changed by this experiment.

The A100 reference-baseline proposal therefore remains **proposed**, not an
implemented automatic FP64-family selector. Before another GPU search, address
the reference-stage cost: prepare or safely reuse a fully model-matched,
independently produced reference outside the GPU allocation, with explicit
provenance and invalidation. Do not monkey-patch the oracle, relax its gates,
silently extend the budget, or promote the preceding source's numerical data
as current-source performance evidence. Any new reference-reuse API needs its
own executable integrity and mismatch-rejection tests.

### Retained receipts

- `jk-build-225184.tar.gz`:
  `639a730fa73a37f8358765cf90feb2599d1f4f25d9661f794ec642919d2aa133`.
- Initial bootstrap inspection, `jk-inspect-225187.tar.gz`:
  `f9e8e27b7d1d6f50f74684c5072d1c8f29f5805b2394d0666e91a1c8a87d4d26`.
- Read-only CPU progress snapshot, `jk-progress-225193.tar.gz`:
  `9da0de5a1622ced13ee35c0a36a881aee7625331172b7ad67a82915f6cdcf18d`.
- Terminal search, `jk-search-225188.tar.gz`:
  `719b22f319804cfc42b2b9f0a58dfa5d39b31c9c4f716906e36087f283c791be`.
- Corrected runner SHA256:
  `05d0534b8938ddeedf89ed01fe5862db2e706374e1a7c373902fdc9ae315e6b7`.

## Rejected alternatives

- An `A100` string or SM80 dispatch guard would repeat the original portability
  problem and would not describe a general FP64 performance class.
- Making A100 measurements the universal default would transfer qualification
  across hardware and workloads, and could regress an already tuned 5090.
- Shipping a default from an isolated J/K or force timer ignores complete
  endpoint behavior; source-mismatched timings are likewise insufficient.
- Extending the force scheduling search before testing the existing J/K policy
  owner would expand scope without evidence. Its candidates do not currently
  search force queue width, CTA geometry or generated recurrence schedules.

## Evidence and reproduction

The task-owned inputs, frozen master source, commands and raw receipts live in
ignored `.artifacts/direct-jk-autotune-20261103/` in the existing
`/data/jzzeng/direct-autotune-20261102` worktree. Run the native frontend inside
the scheduler allocation with the matching Release library:

```bash
python -m generativeqc.tune_cli water96-bohr.xyz --units bohr \
  --method pbe0-rks --basis def2-svp --representation spherical \
  --grid-level 3 --component jk --repeats 5 --max-candidates 6 \
  --budget-seconds 1800 --source-dir /path/to/matching/master
```

See [the current tuning contract](../../../docs/user/local_autotuning.md#tune-native-dft-jk-policies)
and [the preceding queue-capacity experiment](2026-11-03-bounded-direct-force-queue-capacity-pages.md).
