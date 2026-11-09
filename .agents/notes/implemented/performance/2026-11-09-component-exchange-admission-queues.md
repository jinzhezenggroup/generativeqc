# Decision: compact strict component-lane K admission cooperatively

Status: implemented
Date: 2026-11-09

## Problem

Complete A100 hybrid DFT cold endpoints still spend substantial time admitting
screened products into shared-Rys component-lane workers. Optimizing recurrence
arithmetic alone misses the serial candidate-domain/predicate overhead. The
required unit of improvement is fresh-calculator energy plus analytic forces,
with unchanged SCF, precision, independent numerical gates and every own-density
warm, not a standalone kernel sweep.

## Decision

The compiler selects a bounded admission queue by existing schedule properties:
strict shared-Rys component lanes with at least32 threads. There is no device
name/SM profitability whitelist, new user opt-in or SCF-method policy in the
integral emitter. Existing default dispatch remains responsible for schedule
selection;5090 is not forced into the shared-Rys profile.

Restricted pure K with both shell/system density bounds cooperatively admits
products through one complete warp. Different-pair classes claim32 flat products
and compact bra/ket indices, preserving two-dimensional concurrency. A rejected
local Schwarz tail is **not** the end of the flat domain: only a batch with no
in-domain lane can retire the worker. Same-pair classes retain sorted bra rows,
the original conservative coarse ket tail and triangular physical-pair predicate,
then compact32 ket candidates at a time.

The complete original worker remains for other/null-bound consumers; unsupported
unrestricted/mixed/compensated wrappers still refuse. Scientific recurrence,
precision recording, screening thresholds and scatter are unchanged. Static CTA
state is264 bytes for different pairs or136 bytes for rows, with no new device
allocation or resident cache. Full-warp ballots, widened cursors and explicit
reader-retirement barriers preserve partial batches, changed densities and graphs.

## Qualification

Matching normal build16924, base76bb87c76, source `da106c87` versus `b11d2832`:

- A100230562 B3LYP96 complete cold11.873728→10.340895s, saving1.532833s/12.91%.
- A100 PBE09611.762709→10.221247s, saving1.541463s/13.10%.
- Both gains clear the unchanged2%/MAD gate; every48 cold and every paired warm
  passes3%, worst A100 paired warm ratio1.022951.
- Physical509016931 passes all cold/every-warm guards, worst paired warm1.010280.
- Every M06-2X/omega-B97M-V96/48 single-pair guard passes on both cards. These
  are not ABBA speedup claims; omega-B97M-V cold remains materially unchanged.
- All clean native/reference phases pass unchanged tighter independent oracles,
  strict native E/F and initial-guess/Fock/precision/J-K work parity. Whole-endpoint
  quartet counters are unavailable and must not be inferred from fixture counts.
- Both actual normal A100 ELFs pass18 independent raw-K/count/graph definitions,
 864 repeated predicates and54 unsupported refusals/image. Physical5090 actual
  queue fatbin16921 and focused sanitizer16923 pass; those are not normal-ELF tests.
  Normal5090 dispatch proves the unavailable shared-XY path never executes there.
- Host tests compile and execute the actual emitted admission with SIMT stand-ins
  and an independent Cartesian domain. They are not CUDA or ERI scientific oracles.

The checked-in capsule preserves every normal endpoint value, build/cache/restore
receipt, dispatch observation, raw-K producer/manifest and complete gate logs.
Its independent endpoint references remain in the existing README oracle corpus;
versioned raw-K inputs and earlier retained independent arrays are not removed.
New generated binary fixtures/normal products remain ignored, not production
dependencies. The old raw-J population container is losslessly repacked with the
existing v3 codec, authenticating every original text byte, to avoid raising caps
or deleting scientific coverage. Alternative compression/filter/schema experiments
did not justify a new codec or broad unrelated evidence migrations.

## Rejected alternatives and boundaries

Reject hard-coded A100/SM80 profitability branches, a global shared-XY switch for
5090, CPU/PySCF production work, looser convergence/precision and unbudgeted caches.
Do not replace flat product claims with bra-only scheduling for different classes;
that loses concurrency. Do not stop a flat worker on an empty screened batch.

Interposer230539 established~1.5s mechanism gains but did not qualify production.
Normal509016927 failed before timing by calling an SM80-only cubin on5090.
A100230553 failed in Python codec startup before reconstruction. Both are retained,
not relabeled PASS. Task-local bootstrap correction preserves the same interpreter
and bytecode without modifying the shared environment. No clean cohort is retried.

Revisit on changed consumer contracts, schedule/block geometry, screening order,
resource constraints or independently demonstrated whole-endpoint regressions.
Do not infer H100 gains, HF/DF qualification or all non-CBS README leadership.

## CI integration after the endpoint cohort

PR3056's first hosted CI correctly rejects three integration omissions: the
module-boundary inventory lacks the new pure compiler queue owner, standalone
replay imports the repo package before locating its root/refusing optimized
Python, and live source fingerprints differ from retained Libxc/static-AOT
metadata. Add the queue to the cycle/purity checks, with only `__future__` as
an allowed dependency, and test standalone replay from an unrelated directory
without PYTHONPATH, both ordinary and optimized Python.

Fetched/merged masterc6967d7a8 contains HF PR3053, not additional DFT generation
changes. Actual failed hosted resource artifact13649452613/run42559608354 is
authenticated by ZIP SHA256
`73bacd5012427288d0309c9dc2835d8a760435cd94d3a7287a21d3df01ea3f00`.
All102 kernels' REG/STACK/SHARED/LOCAL, manifest, headers, flags and toolchain
match the old ledger exactly. Retain the original failing report and old ledger
under the endpoint family's `resources/`, and deliberately update only source
and preprocessed hashes using the unchanged official validator. The replay also
rejects injected register drift; no guard or resource limit is loosened. This
is actual static compile evidence, not a fresh physical GPU or endpoint run.

Regenerate the Libxc snapshot through its official checker. Only checked source
and native-input identities change; summary/stages/capability identities remain
identical, stale stays true, qualified products remain zero. Do not relabel old
native measurements or rerun successful endpoint cohorts for these CI repairs.

## Concurrent repair and later master synchronization

The first push is rejected because another writer already published remote
faee0a0dc's equivalent CI repair. Merge it without force-pushing or discarding
its history. Resolve the duplicate guard/import inventory as one contract and
retain the additional unrelated-cwd replay and authenticated resource tests;
the resolved tree is exactly the already-validated503fbf86f tree.

Master then advances to51d887ff9's source-bound Libxc acceptance. Preserve that
new checker, fresh historical campaign and upstream scientific demotions. Its
accepted snapshot is already stale with zero current E/F endpoints. The checker
correctly refuses a fingerprint refresh against the older qualified previous
summary. Advance `previous` to the exact accepted51d stale summary through the
official serializer, and regenerate through the official checker. The new
summary is identical; there is no new transition or acknowledgement to hide.
The former66 demotion acknowledgements and original previous summary remain
in accepted51d Git history and exact ignored local copies. `--require-current`
must still fail; this metadata repair does not manufacture fresh native E/F.

Retain the unsuccessful all-files mutating-compaction hook attempt: its
whole-batch resource ceiling refuses before mutation. Use the repository CI's
existing explicit skip of that local mutator, while retaining the separate
all-files read-only compaction and evidence checks. No cap, guard, successful
GPU population or independent numerical tolerance is changed by this repair.

## References

- `benchmarks/results/dft-component-queue-20261109/README.md` and offline replay.
- `docs/maintainer/performance_engineering.md` and `evidence_retention.md`.
- Issue2976 comments6731043233,6731236674,6731367810 and6731535816.
