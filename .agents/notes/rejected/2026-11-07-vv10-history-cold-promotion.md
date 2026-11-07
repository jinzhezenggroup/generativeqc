# Decision: reject A100/5090 cold promotion of exact frozen-VV10 history

Status: rejected for default promotion; no new production algorithm
Date: 2026-11-07

## Problem

The latest #2976 analysis recommends testing #3037 rather than rewriting DIIS.
Its fixed-operator transport is algebraically distinct from earlier rejected
orthonormal-history experiments: bank actual D/F/raw-R triples, add DeltaV to F
and DeltaV D S - S D DeltaV to R, rebuild the complete Gram, then publish.
The available observed pool is only four frozen inner Focks, not seven completed
seed steps or all expensive physical targets. Counts alone cannot justify it.

## Evidence

The existing implementation at08f8abe13 is frozen without modifying its native
code. Normal Release SM80-real+SM120-real compilation uses verified private
sccache0.16.0. The source is ef7f04bd77cae25a87aca1452c4640ac30ca95c7ff89817ae95e517fe24d9191;
ELF53b8e6ca2d11ba260d30857d28d1fcea04542756bd5e2e304a6f2ae09e4a2ee4.
The publication branch joins masterb40, including the now-merged #3036; the
frozen measured image and #3038 HF image retain their distinct identities.

Actual off/on first complete cold pairs, including analytic forces and host
export, are A100 LSF227923:82.605219→88.449537s (+7.075%), and RTX5090 Slurm16695:
165.173598→184.785585s (+11.874%). Each arm retains three own-density warm calls.
No negative pair is rerun or replaced by a faster first call.

Both cards show identical work transitions:

- Focks16→15: semilocal7→7, frozen4→3, targets5→5.
- SCF VV10 sweeps4→5; exact component cache hits1→0 over four probes.
- Two successful transports,14 restored entries, no numerical refusal.
-141558528 extra retained bytes, bounded below256MiB, charged through force.
- All warm calls refuse history transport, write no density-ring entries,
  keep one target Fock and zero SCF VV10 sweeps, and pass the1.03 warm gate.

Each card passes seven linked-native independent residual/Gram/affine/refusal
cases. RTX5090 memcheck/racecheck report zero errors/hazards. Both complete all
physical endpoints. Original independent Standard E/F gates pass32 comparisons;
worst max force error is1.0465e-6 Eh/Bohr. The candidate is closer to that
reference than the baseline, but their5.8650e-7 max /3.1622e-7 RMS force difference
fails the unchanged1e-7 /3e-8 paired gates. Both payloads intentionally exit2
at that refusal. This is neither a CUDA failure nor evidence that the exact
residual correction is mathematically wrong.

Raw original records, hashes, source/ELF/scheduler receipts, build/cache and
restoration receipts, controls, seven-case logs and sanitizer logs are retained
under `benchmarks/results/vv10-history-cold-20261107/`. The offline reader reports
missing active compact-point counts, force pair sweeps and exclusive rebase/Gram
time explicitly. No fresh reference latency ranking is made.

## Decision and causal boundary

Reject A100/5090 default promotion. Leave main defaults unchanged and report the
result to #3037 and #2976. Do not weaken force gates, relabel failed jobs as
passed, retain a new opt-in fork, or repeat the failed screen in reversed order.
The existing draft's other-device research remains separately unqualified by
these measurements; this note does not close it or claim a universal theorem.

The measured explanation is lost exact component reuse and one extra SCF VV10
sweep despite one saved inner Fock. These data do not assign the entire timing
regression exclusively to the sweep or to Gram reconstruction.

Code/history inference, not a new intrusive profile: the incumbent target tail
can confirm an already stationary density unchanged, allowing exact pair-cache
reuse. The changed trajectory crosses its final physical checks differently.
`nonlocal_ks_density_confirmation` explicitly refuses a density whose canonical
final probe failed, so blindly retaining it to recover a cache hit is unsafe.
The final-target histories are preserved for further diagnosis. Any hypothesis
about a particular probe refusal still needs a phase-labelled diagnostic.

## Rejected next moves and revisit conditions

Do not tune bank size or weaken final closure just to restore the lost hit.
Successful transport counters and lower Fock totals did not improve cold.
Do not stack a VV10 secant predictor into this history experiment: it changes
the proposed fixed operator and needs separate anchor-residual evidence and
trust/refusal rules followed by unchanged exact physical targets.

Revisit history promotion only with a new causal mechanism that preserves
scientific/lifetime gates and reduces complete cold work on A100 without5090
regression. Separately rank changed-density outer sweeps and shared geometry
pullback against measured XC/order-six force work; neither this failed pilot nor
the older AO cache/map-hoist rejection establishes a win for those branches.
