# Decision: Share bounded RAM ERI backing with global-hybrid KS

Status: implemented
Date: 2026-10-25

## Problem and decision

Exact PBE0/B3LYP KS phases regenerate omitted full-range ERIs even though the
same geometry-only values serve every density-current J and K contraction.
Extend the existing FP64 LR host owner to FullRange, retaining its operator
identity. Select the full radial recurrence during construction and reuse the
existing compensated J(total), K(alpha), K(beta) consumer. No kernel equation or
precision changes are needed. Separate `GENERATIVEQC_KS_HOST_*` controls preserve
the LR policy. Unset uses the same bounded available-job-RAM default for cold
strict sources; explicit zero disables it.

GPU buckets retain priority. If enabled backing has headroom, an unmasked GPU
source may fit whole buckets to its existing ceiling. Full GPU residency is
unchanged when it fits. Explicit masks remain authoritative; identical masks
must be used for controlled GPU-versus-RAM comparisons. Small/full-cache cases
must accompany capacity-limited endpoint evidence, since past PBE0 results show
that replacing a profitable complete GPU cache with partial residency regresses.

## Invariants and coverage

The phase source is retired at geometry and precision handoff; zero-order
cached values do not authorize derivative-integral reuse. Warm eligibility,
independent final physical audits and complete budget checks remain unchanged.
Signed/zero-density full J/K and UKS matrices are compared to PySCF, including
screening and 4 KiB tile tails. Complete cold/warm/moved E+F tests compare both
cache parity and independent PySCF references; they require actual host use,
full-cache priority and bounded refusal rather than numerical parity alone.

## Other candidates

DF gradient `PinnedResponsePanels` already caches gathered host panels through
two pinned slots. Duplicating that cache is not useful. RHF resident values can
be graph-captured; pageable replay requires eager staging and cannot be added
without a separate scheduler/lifetime qualification. WB97M-V physical response
has independent device/work caps and source ownership. Those are future
consumers, not implicitly enabled by this provider extension. Temporary disk
spill is a separate resource tier with its own I/O and cleanup contract.

## Complete endpoint evidence and retained admission

The [source-bound cohort](../../../../benchmarks/results/host-full-eri-20261025/README.md)
measures 96-atom PBE0 cold medians of 94.868 s GPU-only and 39.103 s GPU+RAM
when incremental Fock is disabled and the existing source is forced. Both have
nine iterations and ten Fock builds, and every cached action avoids omitted
recurrences. Independent E+F gates pass. Natural incremental cold is already
21.816 s and does not consume this source; forcing storage on a warm two-Fock
call rebuilds a quartic cache and is also slower than the natural warm path.
Keep these admission gates. RAM capacity alone is not a reason to override the
incumbent incremental/generated consumer. The benchmark runner records this
control explicitly so an unconsumed cold source cannot masquerade as a cache
performance experiment.
