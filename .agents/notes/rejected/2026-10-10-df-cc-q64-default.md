# Decision: do not enlarge the automatic energy cap beyond Q32 on this evidence

Status: rejected
Date: 2026-10-10

## Hypothesis

After the Q32 automatic energy-only candidate in PR #2197, Q64 could further
amortize tile launches and repeated packing. Test explicit Q32/Q64 on the same
already qualified binary, not a rebuild or another precision policy. This is a
fresh feasibility pair, not a matched statistical campaign or a default change.

## Evidence

Slurm 2828, node2/NVIDIA RTX PRO 6000 Blackwell, scheduled visibility preserved,
finite eight-minute allocation. Two fresh complete ethane230/o9/v221/Q488 energy
endpoints, Q32 then Q64, FP64, ordinary DIIS8, original expanded physical replay
and ordinary FP64 (T), 64 GiB correlation budget. Complete process wall includes
startup/RHF/source/CCSD/admission/projection/replay/(T)/teardown.

| Metric | Q32 | Q64 |
| --- | ---: | ---: |
| Complete wall seconds | 153.92013003397733 | 154.01529652392492 |
| Complete CCSD seconds | 58.176130567 | 57.859320594 |
| Numeric capacity bytes | 8,945,140,974 | 14,769,568,750 |
| Q operations | 73,336 | 56,920 |

Complete wall increases about 0.062%; CCSD decreases about 0.545%. This single
pair neither proves a regression nor demonstrates an endpoint win. Extra
numeric capacity is 5,824,427,776 bytes (about 5.42 GiB), not complete peak
physical memory or RSS. It is not justified by the observed CCSD-only change.

All independent total energy 1e-8, (T) 1e-10 and expanded r1/r2 replay 1e-10
gates pass, with total energy bits identical. Thirty-eight pair/evaluation
admissions and zero refusals remain unchanged. Iterations, evaluations, setup,
Q slices, projection/geometry and arithmetic work are identical. Exactly 304
primary tiles/accumulations, 16,416 Q operations and 5,168 GEMM calls disappear;
logical packing/accumulation bytes decrease 59,865,781,248 / 77,217,527,296 bytes.
These are logical counters, not measured DRAM traffic. Preserve the complete
records rather than interpreting fewer operations as an endpoint speedup.

## Decision

Keep the Q32 proposal and existing explicit caps/resource fallbacks. Do not
run fresh ABBA, open a Q64 performance PR or try larger caps merely to turn this
small CCSD-only observation into a claimed endpoint win. The dominant two
ladder GEMMs account for about 92% of a representative Q32 tile's GEMM summands;
the next useful question is their implementation efficiency, not unbounded
tile or packing storage. Static work percentages are not profiler timings.

## Source and recovery boundary

Both selections use the rebuilt #2197 candidate binary from the same frozen
source as its existing qualification. Library SHA-256:
`e97b45081866ee0eae5b296e0d9185b1b14e9996e701b62d22da7cc39b028c77`;
endpoint `bd60fd5b4a5e3b1e98b0f4864b655ebd40e4e00119c979b1a59299c73979730f`.
Master observation `82c166caca181fc0df51b1ff9378b972e3b1eba8` has no relevant
candidate/CC drift. This is not latest-master endpoint recomposition.

Recipes, full records, exact work gates and provenance remain in ignored
`.artifacts/df-cc-q64-pilot-20261010/` and its node2 counterpart
`/data/jzzeng/qc-cc-q64-pilot-20261010/`. No aggregate evidence cap is raised,
accepted evidence dropped or external archive published for this rejected route.

## Revisit when

A different hardware/shape or independently measured bottleneck supports
larger tiles with an acceptable complete memory/performance tradeoff. Require
independent numerical gates, complete matched endpoint timing and semantic work
counts rather than importing this scoped pilot as promotion evidence.
