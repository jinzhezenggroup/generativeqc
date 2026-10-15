# Decision: precision-bound resident sources for PBE0 budget-Fock Fast

Status: implemented; endpoint qualification recorded separately below
Date: 2026-10-15

## Problem

The first independently qualified 96-atom/768-AO budget-Fock endpoint was
numerically acceptable but took 210.505000 s versus optimized FP64 97.183134 s
(three alternating repeats). Its twelve mixed Focks repeatedly constructed
17,717,684,496 recurrence values; strict FP64 already admitted resident sources.
Narrower arithmetic cannot repair that algorithmic work deficit. These old
measurements remain bound to their original sources and are not overwritten.

## Decision

Reuse the common canonical integral producer and orbit scatter, rather than
introducing another integral implementation. A mixed lease freezes geometry,
full-range operator, Schwarz layout and FP32 recurrence/storage precision.
Its consumers convert source values to double before the existing FP64
density products and compensated scatter. All source values must pass the
finite audit before atomic publication. Values, compensation, canonical layout,
transient setup, free-device headroom and host capacity retain bounded admission.

Only budget-Fock Fast opens this optional restricted, complete-bucket,
through-d source. Ordinary AUTO and strict sources retain their existing
selection policies. Unsupported partial masks, insufficient capacity, warm
seeds and disabled residency preserve bounded uncached mixed execution.

At strict handoff, audit and retire the mixed lease before admitting an
independent FP64 source. A numerical mixed retry retires without certifying
the failed action, then prepares strict work. Captured borrowers are invalidated
and outstanding work drains before source release. Strict construction cannot
change an existing mixed identity into FP64, and post-SCF consumers cannot
borrow a mixed source. The two quartic value buffers never overlap.

## Work accounting

Record completed FP32 source construction once, including work discarded by a
late publication refusal. Replays count real mixed-source J/K actions but zero
new recurrences. FP32 source storage is distinguished from FP64 products,
scatter, accumulation and reductions. A zero replay recurrence count is not
evidence of zero exchange work; only an authenticated mixed-source replay may
use that distinction. Uncached zero-work mixed failures still cannot fabricate
positive exchange evidence.

## Rejected alternatives

- Disabling the comparator's resident cache would handicap optimized FP64.
- Reusing float values for strict closure would change the physical target.
- Holding both quartic caches would conceal the phase-transition memory peak.
- Repeating the construction count per replay would manufacture semantic work.
- Further loosening acceptance to obtain a timing win would not fix source reuse.

## Evidence

Local source-matched evidence is retained outside the checkout in
`../evidence-pbe0-fp32-resident96-20261015/`. Host executable tests cover optional
capacity refusal, FP32 byte inventory, immutable precision, restricted/full-range
admission, repeated zero-recurrence replay, census and teardown. A Slurm-only
native probe checks the actual float producer/storage/replay against independent
PySCF J/K for signed, scaled and zero densities, including immutable-precision
refusal and a stricter replay mask. Complete energy/force qualification uses
the separate GPU4PySCF oracle and the unchanged optimized FP64 comparator.

Qualification completed on RTX 5090 / CUDA 12.9.1 through finite Slurm jobs
8494–8499. The real float-source J/K error is at most 1.05613e-7 against PySCF;
Compute Sanitizer memcheck reports zero errors. Host regressions pass 365 tests,
with 539 compiler modules and zero dependency errors. Source-matched main library
SHA256 is `41fd8383966943b06ca6c164392b4022392d859dd9ac4b2e36553472dfffe8fe`.

The 96-atom/768-AO complete energy/force endpoint now executes one solve,
12 mixed and 5 strict Focks, zero numerical retries. Each mixed phase constructs
1,476,473,708 FP32 values once (5,905,894,832 bytes), not twelve times; its
independent strict source has the same inventory in 11,811,789,664 bytes.
Resident replay has zero new recurrences. Across both measurement scopes the
maximum Fast errors are 2.31376e-9 Eh energy, 4.15953e-7 Eh/bohr maximum force
and 1.58218e-7 force RMS, within all independent observable gates. Cartesian
water and the spherical tetramer also qualify, but remain slower than FP64.

### Distinguish the two performance scopes

The original same-process alternating runner gives three-repeat medians
97.315623 s FP64 versus 67.305839 s Fast (1.445872x). Its first FP64 call is
78.625499 s and successfully builds the existing resident source. Later FP64
calls naturally refuse optional residency at the unchanged free-memory gate:
14,985,773,728 source bytes exceed about 14.82–14.83 GB of allowance after
context/library retention. No FP64 optimization or control was disabled.
Those calls have unobserved Direct recurrence counts, not zero work. Do not
present 1.45x as the speedup against an always-resident optimized comparator.

To avoid overstating the result, a second finite allocation alternates six
independent processes on the same GPU. All three FP64 sources and all six Fast
phase sources are admitted, and every replay reports zero Direct evaluations.
Complete endpoint medians are **78.153746 s FP64 versus 68.835479 s Fast**:
**1.135370x**, or **11.923% less wall time**, against that stronger resident
baseline. Raw times and all gate/admission evidence are retained in
`comparison-96-isolated.json`, `isolated/` and `progress-isolated-*.jsonl` in the
evidence directory. Timings include construction, both source phases, complete
forces, export and teardown; first-process work is not silently excluded.

This fixes the recurrence-reuse deficit, not every possible Fast overhead.
Keep the stronger baseline, the slower small cases, and both timing scopes
when making future performance claims.

## Revisit when

Broader spin, range, angular or partial-residency domains have independent
endpoint evidence and a bounded precision/source identity. Do not infer their
qualification from shared template instantiation or this shape admission.

## References

- [Original budget-Fock decision and slower endpoint](../numerics/2026-10-14-pbe0-budget-fock.md)
- `docs/user/application_precision.md`
- `benchmarks/pbe0_fast_fock.py`
