# Decision: bound prepared precision provenance to the last actual geometry

Status: implemented; bounded complete endpoint timing qualified
Date: 2026-10-14

## Measured residual

HF-DF cost-selected shell arithmetic passes independent endpoint gates, but
Slurm 3007 measures only near-neutral complete README timing at 96 AOs.
The public Fast batch ledger separates about 47.0 ms native candidate execution
from 49.2 ms complete frozen execution. Roughly 2.2 ms is facade work, not a
missing GPU kernel. The independent intrusive Nsight job 3008 records three
frozen calls: strict/Fast shell kernels 85.927/77.475 ms, generic derivatives
10.535/10.549 ms and other kernels 68.586/68.656 ms. Kernel savings alone do not
justify a complete speed claim; facade overhead must remain in the timer.

## Decision and invariants

Do not rebuild the same mathematical basis/geometry identity for every frozen
replay. A prepared precision plan keeps at most one immutable actual-geometry
and model-identity pair per input slot, replacing it when coordinates change
and clearing it on close. The pre-existing calculator-signature check runs
before each execution; topology, charges and multiplicities remain plan-owned.
Equality compares complete validated Atom tuples, not a lossy coordinate hash
or a native density pointer. There is no retained trajectory or device storage.

Only candidate provenance is cached. Every call still executes native work,
checks the complete current gradient/final-state evidence and applies the
requested policy. Auditor model identities are always freshly resolved. No
result, density, error allowance, qualification decision or audit result is
reused. A failed peer cannot acquire an earlier success record. Returning to
the original geometry and explicit moved/frozen inputs retain their actual
public identities; a changed model rejects before resuming an owner.

The scalar helper also stops doing an unused basis-metadata resolution before
the HF model resolver, which already resolves that metadata itself. KS still
uses the same canonical metadata and payload, and output identities are unchanged.

## Evidence and limits

The focused host policy/parser/fleet suite passes 154 tests. Added behavior
coverage checks moved/frozen/original identity, independent peer identity and
fresh returned energies, without pinning internal helper call counts. Existing
audit isolation, seed restoration, mutation rejection and bounded suspension
tests remain unchanged. The broader host subset passes 226 checks; all
touched-file pre-commit hooks pass.

Slurm 3009 requalifies the same HF-DF binary with the new Python controller:
57 HF/DF checks pass, plus 17 prepared-precision checks. One 3c fixture skips
because this node lacks the independent `dftd4` package; it is not counted as
verified under this controller. Existing independent audits and failed-peer
cases run rather than being bypassed by the cache.

Clean job 3010 completes all 72 actual README calls on n2/PRO6000. At 96 AOs,
FP64/Fast ratios for cold/frozen/moved/moved-frozen are
0.9961/1.0483/1.0043/1.0482. Frozen time is 49.492 -> 47.213 ms; moved-frozen
time is 49.674 -> 47.391 ms. The Fast frozen ledger records about 47.032 ms
native candidate / 47.164 ms facade endpoint, rather than the preceding
roughly 2.2 ms facade overhead. These are modest measured warm-endpoint gains,
**not cold acceleration or a universal speedup**.

Maximum 96-AO force error is 6.78e-8 Eh/bohr. Every Fast call still reports one
solve, 29,709 mixed tasks, 52,053 prepared products and 99 task replays, with
matching strict/Fast SCF iterations. No arithmetic or work count changed through
this cache. The 24-AO case still has no mixed operator and pays two strict audit
solves; its cold/frozen/moved/moved-frozen ratios are
0.9076/0.1671/0.4451/0.1684. Retain this negative population.

This is a bookkeeping optimization, **not additional FP32 work**, and applies
to all prepared precision policies. The strict comparator remains the ordinary
optimized Calculator, not a deliberately slower facade. Do not attribute any
recovered endpoint gain entirely to arithmetic or claim small generic-only HF
now has a narrowed operator. See the native HF-DF decision note for its retained
losing/near-neutral populations and the missing all-method/CBS work.
