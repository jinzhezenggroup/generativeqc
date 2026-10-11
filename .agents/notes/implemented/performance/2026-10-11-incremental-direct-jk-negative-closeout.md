# Decision: keep Incremental Direct-J/K opt-in

Status: implemented
Date: 2026-10-11

## Problem and decision

Issue #1598 requires either qualified guarded default promotion or an explicit
negative decision for the exact incremental Direct-J/K route (#990/#1579).
Keep the route opt-in and classify `scf.incremental-direct-jk` as
`negative-evidence`. No runtime/default route or scientific acceptance gate
changes. The option remains available to narrower owners, including #2053.

## Retained evidence and its limits

The reduced capsule in
`benchmarks/results/issue1598-incremental-direct-jk-20261011/` retains one
matched RHF 384-AO baseline/incremental round. The prepared-plan cold execution is
71.076 versus 100.190 seconds; warm medians are 6.835 versus 13.862 seconds.
Incremental is approximately 41% slower prepared-plan cold execution and 103% slower warm for this pair.
Three warm samples per arm pass independent GPU4PySCF and matched-arm energy
and force gates of 1e-8 Eh and 1e-7 Eh/Bohr. The cold native result arrays are
not retained, so cold numerical parity cannot be rechecked offline.

The measured checkout is `d83aa86585a2f72ed852872c533723648b2799c3` (clean).
The README and capsule verifier retain the exact library SHA256, native
source-content fingerprint and common RTX 4090 device UUID. These are measured
identities, not a claim that the publishing head or reported master lineage
was the measurement checkout.

Seven final outcomes record unsuccessful retried points at 192/768 AOs,
without per-attempt diagnostics. They cannot establish SCF nonconvergence,
baseline success, or four independently inspected failed attempts. Two old
source/library receipts remain; no old successful comparison records remain.
The old receipt's checkout identity alone does not reconstruct its source.

The original campaign report described additional sizes, rounds and a small
96-AO cold gain. Those successful records are intentionally absent from the
reduced capsule, so this decision does not assert that every endpoint lost or
that the complete two-campaign matrix can be re-derived. KS evidence was not
independently accepted, and UHF/moved populations were incomplete. Resource
ledgers, progress journals, reducer summaries, device receipts and the old
dirty-source reconstruction patch are absent. The documented broader A-B-B-A
campaign ordering and continuous device isolation cannot be certified from
this subset.

## Why no promotion

A retained matched endpoint regression provides negative evidence; it does not
establish a profitable default guard domain. Failed-point dispositions add
context without attributing a solver failure mechanism. Incremental diagnostics
show activation in the retained pair, but their quartet work counters are
invalid, so no bounded-resource work reduction is certified here.

Refusing broad default promotion does not require a new GPU campaign or a
complete successful qualification matrix. Historical evidence must still be
labelled honestly. The capsule has its own fixed-inventory, checksum and
semantic audit; it is not a standard accepted scientific validation envelope.

## Invariants and alternatives

- Incremental Direct-J/K remains off by default; exact-only, strict-FP64
  admission and the final physical rebuild remain unchanged
- Do not weaken numerical gates or infer convergence failure from an exit code
- Do not retire the option while narrower owners still use it
- Do not promote from this single paired population or extrapolate missing data

## Revisit when

A candidate guard needs provider-owned bounded-streaming work accounting,
independent cold/warm/moved RHF/UHF numerical gates, repeated matched complete
endpoint wins, exact source/library/device identities and admission costs.
A materially different late-SCF work distribution may justify requalification.
The retained runner and reducer support such future work; they do not recreate
missing historical inputs.

## References

#1598; #990; #1579; #2053; capsule README and `capsule.json`;
`manifests/maintenance/default_promotion_inventory.json`, entry
`scf.incremental-direct-jk`. The earlier PBE0-domain rejected evidence remains
separate from this scoped RHF decision.

## Storage closeout and corrected timing scope

The adopted reviewer repair is commit
`0d712298bd59f1842911ae481e3790b9a417df21`, capsule tree
`4db07a60ec635f7914d39936b55c33b20b54037a`. At takeover, upstream
results used 67,061,112 bytes of the unchanged 67,108,864-byte cap,
leaving 47,752 bytes (the prior 2,752-byte headroom report was arithmetic
error). The 58,489-byte capsule still exceeded that allowance. Keep only a
small historical pointer, README and entry point in the results tree; recover
the exact reviewed inventory and verifier from existing Git objects. No other
owner's evidence is removed and no retention policy or validator is weakened.
Shallow clones explicitly fetch the pinned public commit. Recovery checks its
tree, manifest SHA256, every member size/hash and all original semantic gates.
The original scientific limitations survive this storage change. A historical
capsule must not masquerade as a standard accepted publication.json envelope.

The comparator starts cold timing after prepare_batch. Its cold_seconds are
prepared-plan execute-only, excluding preparation/calculator setup; +41% is
not a complete cold endpoint claim. Warm fixed-density E/F regression remains
negative evidence, not proof that every population or complete cold route loses.

Build provenance supplied by the campaign owner: Inspire notebook
`issue-1598-dijk-20261011`, RTX 4090, CUDA 12.9, with
`-Xptxas --disable-optimizer-constants` to bypass the nvlink failure.
The retained library/source hashes bind the measured artifact, but retained
receipts do not independently reconstruct compiler commands or certify this
workaround. No fresh GPU campaign or build is claimed by this closeout.

Future reproduction requires a clean checkout (including untracked inputs);
the reducer rejects dirty measured source rather than implying its HEAD
reconstructs modified inputs. SCF quartet census retains candidate, admitted
and rejected categories separately; admitted-only reduction is not total
screening work. Invalid counters remain unknown with raw diagnostics retained.
KS pooling binds the runner's mode/cadence/threshold, common environment and
exact harness bytes before accepting recomputed native/oracle E/F values.
These reproduction constraints do not requalify missing historical evidence.
