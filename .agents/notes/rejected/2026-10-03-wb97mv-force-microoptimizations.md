# Decision: reject two WB97M-V force defaults without endpoint benefit

Status: rejected
Date: 2026-10-03

## Problem

After scaled Boys LR moments and retained SPD sources improved resident SCF,
full24 force work remained dominated by nonlocal pairs and RSH derivatives.
Two plausible alternatives were implemented and independently qualified before
being rejected by measurement. Their production changes were reverted.

## Canonical RSH derivative dispatch

Preferring the existing canonical derivative source over the bounded shell join
passed all six independent complete RKS/UKS and displaced-energy tests. However,
the same full24 fixed-state intrusive profile increased the complete force from
50.53 to 63.08 seconds and integral derivatives from 18.53 to 31.23 seconds.
The canonical implementation already evaluates Full and LR and derives SR by
subtraction; proposing removal of a third SR recurrence repeats a false premise.
Retain the bounded shell route until a new endpoint-qualified schedule improves it.

## Shared reciprocal geometry in prepared rVV10

A separate bounded prepared-geometry closure reused feature reciprocals, reducing
six FP64 divisions to four while leaving energy/feature IR roots unchanged. It
retained the ordered original closure outside positive [2^-32, 2^32] feature
ranges. Host tests (101 passed, two skipped), independent 90-digit energy
finite differences, extreme-domain fallback tests, and all six complete device
tests passed.

Node1 job 5378 compared 40000-point kernels (1.6 billion ordered pairs): about
0.158 seconds original versus 0.136 seconds with reciprocal reuse. This roughly
16% kernel improvement cannot improve this WB97M-V endpoint: it changed the
rVV10 closure, while the method dispatches VV10. The retained profiler names
`pair_kernel_ordered<(Vv10Variant)1,...>` and the enum defines VV10=1, rVV10=2.
The microbenchmark explicitly selects rVV10. Thus the endpoint comparison below
is not evidence against applying reciprocal reuse to the actual VV10 closure;
it is evidence that the experiment targeted the wrong specialization.
Node1 job 5379 used the same GPU, water24/192 spherical def2-SVP AOs, the full
48x16x32 grid (589824 points), three timed warm samples and matched reference
VV10 masks. Original cold/warm median was 355.527/65.2817 seconds; reciprocal
was 356.402/65.1375 seconds, with overlapping warm spreads. Both remained about
2.36 times slower than the paired GPU4PySCF warm endpoint (~27.55 seconds).
All five pairs for each native variant passed: maximum energy error <3.28e-11
Eh and force error <2.99e-10 Eh/Bohr. No acceptance gate was relaxed.

## Evidence retention and revisit conditions

The composed checkout retains rejected patches, original source archives,
verified ccache receipts, independent tests, profiles and comparator JSON under
ignored `.artifacts/wb97m-canonical-force/` and `.artifacts/wb97m-reciprocal/`.
The reciprocal library SHA256 was
`34a47c2bd722eefdf060a3fc4c75991769ed64cced10369855449805f95a57e4`.
A library left in that experiment build directory does not describe subsequently
reverted source; rebuild before drawing new provenance conclusions.

Before revisiting, verify the method registry, emitted closure and measured kernel
specialization agree. Revisit when a different schedule, workload or generated closure demonstrates
repeatable complete SCF plus analytic-force benefit with semantic work counts,
independent per-sample numerical gates, and preserved bounded fallbacks.
