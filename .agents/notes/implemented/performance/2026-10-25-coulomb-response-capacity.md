# Decision: choose pure Coulomb algebra before optional tensor storage

Status: implemented
Date: 2026-10-25

## Problem

The DF48 AO-force investigation exposed a separate nonmonotonic force cost:
r2SCAN-3c at 36/42 atoms spent about 4.5/8.1 seconds in DF response, while
48 atoms took about one second. Both smaller cases had enough response scratch
for one complete AO-pair/auxiliary tensor but not two. The bridge selected its
single-fitted-tensor fallback before considering pure Coulomb charge algebra.
That fallback applies the metric inverse to every AO pair, even though pure J
needs only the density-contracted auxiliary charge. Reducing available memory
could therefore remove trillions of arithmetic operations.

## Decision and invariants

Determine full-rank raw-source pure-J eligibility before optional tensor sizing.
Use the existing factor-ordered metric-vector solve and authenticated raw charge
consumer regardless of whether one or two complete tensors could fit. Scratch
capacity controls panel sizes and bounded allocation retry, not this algebra.

No equation, cutoff, derivative, grid, source identity or numerical gate changes.
Exchange, retained fitted/occupied factors, truncated metrics and explicit
panel/dot diagnostics retain separate owners. The existing stream, raw-value
lifetime and current-headroom contracts remain authoritative. There is no
atom-count, functional-name or architecture-product admission rule.

## Evidence and rejected alternatives

[The retained report](../../../../benchmarks/results/df-response-capacity-20261025/README.md)
contains 144 main complete calls, 16 separate same-binary calls, 12 independent
moved/property checks, source/binary/cache/Slurm identities, all test failures
and reruns, and offline replay. Standard 36/42-atom warm endpoints change from
6.155/10.206 to 2.219/2.884 seconds. Main paired energies and Fock counts are
identical; maximum force difference is 2.853e-12 Eh/Bohr. Smaller/larger and
exchange controls remain visible. All 159 final related test cases pass.

Do not lower the response budget to force a faster algorithm: that conceals
the algebra-selection error, changes unrelated resource admission, and can fail
mandatory preparation. The first test's HF-only response override did not reach
this DFT owner; the corrected executable boundary test varies the public
allowance and proves the intermediate-capacity failure on the old binary.
The 11 old resident-response error assertions independently fail on the control
because finalization now preserves INVALID_ARGUMENT; updating their category
does not remove their cleanup/recovery checks.

Separate Direct HF experiments reject a larger forced resident ERI allowance:
96-atom cold becomes 24.357 rather than 19.341 seconds. Ordered DIIS observes
18.456 seconds with equivalent outputs, but lacks broader default qualification.
Do not promote residency merely because its allocation fits or pool these
exploratory timings with the README cohort.

## Revisit when

A new owner needs exchange, a truncated metric, or reusable fitted factors.
Qualify its complete scientific endpoint and executed work separately; more
available memory alone cannot justify extra tensor arithmetic.
