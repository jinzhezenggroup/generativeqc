# Decision: reuse raw pure-J values and bound response by live headroom

Status: implemented with bounded raw-value default; response allocation fix
Date: 2026-10-23

## Problem

A pure Coulomb method needs the action of the fitted operator on a vector,
not every element of a whitened three-center tensor. Water96/def2-SVP retains
about 8.54 GB of packed values. Whitening all of it adds cold preparation even
though neither K nor occupied projection consumes the result.

Separately, a prepared response allowance is not a physical CUDA reservation.
The 96-atom DF WB97M-V warm force failed after a nonlocal consumer grew:
the response requested 4,320,748,032 bytes with 4,263,903,232 bytes free,
despite fitting its 4,335,240,806-byte declared allowance. A small isolated
pressure fixture also demonstrated that an allocation can fail below the
reported free count, owing to allocator granularity or fragmentation.

## Decision

Capture the raw-value request and its explicit `0`/`1` control in
`FockExecutionVariant`. Unset requests raw storage; `0` retains whitening.
Only restricted pure-J, source-backed, resident
single-packed owners with zero occupied rank capacity may request it. The
existing metric rank cutoff and available vector scratch decide actual
admission. Rank truncation or insufficient capacity retains the fitted owner.
Invalid control spelling is an error.

Admitted plans retain raw A in the existing `three_center` allocation and
compute `A M^-1 A^T D` using the original eigensystem. They cannot supply
fitted K, and raw storage cannot be published as a whitened-response lease.
The force bridge authenticates the packed raw owner, folds the current density
into existing workspace and directly contracts A. It then uses the existing
metric solve and analytic derivatives. Borrowed bytes stay charged to their
owner; there is no second tensor, alias ownership or new production oracle.

For every response call, bound panel planning by the minimum of the declared
remaining allowance and current free device bytes. A failed CUDA workspace
allocation may halve only the auxiliary panel before any response work starts.
The retry is finite and retains fixed factors/source storage. A request that
cannot shrink, host allocation failure, or non-allocation CUDA failure still
propagates. Cleanup drains the stream before releasing borrowed lifetimes.

## Rejected alternatives and retained boundaries

The first raw experiment reused the value tensor for J but regenerated raw
values during force response. Its PBE cold endpoint grew from about 25 to
45 seconds and warm from 6.5 to 30.6 seconds. A diagnostic attributed roughly
24.2 seconds and 17 GB of generated values to this missed resident borrow.
The packed raw charge reader fixes that work amplification.

Do not force raw residency for r2SCAN-3c96: its 1,088 orbital and 6,816 auxiliary
AOs need about 32.3 GB for the packed tensor alone. It correctly retains the
streamed owner under its existing complete resource allowance. Raising the
budget or claiming memory bounds prove work bounds would conceal this limit.

## Evidence and qualification boundary

Historical candidate-v3 single-process observations were PBE DF96 cold
25.22481 to 20.73441 seconds and r2SCAN 27.22007 to 22.78818 seconds, with
roughly unchanged warm time. These remain experimental observations, not
default-promotion or repeated performance claims. Their sources included an
independent triangular-Schwarz experiment; the clean second-stage branch
excludes it. Source manifests, immutable libraries, failed pilots and trace
streams remain under the local and n1 `direct-dft96-cold-20261023` artifacts.

The headroom-only v4 WB97M-V diagnostic completed cold/warm in
694.983980/105.959470 seconds, passing the retained independent energy/force
gates. The old image failed warm. This diagnoses resource correctness; the
cold time remains behind GPU4PySCF and is not a speedup claim.

The native `--raw-coulomb` fixture checks unset/explicit/disabled selection,
independent CPU J and derivatives,
changed density/geometry, post-force replay, rank-truncation fallback and
invalid controls. `--df-response-headroom` warms the complete force path,
retains unrelated device memory, compares the constrained force with its
unconstrained result and checks retained J afterward.

The clean opt-in source `8b68f626c82e7bf0f33bd6a993f91ec64b5c790b`
completed five alternating fresh-process pairs per method. PBE DF96 median
cold changed from 25.131312 to 20.703457 s (17.619%); r2SCAN from 27.171812
to 22.442908 s (17.404%). Each process retained three warm repeats: neither
warm difference passed the unchanged 2%/relative-MAD gain gate. Five additional
48/96-atom moved pairs per method qualified all four phases. At 96 atoms,
moved medians improved 20.421% and 19.229%; 48-atom and moved-warm differences
were not significant. All 264 calls passed: stationary samples against dense
independent references and moved samples against fresh same-geometry GPU4 DF
references. The Standard/Fast matrix includes six functionals; WB97M-V headroom
has separate diagnostic evidence and is outside raw-J admission.

The same source passed raw-J independent native numerics, actual external-memory
pressure, Compute Sanitizer (zero errors), full native Fock composition, 55 GPU
Python cases and 82 host cases. Source-matched evidence for the subsequent unset
default and reviewed stage-one integration is retained separately; historical
opt-in receipts must not be relabeled as that integrated default.

## Revisit when

Broaden raw storage only after source-matched complete cold/warm/moved evidence
and independent numerical gates qualify the new domain. More extensive
allocation recovery needs a different ownership plan if fixed response
storage itself cannot fit; it must not silently switch numerical algorithms.
