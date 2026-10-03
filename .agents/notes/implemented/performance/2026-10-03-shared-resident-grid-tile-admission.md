# Decision: share complete resident-grid tile admission

Status: implemented; experimental opt-in, default unchanged
Date: 2026-10-03

## Problem

Ordinary PBE0 stationary forces fixed the AO tile at 256 points, whereas the
composite nonlocal consumer preferred 1024 under its own resource inventory.
The kernels are shared, but neither the simultaneous owner inventories nor
their host budgets are interchangeable. Merely increasing the ordinary default
would reject some previously admitted systems or conceal additional memory.

## Decision

Use one compiler-owned finite tile search with caller-owned dry admission.
Both consumers try the same candidates; each retains its exact existing
resource formulas and native fallback reservation. Explicit requests are never
resized. Ordinary automatic selection remains opt-in via `tile_points=None`;
the existing 256-point default is unchanged pending full qualification.

The ordinary resource preflight is extracted without changing its equations.
Each candidate also checks the bounded pending pair-visit window. Selection
precedes artifact lookup, native allocation and prepared-owner replacement.
Prepared identity already binds tile capacity, so replay cannot silently reuse
an owner with a different layout.

## Invariants and limits

- No change to functional, density, precision, source coverage, grid points,
  AO membership or error gates.
- Complete concurrent AO/source/TensorIR/native capacities remain admitted
  under both byte caps, including the existing conservative host allowance in
  the device-bound grid plan.
- Whole-grid nonlocal storage cannot be shrunk by trying a smaller AO tile.
- More point lanes do not reduce `G*M^2` dense AO contractions or `G*A^2`
  partition visits. Active-AO producer/caller work is a separate change.
- Runtime XC provenance reads the same cached `XCfun.on_gpu` flags used by
  GPU4PySCF, outside the endpoint timer. Missing flags mean unknown, not CPU.

## Evidence

The 235 focused host checks cover both consumers, budget fallback, explicit
requests, the unchanged Direct/DF native allowance and reference provenance.
An additional 31 ownership/DF-scope checks pass. Existing exact ordinary
24/48/96 resource fixtures remain unchanged for 256-point tiles.

For README water-96/768 AO, the full-AO 1024-point grid alone needs
603,451,392 numeric bytes, exceeding the ordinary 512 MiB device allowance.
Even raising only the device cap still fails the 256 MiB host cap. Automatic
selection therefore retains 256 at this size; 24/48 admit 1024. A 96-atom
256/1024 experiment with both caps explicitly set to 1 GiB is a separate
capacity regime, not evidence of a speedup under the defaults.

Fresh matched full endpoints completed in n1 Slurm jobs 5542/5543. All 144 native
and 72 reference calls pass independent all-same-geometry-repeat comparisons
under the unchanged `1e-8 Eh` / `1e-7 Eh/Bohr` gates. Maximum errors are
`1.051e-10 Eh` / `3.037e-11 Eh/Bohr`. Both native variants use the same
fb53bb548-based source/binary and allocation within each size.

At default caps, warm 24-atom time changes from 5.036699 to 4.585956 seconds
(8.95% less); 48 atoms changes from 17.961313 to 15.177445 seconds (15.50%
less). Dense `G*M^2` work and partition pair counts are unchanged. This does
not close the independent-reference gap: the fresh 24-atom reference is
2.201190 seconds. The runtime reference XC cache reports functional 406 on GPU.

With both caps explicitly raised to 1 GiB, 96 atoms changes from 77.349014
to 66.358783 seconds warm (14.21% less). Its moved endpoint is worse:
289.241844 to 341.733907 seconds, with 14 versus 18 SCF iterations. All
iterations and negative outcomes remain in the evidence. These are not
default-budget timings: default automatic selection at 96 retains 256.

Ordered processes share disk compilation/artifact caches. Cold measurements
are retained but cannot establish an isolated cold-compilation speedup.
The source snapshot precedes the separate #1767 claim synchronization repair;
these observations are not relabeled as measurements of that corrected binary.
No default promotion is claimed.

Review found one additional prepared-only admission charge: ECP tensor owners
retain the sum of their host storage on top of the ordinary host bound.
Candidate admission now includes that existing prepared-owner charge, so an
automatic 1024-point rejection can still try 256 before artifact lookup. A real
two-atom/12-AO ECP plan reproduces the former mismatch: at a 13,387,008-byte
host cap, 1024 requires 13,809,216 bytes and 256 requires 10,251,840 bytes.
Tests cover automatic fallback, explicit refusal, unprepared behavior, a
tighter cap, and the actual prepared admission boundary. This post-measurement
repair does not change the empty tensor inventory of the PBE0 campaign, but
the retained measurements still identify the exact earlier source and binary.

The capacity qualifier now follows the extracted resource helper and nested
candidate callback, rather than merely accepting a new endpoint hash. It
retains the original resource equations, native reserve and work checks, binds
the complete helper/layout/endpoint, and checks selection before page execution.
Negative controls also cover helper capacity, layout, selected owners and the
prepared tensor charge. Moving resource logic out of the endpoint must not
remove it from fail-closed qualification.

The lossless complete campaign and executable verifier are retained under
`benchmarks/results/pbe0-resident-tiles-20261003/`. Stored members include the
exact numerical records, source patch, harness, work/budget and build receipts.

## Rejected shortcut: CPU bounding boxes on every contiguous tile

An eight-tile diagnostic sample at each of 24/48/96 atoms compared the existing
order-two `ao_region_envelopes` against the actual GPU maxima retained in
job 5540. Median CPU envelope times were approximately 0.032/0.065/0.131 seconds
per sampled 1024-point block. No sampled omission violated the `1e-16` AO-jet
cutoff. These are samples, not whole-grid performance or work measurements.

The outer radial blocks expose a structural problem: a huge spherical shell's
axis-aligned bounding box encloses the whole molecule. At water-96, sampled
blocks 23 and 2303 retained all 768 AOs although actual sampled-jet maxima
retained zero. Reusing the current box API blindly would pay substantial host
setup while discarding important locality. This does not reject spatial
screening generally; better regions or resident pointwise maxima may avoid it.

Raw CPU sampling scripts and receipts remain in ignored `.artifacts/`.
Active-AO API qualification is separately
tracked by PR #1768; this change does not claim to enable sparse production.
