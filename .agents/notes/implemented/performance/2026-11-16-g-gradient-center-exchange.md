# Decision: share g-gradient Gaussian-center exchange orbits

Status: implemented
Date: 2026-11-16

## Problem

The earlier [y/z covariance lowering](2026-11-16-g-gradient-coordinate-covariance.md)
reduces the complete g derivative artifact to 26,657,437 bytes, but one complete
three-architecture compiler worker still reaches 7,432,664 KiB. Two unrelated
large workers can exceed the wheel's 12 GiB container envelope. Further source
sharing must preserve g channels, strict FP64 arithmetic, every architecture,
independent oracles, launch schedules and bounded accumulation behavior.

## Decision

Reverse Gaussian centers for g/lower shell pairs and transpose their flattened
Cartesian component indices before calling the retained lower/g helper. Swap
alpha/beta, A/B coordinates and P-A/P-B offsets, copying all remaining geometry
fields so the already rounded radial values are retained. S/T first-center
derivatives change sign by translational invariance; attraction swaps its two
independent A/B roots. Retain the original Boys input array and external center.

For g/g, compose center exchange with the existing y/z covariance. These
transformations commute and leave the differentiated x axis fixed. Select one
representative and its two transform flags per orbit, preferring no exchange
on ties. Representatives map to themselves with both flags false. The wrapper
calls a separate noinline canonical body, not itself. There are 66 retained
representatives for the 225 g/g component pairs. Guard invalid component IDs
before canonicalization or transposition; preserve explicit NaN refusal.

This is a mathematical identity, not bitwise floating-point equivalence:
translation identities can change cancellation and operand ordering. The default
through-f and scoped COSX artifacts remain byte-identical in a one-off proof;
all 32 lower-angular scalar bodies and all 150 mathematical programs are
unchanged. Do not convert this migration proof into required source hashes.

## Complete compiler evidence

The unchanged full `src/scf/cuda/one_electron_derivatives.cu` owner compiles with
NVCC 12.9.86 and verified sccache 0.16.0, retaining sm80, sm90, sm120 and
compute80 PTX, production optimization and the same compiler arguments.

| Metric | Earlier y/z lowering | Center exchange |
| --- | ---: | ---: |
| Generated header bytes | 26,657,437 | 17,589,941 |
| Complete object bytes | 98,958,784 | 64,054,400 |
| Maximum reaped-worker RSS, KiB | 7,432,664 | 4,359,996 |
| Observed complete request, seconds | 2,493.96 | 1,402.38 |
| Cacheable stage misses / hits | 8 / 0 | 8 / 0 |

Source shrinks another 34.0%, or 59.7% from the original 43,685,629-byte g
expansion. Worker RSS is not aggregate cgroup peak. These compiler requests
overlap host/native qualification and are not an isolated cold-wheel speedup.
All 15 architecture/launch entries retain registers, shared and local memory;
stack requirements increase 128--416 bytes. Do not describe this as spill-free
or infer that arbitrary concurrent large owners now fit the container.

## Independent execution evidence

- Full emitted-header host execution covers all 25 s--g shell pairs, five
  geometries, 6,125 Cartesian primitive pairs and all 27 S/T/V A/B/C channels,
  against independent Libcint at `atol=2e-9, rtol=2e-11`.
- Additional host stress covers all nine g-bearing pairs, seven geometries and
  five unequal exponent pairs from 0.0005 through 150: 315 cases / 28,875
  Cartesian pairs. Physically normalized primitives pass the same gate, with
  maximum absolute error 4.45e-12. Retain raw-integral diagnostics separately:
  both variants fail ten raw absolute/relative stress cases at very large
  unnormalized magnitudes; this transformation introduces no new raw failures.
  Do not weaken the existing ordinary-exponent raw-integral gate.
- Slurm jobs 17508/17510 on authorized node n1 execute each complete
  three-architecture owner through the native weighted-gradient launcher:
  750 independent vectors times five schedules, or 3,750 updates per variant.
  Cover every shell pair, unequal exponents, masks/scales/signs, nonsymmetric
  weights, zero/inactive work, three nuclei, nonzero accumulators and arena
  guards. The full-sum gate remains `atol=3e-8, rtol=3e-10`; maximum oracle
  error is 9.02e-14 and maximum old/new difference is 1.84e-15.
- Timings include output clearing and complete weighted updates, retaining AO
  pairs, primitive products and nuclear visits. Five post-warm samples of 32
  updates cover s/s, f/f, g/s and g/g under every schedule. Observed new/old
  medians span 0.817--1.027; default cooperative g/g is 0.992 (explicit) / 0.997
  (implicit). Sequential unlocked-clock jobs are not a universal runtime bound
  or an interleaved, full-method speedup qualification.
- A complete sm120 Release library and CLI build succeeds with automatic
  portable XC AOT. Slurm job 17509 passes all 16 independent PySCF Direct-HF
  RHF/UHF, Cartesian/spherical, cold/warm/moved/restored endpoints, preserving
  every baseline iteration/Fock count. Maximum energy/force errors are
  6.40e-14 Hartree / 8.04e-11 Hartree/Bohr. These Direct endpoints use the
  existing through-f method domain; the complete weighted-gradient gate above
  covers the changed g arithmetic, not an unsupported g Direct-HF claim.
  This initial library is base `405d789c4` plus frozen prototype patch SHA-256
  `dd4698665f179d951c6724f181682879da9dfbce9c500bdd41fbd7e3bc6e48ff`,
  not a falsely clean final-commit artifact. Library SHA-256 is
  `13a7b32e0d5dccf578601414422732b8bc2c040c4e3fca9162e5eb1c1e27e719`.
- The permanent cached host regression expands to both g/s and g/p orders and
  three exponent pairs, independent Boys inputs, seven geometries and invalid
  IDs. The complete focused module passes 306 tests. Keep its canonical helper
  dependencies before reversed callers; do not restore redundant heavy bodies.

The final focused one-electron/catalog/CI-budget suites pass 776 tests with 63
host-run skips. The pinned `ty==0.0.82` check, all 604 compiler modules and the
complete applicable pre-commit hooks pass. Before the master refresh, official
catalog regeneration changes only its two current source identities;
historical measurements, receipts and
qualification claims remain unchanged. Normal catalog integrity passes; the
separate `--require-current` Libxc acceptance gate still lacks a matching
resource/lifetime receipt and is not claimed by this optimization.

A subsequent clean complete sm120 Release library/CLI build at code commit
`6bd41de6fbf5da371dfb7d6769e738c4c0379dc4` also succeeds. Slurm job 17511
repeats all 16 independent Direct-HF endpoints with identical baseline work
counts; maximum energy/force errors are 6.31e-14 Hartree / 8.04e-11
Hartree/Bohr. The source is clean before and after the build. Its library SHA-256
is `8949a3e65d24d76032c5c4703614c3c5d0d85ed11efe2c95d1c1046e3ffc9283`;
the artifact, reports and cache statistics are frozen in `clean-code-native/`
under the investigation directory. Retain its 16 CUBIN hits and 66 stage misses;
the 350-second incremental rebuild is not a cold compile or whole-wheel timing.

The follow-up refresh merges master `01068725898ec46bc2b0ddfd35772e4cf44874cb`
with its independently landed AOT-policy separation and cold-Python CI budget
changes. Resolve the catalog's binary merge from the upstream snapshot, retaining
its resource-manifest receipt, then regenerate current metadata. Relative to that
upstream snapshot only two current source identities and 28 changed-input lists
change; no historical measurements, summaries or qualification receipts change.

## Rejected alternatives and boundaries

Do not drop g support, wheel architectures, compute80 PTX, numerical/safety
tests or bounded fallbacks. The earlier partition-only and global-CSE outcomes
remain recorded in the y/z decision. ECP g/g generated units are not redundant
merely because orbital ECP inputs stop at f: `gaussian_ecp_specs()` deliberately
includes shifts reachable by two derivatives, including g/g from f/f.

Complete manylinux link/repair and installed-wheel acceptance remain separate
gates. Keep the pre-existing COSX grid-plan mismatch separate from this change;
no lifecycle/grid guard is bypassed and no complete COSX endpoint is claimed.

## Reproduction and revisit

Frozen compiler/source/IR proofs, cache statistics, full objects/resources,
Libcint fixtures, Slurm reports and complete timings live under
`/tmp/gqc-wheel-g-center-exchange/`. The retained header SHA-256 is
`fe086bf5b0c2a511ec4f1cb45b05ca7ef3764a7244c2511c30f9e34093aa0eb6`.
Revisit sharing with complete owner, independent gradient/endpoint and resource
qualification when geometry fields, symmetry lowering or the angular domain
change. Do not reuse these measurements for another emitted artifact.
