# Decision: reuse CPU AO axis derivatives across Cartesian jets

Status: implemented locally; performance qualification below
Date: 2026-10-01

## Problem

`AoBasis::evaluate` already shares each primitive's radial exponential across its
jets, but recomputes the same one-dimensional polynomial derivative when that
axis/order appears in multiple Cartesian jets. This is production work in the
CPU DFT singlepoint endpoint, not an isolated alternative evaluator.

## Decision

Within each existing point/AO/primitive/Cartesian-expansion term, retain each
axis derivative from zero through the requested order in a fixed stack array.
Every jet consumes these values in the original x/y/z multiplication order.
The original polynomial differentiation and Horner evaluation remain unchanged.

Per non-underflowed primitive/expansion term, polynomial evaluator calls change:

| Requested order | Jet count | Before | After |
| --- | --- | --- | --- |
| 0 | 1 | 3 | 3 |
| 1 | 4 | 12 | 6 |
| 2 | 10 | 30 | 9 |
| 3 | 20 | 60 | 12 |

The value-only path retains its direct traversal and adds no scratch. Derivative
paths add 48/72/96 bytes, fixed independently of molecule/grid size.
The derivative-zero polynomial bypasses coefficient-array construction but
retains the original `result * x + 0.0` Horner operations.
No heap allocation, persisted geometry state, grid screening, density fitting,
precision change, or new resource-dependent fast-path admission is introduced.
Exponential, primitive/expansion, output, and SCF work counts are unchanged.

## Rejected alternatives

Do not replace the recurrence with closed-form derivatives, reassociate sums, or
relax FP64 to obtain a timing win. Those would need a different numerical review.
No cross-geometry cache is needed for this strictly local invariant reuse.
An initial all-orders temporary-array version showed no reliable value-only
benefit, so the direct value-only traversal was retained before final testing.

## Invariants

Preserve coefficient differentiation, Horner order, x/y/z products, and each
jet's primitive/expansion summation order. Preserve underflow short-circuiting,
nonfinite checks, selected/partial AO tiles, and empty output handling. All legal
spatial orders 0–3 and Cartesian/real-spherical bases through f remain supported.

## Verification and evidence

`tests/python/test_grid_cpu.py` now checks the 1/4/10/20-jet specializations against
all six independent libcint fixtures, including Cartesian/spherical f shells,
diffuse and tight functions. Existing tests cover finite differences and tiles.
The local CPU suite passes 57 native tests and 85 focused Python tests (4 CUDA
skips). Independent same-grid r2SCAN/STO-3G water SCF agrees with PySCF within
4.3e-14 Hartree, with unchanged baseline/candidate energy.

`benchmarks/cpu_ao_endpoint.py` retains all cold, repeated one-shot process-warm,
and changed-geometry endpoint samples, their energies/iterations, and native
binary SHA-256. It does not time compilation/import or claim prepared-plan reuse.
The `--forces` option is only for methods which advertise that property; r2SCAN
at this checkout does not, so the qualification is energy-only.

Matched GCC 14.2 `-O2` single-thread results (three alternating process pairs,
six warm samples per variant) on water:

| Method/basis | Baseline seconds | Candidate seconds | Time reduction |
| --- | ---: | ---: | ---: |
| LDA/STO-3G | 0.42703 | 0.34106 | 20.1% |
| r2SCAN/STO-3G | 2.37882 | 2.19355 | 7.8% |
| r2SCAN/def2-SVP | 8.77848 | 8.22731 | 6.3% |
| PBE/STO-3G | 0.74730 | 0.73462 | 1.7%, small/noisy |

All 60 paired cold/warm/changed-geometry energies were identical, as were
iterations and recorded Fock/grid work. First-call and changed-geometry medians
also improve for the three qualified DFT cases. Def2-SVP r2SCAN independently
agrees with same-grid PySCF within 1.14e-13 Hartree.

The short RHF control initially appeared 7.2% slower; four further alternating
pairs with 20 warm calls each yielded 39.81 ms baseline versus 39.27 ms candidate.
Keep both observations: this is not evidence of HF acceleration or a confirmed
regression. No DFT claim is generalized to HF, CUDA, forces, large molecules or
multithread scaling.

Separate three-endpoint interposition profiles preserve actual AO counts:
LDA 9,720 calls / 17,418,240 AO-point pairs; r2SCAN 8,640 / 15,482,880. Inclusive
AO time changes 0.866→0.572 s and 2.787→2.256 s, respectively. These are component
attribution only; the table above measures complete singlepoint endpoints.

The local evidence bundle retains every raw JSON, tests, profiling interposer,
reproduction driver, cold/changed-geometry summaries, excluded pilot failures,
compiler flags and binary SHA-256 identities. These measured binary hashes are:

- baseline: `8d4fe9577387b3f5e689a47d447f75b2ec3a2fa1b658f71955a97465732a8bb1`
- candidate: `041ec5830d6015383067963946e4729b98f8d3751d608b4c964ea4b1af4274f7`

## Revisit when

A compiler-generated CPU AO evaluator replaces this independent reference owner,
or larger-shell/order support changes the fixed-size derivative recurrence.
