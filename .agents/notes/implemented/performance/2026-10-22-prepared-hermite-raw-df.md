# Decision: reuse normalized Hermite E in resident packed DF generation

Status: implemented
Date: 2026-10-22

## Problem

After the cold endpoint fixes in #2674 (`57c514a0f`), the 96-atom def2-SVP
water fixture still spent about 21 seconds materializing resident fitted B:
12 seconds generating raw three-center values and 9 seconds whitening them.
The generic source repeated Cartesian Hermite coefficients and public spherical
expansions for each output and auxiliary. This is independent of SCF iterations.

## Decision and bounds

Prepare normalized orbital-pair and auxiliary E on the GPU using the existing
Hermite recurrence and public AO layout. Retain only the primitive metadata,
transforms and component-to-primitive intervals; charge/density/potential and
raised derivative scratch are absent. Raw kernels specialize by total orbital
and auxiliary angular degree, reuse the compiler-owned radial/address/parity
algebra, and have one writer per packed public output. Ordered primitive
products, including same-shell products, retain their original multiplicity.
Panel lookup uses sorted public component intervals; panels do not rescan the
whole integral domain or regenerate already completed outputs.

Automatic admission is restricted to fresh single-system resident packed
sources. It uses only capacity left after the unchanged B/U, panel, setup and
SCF reservation, capped at 256 MiB for both optional host and device storage.
It neither replans panels nor borrows response capacity. Explicit mappings or
scalar-math requests, batches and capacity refusals keep the generic source.
Partial allocation failure is transactional. The immutable source owns the
prepared storage; synchronous CUDA frees retire readers before teardown.
Replay allocates nothing and works in captured graphs. Changed geometry/basis
creates a fresh physical source rather than reusing E by pointer identity.

The 96-atom source holds 248,768 orbital primitive products and 1,568 auxiliary
primitives in 101,895,808 device bytes (reported host preparation peak: 107,629,752 bytes).
It still emits 1,067,790,336 raw values in 15 panels into the same 8,542,322,688-byte
fitted B, and performs the same 30 whitening GEMMs / 15,444,519,419,904 FLOPs.
A separate synchronized trace measured raw generation 11.993 -> 1.942 seconds
and materialization 21.042 -> 11.025 seconds. Standalone E preparation was
0.087 seconds. These component times are distinct from clean endpoint timings.

## A link-layout regression found during endpoint qualification

The first integrated library reduced cold time but added about 0.5 seconds to
hybrid-DFT warm endpoints. GPU work and timing were unchanged; the difference
was the CPU final-state validation. Loading both libraries in one process and
calling their unchanged `reference::multiply` on the exact same dense 768-AO
inputs reproduced 74 -> 124 ms per product, with bit-identical outputs. The
new CUDA objects shifted this host function by 32 bytes modulo 64 on n1's
AMD EPYC 9354 (Zen 4). This was not additional SCF work or changed tolerances.

Align the existing fallback function to 64 bytes, as the adjacent Jacobi
fallback already stabilizes its own sensitive loop layout. The paired probe
then measured 73.8 / 73.7 ms, still bit-identical. No loop, arithmetic,
validation criterion or CPU provider selection changes. Keep this small
collateral fix with the integral addition so unrelated link placement cannot
turn a cold improvement into a warm regression. The failed pre-alignment
endpoint receipts are retained, not replaced by the final measurements.

## Scientific and lifetime qualification

- 168 independent libcint raw populations: all orbital s–f / auxiliary s–g
  classes, Cartesian/spherical and mixed representations, signed and long
  contractions, coincident/moved/translated centers and tight/diffuse exponents.
  Every population also exercises complete/ragged panels, dirty graph replay,
  invalid/overflowing ranges, zero host/device budgets, partial allocation
  rollback, allocation-free replay and ledger-clean destruction. Raw gates
  remain `atol=2e-10, rtol=2e-11`; full and tiled outputs must be bit-identical.
- 1,163 existing host Hermite algebra/layout checks pass. E preparation was
  extracted into a shared header, not duplicated; 18 existing device
  charge/projection geometry populations also pass.
- Four memcheck and four racecheck populations report zero errors/hazards,
  including f/f/g and long mixed-basis contractions.
- 62 existing complete DF value/final-state/response tests and two new
  HF/PBE0 cold/warm/changed-geometry cases pass. The new tests also select the
  explicit generic fallback. Independent gates remain 1e-8 Eh / 3e-7 Eh/Bohr
  in the new complete endpoint fixtures; existing stricter gates are intact.
  An initial PBE0 trace-coverage assertion failed because the test started
  observing after `prepare_batch`; the fix observes its full cold boundary.
  Numerical checks passed in that first attempt as well.

## Endpoint evidence

The comparison baseline is #2674, not the earlier master with the occupied
response cold bottleneck. Its measured commit `57c514a0f` and the merged master
`4a783a27e` have the identical tree `264829c86b05182e720b7084f827723c161bc272`. Both libraries use the same CUDA 12.9.1 / sm120
build options and verified sccache 0.16.0. All GPU work runs through finite
Slurm allocations on node1 / RTX 5090, preserving scheduler visibility.
Native Standard/Fast, default XC/VV10 grids, GPU4PySCF's frozen default
auxiliary tables and complete grid/auxiliary force response are unchanged.
Only cold and frozen warm endpoints are timed. README and CBS are untouched.

The initial five-method, two-profile comparison preserves all SCF/Fock counts;
maximum baseline/candidate energy and force differences across all repeats are
2.28e-12 Eh and 4.16e-12 Eh/Bohr. These preservation checks do not certify
absolute default-grid/default-fitting accuracy against a denser direct target.

Final aligned Standard endpoints (seconds; warm is the median of two frozen
calls, and includes complete forces/host export):

| Method | Cold baseline → candidate | Warm baseline → candidate |
| --- | ---: | ---: |
| hf | 43.935 → 33.559 | 4.902 → 5.005 |
| m06-2x | 55.170 → 45.046 | 11.812 → 11.821 |
| pbe | 40.642 → 30.735 | 7.045 → 7.039 |
| pbe0 | 50.831 → 41.003 | 11.164 → 11.109 |
| r2scan | 42.374 → 31.801 | 8.001 → 7.838 |

Across both profiles, cold improves 18–27%. Warm samples range from a 2.4%
improvement to a 2.1% regression; no warm acceleration is claimed. The two
hybrids were rerun in reversed candidate/baseline order in the same allocation
after fixing the link-layout regression. HF/PBE/r2SCAN retain the earlier
baseline receipts and new final-image candidate receipts. All populations
complete and preserve SCF/Fock counts and the numerical gates above.

Baseline core SHA-256:
`a7fe1d09bdd07fce2265d6b03a213db2867bd6e370d00b5ccbf4cfa1781e62fb`.
Final candidate core SHA-256:
`b51b85ea8324bc104bbb7f3264c3220324a3c7e21c338ea714441749ae78c2ad`.
The final aligned image passes the 64 complete DF tests. The independent
raw and shared-E suites additionally qualify the same integral code before
the alignment-only change; the paired host probe proves bit-identical matrix
products across the change.

Artifacts, including unsuccessful attempts, are retained locally under
`.artifacts/` and at `n1:/data/jzzeng/df-integral-generation-20261022/`.
The source-matched build, native images/AOT manifests, launch scripts, raw
inputs/outputs, traces, test XML and checksums are retained separately from
this durable rationale.

## Rejected alternatives and remaining cost

Changing the old generic lane mapping alone did not remove repeated E work.
Do not enable unqualified scalar polynomial/Rys variants for auxiliary g.
Do not replace factor-ordered whitening by a preformed inverse-root product:
that loses cancellation in weak metric directions. Whitening still costs
about nine seconds on this fixture and is outside this integral-generation
change. Broader streamed or batched admission needs separate work/capacity
and complete endpoint qualification.
