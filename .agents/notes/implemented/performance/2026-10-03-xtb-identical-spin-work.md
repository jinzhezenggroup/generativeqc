# Decision: share identical restricted-spin occupation tasks

Status: implemented
Date: 2026-10-03

## Problem

After retaining native runtime ownership, complete GFN2 energy/force endpoints
still trailed xTBloom. Earlier Nsight Systems captures omitted device-launched
SCC graph kernels, so their visible kernel counts could not explain SCC work.
A diagnostic build forced the existing bounded fallback for a 13-iteration
water-32 calculation, exposing the complete iteration body. Occupation filling
cost about 1.20 ms per iteration; its alpha and beta tasks were identical for
restricted equal-population systems but were solved separately.

The diagnostic build was isolated from performance acceptance. Production keeps
its device-tail graph and its original bounded fallback.

## Decision

Extend the compiler's GFN2 electronic schedule with the number of distinct
occupation solves. After native admission has checked both populations and all
spectra, one restricted spectrum plus exactly equal populations implies identical
temperature, root policy and deterministic arithmetic. Execute one solve and copy
its occupations and `SpinResult` to the second output. Every open-shell or
unrestricted request keeps two independent solves. The predicate never compares
approximate populations or scans spectra for equality.

Root mathematics, bracketing, compensated reductions, translated-energy and exact
degeneracy fallbacks, entropy and failure publication are unchanged. No prior SCC
iteration is reused. Work changes from two to one root solve per eligible system
per iteration, with one linear occupation copy and no additional allocation,
transfer, launch or host synchronization. The two-solve path is the bounded
fallback whenever input identity is not proven.

The same change uses the existing compiler matrix tile width for disjoint final
Hamiltonian and density copies. All compute errors settle before publication;
only tile zero publishes scalar and spin-channel diagnostics. This preserves
whole-system failure suppression without extra scratch or launches.

## Rejected alternatives

- Approximate population equality or merging unrestricted spectra: not a proof
  that occupation tasks are identical, especially near a Fermi degeneracy.
- Change compensated summation or relax root acceptance: unnecessary for exact
  common work elimination and would require separate numerical qualification.
- Treat visible device-tail profiler kernels as complete work: those traces
  omitted the numerical SCC body. The isolated fallback profile revealed it.
- Expect matrix publication tiling alone to remove the gap: its complete endpoint
  benefit was only about 0.7% at 192 atoms and negligible for small cases.

## Invariants and validation

All admission checks run before task sharing, including the second population
and every unrestricted spectrum. Native tests compare every binary64 output
against independently solved duplicate spectra, with mixed ragged systems,
distinct unrestricted spectra, unequal populations, serial/cooperative boundaries
at 63/64/65 orbitals, zero/full/fractional populations, exact degeneracy, zero and
finite temperature, failed and inactive members, and graph replay. Analytic
uniform-degeneracy occupations and electron totals add independent checks.
The full endpoints remain gated against independent tblite fixtures and xTBloom.

Both occupation and matrix native harnesses passed Compute Sanitizer memcheck
and racecheck with zero errors/hazards. The full CPU/CUDA lifecycle, tblite and
force finite-difference suite passed 41 tests. Host schedule/provenance tests
passed seven tests; two explicit GPU opt-in tests were skipped on the host.

## Complete endpoint evidence

On n2 RTX PRO 6000 Slurm job 2060, the ten-case cohort used Release
sm_120/CUDA 12.9.1, one BLAS thread, the same scipy-openblas32 provider, fresh SCC,
300 K, Broyden history 8/damping 0.4, maximum 300 iterations, and energy/charge
tolerances 1e-10/1e-8. Cold 1 + repeated 5 + changed-geometry 5 calls per case
retain all numerical outputs and SCC counts; geometries are checked per sample.

| Repeated endpoint | Retention baseline | Shared tasks + tiled copies | xTBloom |
| --- | ---: | ---: | ---: |
| H2 | 2.751 ms | 2.635 ms | 2.695 ms |
| H2O | 9.594 ms | 8.763 ms | 9.202 ms |
| SiH4 | 15.280 ms | 13.210 ms | 14.475 ms |
| 24 atoms | 43.371 ms | 37.276 ms | 41.507 ms |
| 96 atoms | 139.746 ms | 131.872 ms | 129.164 ms |
| 192 atoms | 351.015 ms | 330.520 ms | 313.194 ms |

Eight smaller cases beat xTBloom in repeated measurements; the two largest still
trail it. This is not an overall xTBloom superiority claim. All 110 SCC iteration
counts match both comparators. Closed-shell root solves fall from 26 to 13 per
water-32 call, and from 28 to 14 per water-64 call. Energy/force maxima versus the
retention baseline are 0 Eh/4.17e-17 Eh/bohr; versus xTBloom,
5.69e-14 Eh/5.17e-15 Eh/bohr.

Ignored receipts are `.artifacts/n2/spin-sharing/`, `.artifacts/n2/publication/`,
`.artifacts/occupation-native/` and `.artifacts/publication-native/`. The candidate
binary SHA256 is `04fbf8d16ebdd94f991313da723309ad45eea058e375fcb1b69104781ccc24d8`;
the retained baseline is `efbc3d0a4b0b374544c2780d48aab9df1b9fb4cb21df295a9276ccd8eb71afed`.
xTBloom main `2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3` uses
`6be47a7183a21c10c8204b6fbeb27e8587bc45d1958e59f5c7d61356777db53f`.

## Revisit when

An occupation provider becomes nondeterministic or gains per-spin controls beyond
spectrum, population and temperature. Extend the identity proof before sharing
such requests. Further large-molecule work should use the full SCC profile;
AES2 potential evaluation is another substantial repeated cost.
