# Decision: native moving-grid mixed jets retain exact atomic measures

Status: implemented (geometry source component, not a complete Hessian)
Date: 2026-10-16

## Problem

The Python-free fixed-grid density kernel does not supply nuclear grid motion.
WB97M-V nuclear response needs AO value/gradient motion, owner-point motion and
the first/mixed-second Becke weights. Recovering atomic measures by dividing
partitioned weights fails at exact-zero partitions and cannot authenticate the
actual quadrature. Building molecule-sized scalar AD graphs would also make
runtime generation and work depend on atom-count specializations.

## Decision

Extend the internal live KS grid lease with optional actual CUDA atomic-measure,
point-owner ordering, partition policy and quadrature-owner provenance. Reborrow
and compare these fields with the accepted basis/grid/density binding at every
execution/publication. Existing host-only quadratures retain fixed-grid leases,
but do not gain moving-grid capability through reconstruction. No public ABI or
method capability is promoted.

The build emits scalar first/mixed jets from the existing norm/ratio/Becke graph
and compiler AD of multiplication, addition, subtraction and complement. Native
traversal uses those primitives, not a second set of derivative formulas. AO
motion uses compiler AD of a spatial Taylor polynomial supplied with incumbent
order-three Gaussian jets. A full xy tensor contraction has its correct mixed
multiplicity without modifying the stored ordinary jet convention.

All four components of each partition product share a binary mantissa/exponent
representation. Rescale by exact powers of two based on the largest component,
not solely the primal; a zero primal may still have nonzero mixed response.
Align products to one common exponent before the generated quotient. This is
representation/scheduling, not differentiation of the exponent selection.
Never divide by a switching factor. Finite/smooth-domain checks reject coincident
centers and multiatom point-center norm singularities. One-atom grids avoid
those norms and permit a point exactly at the center.

Pair-center jets are generated once per action, then reused across point tiles.
Scratch distances/products scale as point-tile times atoms; pair storage scales
as atoms squared. Cached AO jets and staged AO/weight outputs scale with the
full declared point/AO extent. Arena bytes, point-pair visits and AO values have
separate checked bounds. Preparation also bounds Gaussian primitive/polynomial
term visits conservatively by 20*points*AO*primitives*3. A small jet-output count
does not bound the number of contraction primitives. There is no hidden
atom-count cap or CPU/JIT retry.
The two caller outputs publish only after numerical and live-owner gates; a
CUDA failure during their final copies invalidates both.

## Numerical gates and rejected alternatives

- **Strict fixed absolute tolerance on weighted primal reconstruction:** the
  actual native H2 radial grid contains atomic measures above 1e6. Hypot versus
  scalar-norm and normalization rounding in tiny switching tails can therefore
  become about 7e-11 in weighted units despite sub-ulp-scale partition error.
  Authenticate the supplied primal in dimensionless partition units, using
  only the actual atomic measure. Default absolute/relative partition gates are
  1e-15/3e-11. Zero atomic measure requires zero reference weight. Negative
  measures/weights and nonfinite inputs reject. This is not division by a
  partition weight or regeneration of the primal.
- **Long-double differences as the sole molecular mixed oracle:** radial
  measures also amplify subtractive cancellation of displaced values by 1/h^2.
  Distorted H3 showed a 2.4e-8 floor in one weighted mixed component. Retain
  independent 60-digit Decimal value-only displaced partitions for molecular
  response qualification, forming differences before FP64 conversion. Production
  derivatives remain compiler AD; no finite differences enter native execution.
- **Handwritten second product/chain rules:** unnecessary duplicate scientific
  authority; use generated scalar jet operations and Taylor AD instead.
- **Passing merely finite packed records to the generic AO leaf:** the incumbent
  leaf intentionally trusts normalized AoBasis records. A finite but invalid atom
  index can read primitive storage as coordinates and then index beyond a nuclear
  direction. The geometry owner now device-validates integer atom/range/term/power
  fields and positive exponents before that leaf. A previous scalar error prevents
  later finite scans from reading unpublished AO storage. Rejection and all four
  sanitizer tools cover the bad-record and nonfinite branches.
- **CPU atomic-measure rebuild or weights/partition recovery:** detached from
  actual storage/policy and invalid at rounded-zero weights; fail capability
  admission when the native source is absent.

## Evidence and scope

`tests/python/test_native_grid_geometry.py` supplies independent polynomial
Gaussian jets through order three, including s/p and real-spherical d/f records,
and independent displaced Becke values at two displacements. Gates cover first
and mixed response, translation/left-right symmetry, same-direction curvature,
one atom, 40 atoms over multiple tiles, rounded-zero factors, scaled/zero atomic
measures, invalid sources and transactional publication. Host-only overflow/work
admission is exercised without CUDA initialization. The actual molecular probe
borrows converged native RKS/UKS Direct/DF grid owners and revokes both geometry
outputs after a new solve, with Python/compiler paths unavailable to the process.

Pre-final Slurm node2/main/PRO6000 job 3108, assigned visibility 0, passes six
resident-frame lifecycle cases and 30 geometry tests. This is component
correctness evidence, not a complete molecular Hessian or endpoint timing.
Final Slurm job 3110 on the same node/partition/GPU type and assigned visibility
0 passes six lifecycle cases and the integrated leaf/GMRES/CPKS/grid/Fock/geometry
regression: **221 passed, four host-compiler/cache prerequisite checks skipped**.
Those four pass locally within **47 passed, 178 opt-in GPU cases skipped**.
The new host-only geometry overflow/work admission separately passes without
CUDA initialization. Geometry's 36 cases are all included in real-device
qualification, with no geometry skip. All **32 sanitizer runs** are clean:
memcheck/initcheck/racecheck/synccheck over actual RKS/UKS Direct/DF owners,
one-/40-atom sources and invalid-record/nonfinite preparation. Racecheck reports
zero warnings as well as zero errors. Pinned ty 0.0.82, 556 compiler modules,
249 shared SCF modules and the CUDA-disabled method/adapter syntax gates pass.
No libpython dependency appears in the qualified probes or shared library.

Qualified binary SHA-256 values (ordinary Release, sm_120, fast compile off):

```text
7dc29b64d14255020c68a914f997e994270c143b2698015fcd2afe423a87d3c6  generativeqc_cuda_grid_geometry_probe
faebafac2d462453c1b3b46d7103948c8add8796512c004db919da800ebe57b1  generativeqc_ks_wb97mv_grid_probe
9b9d83b260358a05468e54bae8253d935a1f55be4a4476e38a9e0b721b28dac3  generativeqc_ks_resident_frame_tests
d580072f29f4a18fa579825603c75a3ba4fbef51bae1337854ccc77d1a24d8f5  libgenerativeqc.so.0.1.0
```

Raw local receipts are retained under ignored
`.artifacts/wb97mv-hessian-aot-20261015/geometry-*`; the source-matched remote
stage is `n2:/data/jzzeng/wb97mv-hessian-native-response-20261015/geometry-20261016`.
Both CXX/CUDA launchers invoke verified ccache 4.5.1; compiler commands and shared
cache statistics are retained before/after compilation. Shared statistics are
not isolated performance measurements. Test durations are not Hessian timings.

Current runnable commands and remaining integration boundaries are documented in
`docs/developer/dft_hessian.md`. Moving XC/VV10 frozen-Fock and stationary scalar
contractions, integral/DF/ECP nuclear sources and full Hessian orchestration are
still missing. The production CLI still uses the explicit Python bridge.

## Revisit when

A qualified stationary owner needs screened/local AO panels, a different grid
owner ordering or a larger admitted point-pair budget. Preserve independent
mixed-derivative gates, actual raw-measure ownership and explicit work/storage
admission rather than silently introducing an oracle or regenerated source.
