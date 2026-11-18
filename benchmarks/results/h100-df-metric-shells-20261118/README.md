# H100: shell-pair auxiliary metric derivative for DF HF/hybrid forces

The DF HF/hybrid force contracts the metric derivative `dM[P,Q]` with a full,
nonsymmetric weight matrix. The previous route evaluated it per pair of
auxiliary AOs, with the generic primitive-product policy. For the README DF
auxiliary (cc-pVTZ-JKFIT, uncontracted s..g) on water-96, that is 89 million AO
pairs. On H100 `metric_pair_derivatives<6>` took 115 ms per call, about 13% of a
DF-HF96 warm call. GPU4PySCF's `int2c2e_deriv` takes 7.7 ms on the same device.

The new route runs one thread per ordered auxiliary shell pair. A harmonic
degree-l AO polynomial is an exact combination of degree-l Hermite Gaussians
(Hobson). So the thread projects `W[P,Q]+W[Q,P]` onto those monomials and contracts
it with the existing DF radial recurrence, raised by one order. It differentiates
only the first center; the ordered sum covers both. Same-atom pairs cancel exactly
and are skipped. The 25 class kernels together take 0.62 ms per call.

## Measurements

Fresh consecutive processes on one physical H100 80GB in qz Job
`gqc-h100-opt-1118-qa`, all from one build (identity `a9da4474…`). In that build,
`GENERATIVEQC_DF_METRIC_DERIVATIVE_SHELLS=0` keeps the previous per-AO pair route,
which gives an interleaved same-binary A/B (legacy, shell, shell, legacy). The
build also contained the then-unmerged #3115/#3116/#3118, the
#3119 auxiliary-g Rys code (its `sm_90` candidate is not selected by `auto` in
that tree) and the stationary `geometry_reduce` load batching. These are common
to both arms.

| Case (DF) | legacy warm (s) | shell warm (s) | legacy cold | shell cold |
| --- | --- | --- | --- | --- |
| HF 96 | 0.848, 0.864 | 0.748, 0.733 | 3.181, 3.336 | 3.072, 3.092 |
| B3LYP 96 | 1.643, 1.637 | 1.527, 1.524 | 5.310, 5.305 | 5.191, 5.159 |
| HF 48 | 0.184, 0.182 | 0.152, 0.149 | 1.126, 1.134 | 1.122, 1.113 |

Warm values are medians of three calls. Iteration and Fock counts are identical.
Every arm keeps its independent qualification status: B3LYP qualifies, and HF DF
stays failed for the existing fitting-energy approximation, with unchanged errors.
Same-phase forces from the two routes agree within 1.8e-12 Eh/Bohr.
`endpoint.json` holds every arm, sample and qualification error;
`results.json.xz` adds all force vectors.

## Independent gates

- `oracle.json`, part A: the standalone weighted bridge (`bar_a = 0`) against libcint
  `int2c2e_ip1`, through `tools/generativeqc_validation`. Fixtures: the
  `test_df_derivatives_cuda.py` ones plus a contracted s/p/d/f/g auxiliary with
  several shells per atom. Weights: random, upper, lower, skew and diagonal, in
  both representations and under both controls. Every row passes
  `atol 2e-9, rtol 2e-11`; the largest error is 2.2e-12 on gradients up to 35.
  Cartesian d and above stay on the per-AO route. An invalid control rejects,
  and the next call recovers.
- `oracle.json`, part B: README water-12 DF-RHF energy and forces against PySCF
  with auxiliary response, under shell, legacy and shell in one prepared batch.
  Energy error is 6.8e-13 Eh and force error at most 8.8e-12 Eh/Bohr under both
  routes. `metric_derivative_shell_panels` is 1 only on the shell calls.
- `tests.txt`: the full `tests/python/test_df_derivatives_cuda.py` of this change
  on the same build, 54 passed, including the new class-coverage and control tests.
  `SLURM_JOB_ID` was set only to satisfy the scheduler guard on this qz job.
- `memcheck.txt`: compute-sanitizer 12.9.79 memcheck over one water-12 DF-HF
  energy+force call on the shell route. 0 errors.

These measurements do not qualify any other GPU. The route itself has no
architecture switch.
