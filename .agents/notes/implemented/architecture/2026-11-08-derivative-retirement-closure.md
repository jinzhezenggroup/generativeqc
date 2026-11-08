# Decision: production derivative mathematics belongs to differentiated primals

Status: implemented; final scoped scientific and endpoint qualification complete
Date: 2026-11-08

Subsequent CI integration and storage-only recovery decisions are recorded in
[the integration note](2026-11-08-derivative-ad-ci-integration.md). They do not
relabel the frozen measurements in this closure.

## Scope and ownership census

This closes the source-migration items identified in the November 7–8 notes.
It does not remove differentiation engines, independent references, or the
linear execution of compiler-produced response data. Moving a supplied slope
into a source template, or adding an unrelated AD witness, is not retirement.

| Production owner | Mathematical authority after migration | Retained native/runtime responsibility |
| --- | --- | --- |
| Nuclear energy, gradient and Hessian-vector products | `method.nuclear_repulsion`, common scalar Graph AD | Pair admission, accumulation, sparse atom incidence |
| D3 BJ/zero/ATM, D4 charge/CN weights, EEQ radial terms and gCP | `geometry.dispersion_scalar`, `geometry.gcp_scalar`, existing differentiated D4/GeometryIR energies | Tables, bounded EEQ solves, traversal, cutoff branches, linear cotangent contractions |
| CPU integral Jet and CUDA Dual/Dual3 arithmetic, radial moments | `integral.scalar_ad`, AD of the Gaussian integrand and qualified value moments | Recurrence traversal, special-function value evaluation, storage and channel layout |
| AO Gaussian axes, angular/radial envelopes and displaced AO products | `dft.ao_axis`, `dft.ao_response_ad`, `dft.density_bilinear_ad` | Bounded collocation/GEMMs, symmetric-density admission and atom ownership |
| Scalar XC primitives, scaled exchange, vacuum continuation and mixed response | `xc.point_ad`, the actual admitted primal/implicit residual | Point-model branches, channel transport and native point ABI |
| DFT/ECP quadrature and native/Python Becke responses | `integral.quadrature`, `xc.grid_response_ir`, `xc.atomic_grid_native` | Newton iteration, clamp/admission, sparse reductions and bounded partition statistics |
| VV10/rVV10 local scales, pair cotangents and row composition | `method.nonlocal_pair_ad`, common scalar Graph | Pair admission, FP32/FP64 policy, tiling and reductions |
| GFN origin translation, cached radial forces, onsite/spin/H0 potentials and COSX local products | Actual TensorIR/GeometryIR primal AD, with shared scalar lowering where needed | Sparse gathers/scatters, ragged work, bounded solves and lifetime policy |
| Existing orbital/implicit response, CC and periodic representation work | Existing differentiated TensorIR/GeometryIR programs and declared implicit linear systems | Solver policy, contractions of supplied partials, translations and incidence maps |

The shared AD engines necessarily contain primitive differentiation rules.
Special-function moments retain their independently qualified primitive
mathematical contracts. Linear JVP/VJP transport, fixed-coordinate seeds,
contractions, inverse/implicit solves and reductions are not second, manually
supplied nonlinear derivative programs. Reference/oracle implementations stay
explicitly test-only or part of an already declared diagnostic contract.

## Late discoveries and rejected approaches

The final census found a native CPU Becke derivative independent of the Python
path, and cached Cartesian norm projections in D3/D4/gCP/GFN2-D4. These now use
actual norm/partition AD. Cached norm scatter cancels a Cartesian factor
symbolically, never by dividing by a possibly zero component at execution.
Native Becke tests cover all four grid versions, one/three/five smoothing
iterations, displaced-energy finite differences and translation invariance.

The first nested-Jet mixed XC emission regressed complete CPU CPKS. Scalar sign
cleanup, power-exponent alignment and forced host noinline did not repair it.
The retained emitter differentiates one scalar DAG for value, first,
directional and mixed lanes, sharing nonlinear intermediates across channels.
Its implicit quotient schedule is reconstructed from the differentiated graph,
not a supplied quotient-rule equation. Extreme-tail and independent Libxc
tests remain required.

UKS CPKS then exposed extra overhead in the dynamic CPU integral value path.
Early-return variants and eagerly constructed derivative invariant frontiers
were measured and rejected. The retained adapter evaluates the generated
primal once for its result and constructs a generated tangent schedule only
inside the nonempty-channel branch. Native traversal and channel counts do not
change. Do not reintroduce per-operation zero-channel return paths merely to
avoid constructing a small mathematical schedule.

Random MO-coordinate right-hand sides are not a gauge-invariant cross-version
oracle near degenerate virtual spaces. Final CPKS qualification projects the
same fixed AO fields into each converged orbital frame, retains the ordinary
true-residual checks and compares the solved AO density responses. It does not
loosen the solver gates or compare unrelated perturbations as equal physics.

## Qualification and limits

The final campaign is based on master
`97c34eeb0eb1a0fdabfb24b5436f667149701f32`. Earlier b40-based experiments remain
diagnostics, not relabelled measurements of that baseline. Native libraries,
external stationary DSOs, dirty source reconstruction and Python owners must
be distinguished in the retained receipts.

Before the final paired timing collection, host qualification passed all 122
CTest tests and 1194 Python tests plus three subtests. The host-only run skips
1601 opt-in GPU/unavailable-toolchain cases; it is not GPU evidence. Scheduled
n1 qualification independently passed 1653 tests without skips, including
the actual CUDA scalar consumers, libcint/Libxc response tests, reconverged
density response and pinned tblite/xTB energy/force comparisons. Native D4,
ragged 4100-reference, D3 and gCP gates are separate executed tests.

The first expanded host run lacked the pinned GFN1/D3 source cache; explicit
registry synchronization repaired that prerequisite without changing pins or
scientific tests. The first expanded CUDA response run lacked packaged
stationary artifacts. Six source-matched Python s/p AOT profiles were built
before rerunning those gates. Missing products were not replaced by CPU
fallback or a relaxed admission guard.

Real-device tests and timing use finite Slurm allocations on n1/node1 and
n5/node5, never the maintained n3 host. Scheduler-assigned GPU visibility is
preserved. n5 uses the same private n1 glibc loader/runtime for both libraries;
no system runtime is modified. Its cluster does not implement Slurm CPU
binding, so the final KS/classical runs explicitly pin the shell to a CPU
inside its inherited allowed affinity. Earlier unpinned timing stays labelled
as such. Compilation uses verified sccache 0.16.0 without clearing shared caches
or stopping unrelated servers/jobs.

GPU integral AOT shells are disabled in this bounded-path qualification; the
six stationary s/p DSOs are enabled separately. This is not qualification of
every default integral AOT profile. Existing complete COSX diagnostics fail
identically on baseline and candidate with `native/Python grid plan mismatch`.
That inherited experimental resource-accounting failure is not a successful
complete COSX endpoint. Its guard is not weakened; independent compiled CUDA
COSX scalar summands pass their primal-only oracle.

Final paired endpoint results and the completion decision follow. No release/tag
publication is authorized by this cleanup or its PR.

## Final CPU endpoint evidence

The final source-matched CPU libraries are baseline `f12a9fd1f1b77350...` and
candidate `b6738cf4384d95bd...`; full hashes and build options are retained in
the publication. Every CPKS/force worker retains first use plus eight complete
samples, in two reversed-order process blocks, using matching old/new Python
trees rather than silently comparing both native libraries under new Python.

- Complete PBE RKS CPKS ratios are 0.787192 (original) and 0.787709 (displaced);
  UKS ratios are 0.957873 and 0.957205. Both process-block ratios improve for each
  cohort, with matching SCF/Fock and GMRES work and true-residual acceptance.
- Complete CPU E+F ratios are 0.976897–0.981167 for DF and 0.987369–0.992133 for
  direct PBE/PBE0 RKS/UKS. Constructor/preparation, cold/warm/displaced execution
  and resource cleanup are included; energy/force and work parity remain gated.
- Sixteen balanced randomized native-library-isolation pairs give nonlocal
  ratios 0.992799/0.996200 for 128-point VV10/rVV10 and 0.995860/0.998258 for
  PBE+VV10/rVV10 NaH-ECP E+F. These use current Python on both selections and
  are labelled as native-library isolation, not an old/new Python comparison.

The GPU timing library initially preceded the final CPU integral adapter change.
Rather than equate that binary with the final one because its GPU mathematics
did not change, the final rebuilt library is separately timed on n5 with the
same endpoint inventory. Superseded timing stays diagnostic, not relabelled
source-matched evidence.

## Final CUDA endpoint evidence and closure

n5 Slurm job 2020 reruns the complete GFN2/nonlocal/KS cohort with final rebuilt
candidate `08a7efc40ffddf44...` and baseline `00244f9e3e189d02...`. This is the
same final candidate used by the 1653-test n1 scientific run, not the earlier
`1aa1733d...` timing binary. Both selections use the same private runtime and
one inherited allowed CPU, with full library/DSO/runtime hashes retained.

- GFN2 total ratios span 0.988126–1.002572 across eight molecules. H2's ratio
  1.002572 has paired-bootstrap 95% interval [0.996357, 1.009418]; HF's
  1.000254 has [0.996388, 1.002646]. Do not call those small positive medians
  speedups or evidence for exact zero overhead.
- CUDA nonlocal total ratios span 0.999996–1.003495 across VV10/rVV10 at
  128/1024/4096 points; every interval includes parity. The largest positive
  median (4096-point VV10) has interval [0.938919, 1.008562]. Retain the
  wide larger-grid distributions, all timing samples and full semantic work.
- Complete CUDA CPKS ratios span 0.997844–0.999480; complete PBE/PBE0 RKS/UKS
  E+F ratios span 0.999073–1.000181. Reversed-order process blocks are
  descriptive, not IID paired confidence evidence. In particular, the second
  displaced-RKS CPKS block is 1.005052 despite its combined median 0.999480;
  no claim of uniformly faster individual blocks is made.
- Independent source-linked classical CUDA endpoints retain all seven samples
  for each cold/unchanged/changed row and both reversed-order blocks. Ratios
  span 0.836930–0.932171 for D3 BJ, 0.8439–0.9365 for D3 zero,
  0.809519–0.876669 for D4 EEQ and 0.6514–0.7358 for gCP. These gains do not
  compensate for a different workload's noise or substitute for its gate.

All 74 selected endpoint cohorts pass numerical and semantic-work replay. The
largest scaled error is 0.033859 of its declared tolerance, including separate
true-residual gates for the retained CPKS loads. Final data, exact reconstruction
of dirty measured source, historical execution recipes and checksum-authenticated
replay are retained under `benchmarks/results/derivative-ad-20261108/`.

The source retirement is complete within the ownership census. The measured
endpoint distributions show improvements or parity rather than a detected
systematic regression; finite noisy measurements cannot prove universally zero
overhead. Scientific numerical acceptance is recorded separately from formal
default/selector performance promotion, which was not requested or run. The
bounded GPU AOT scope and inherited complete-COSX diagnostic failure above
remain explicit limits, not unreported successful qualifications.

## References

- `2026-11-07-classical-scalar-ad.md`
- `2026-11-07-integral-ao-scalar-ad.md`
- `2026-11-07-scaled-xc-graph-ad.md`
- `2026-11-07-gfn2-origin-translation-ad.md`
- `2026-11-08-nonlocal-quadrature-ad.md`
- `2026-11-08-ao-response-and-local-ad.md`
- `benchmarks/results/derivative-ad-20261108/README.md`
