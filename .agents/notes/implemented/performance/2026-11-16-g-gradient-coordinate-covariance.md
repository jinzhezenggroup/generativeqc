# Decision: share g-shell x derivatives under y/z covariance

Status: implemented; complete manylinux wheel acceptance remains separate
Date: 2026-11-16

## Problem

The wheel failure in run `43732294063`, head `a278661ba`, is not a timeout.
After 14,642 seconds, NVCC reports that `ptxas` died from signal 9 while
compiling `src/scf/cuda/one_electron_derivatives.cu`, object 708/1122. The
retained container reports `OOMKilled=true` with a 12 GiB no-swap limit.
The concurrent one-electron values compilation subsequently completes.

The production maximum-g switch arrived in #3057 (`f590a214be`,
2026-11-12 UTC). It expands the default maximum-f derivative header from
7,647,704 to 43,685,629 bytes, about 5.7 times the generated source.
Both the SCF gradient owner and `cuda_cosx_derivative.cu` include it. An
isolated three-architecture compile retains a reaped-worker maximum RSS
of 14,256,840 KiB and fails its finite 3,600-second bound before completing
all architectures. Serializing two build jobs alone cannot make a compiler
whose individual working set exceeds the container limit fit.

## Decision

For g-bearing shell pairs only, choose one representative of each joint
y/z Cartesian-power orbit when emitting an x derivative. Swapping y and z
fixes the differentiated axis. Dispatch the other member to the retained
case after swapping both Gaussian powers, every Cartesian pair-geometry
field, and the external attraction center's y/z coordinates.

Retain the pre-rounded radial geometry and Boys values. This matches the
existing outer x/y and x/z covariance boundary rather than recomputing a
nominally equivalent radial quantity in another floating-point order.
Retained canonical case arithmetic is byte-preserved, but a coordinate
permutation can still change floating-point ordering: this is a
mathematical covariance optimization, not a bitwise-parity claim.

The default maximum-three header and all 32 lower-angular helper bodies
are byte-identical in a one-off migration proof. The 150 g-capable
mathematical programs, all component IDs, invalid-component NaN refusal,
center/sign conventions, weights, schedules and resource fallbacks remain.
These byte comparisons are evidence, not permanent source fingerprints.

## Complete-object and resource evidence

NVCC 12.9.86 and verified sccache 0.16.0 compile the unchanged complete
native gradient owner with Release flags, sm80/sm90/sm120 images and
compute80 PTX. The new request has eight stage misses and zero hits and
completes in 2,493.96 seconds. The original request has six completed
stage misses and zero hits before its timeout. Independent compiler
jobs overlap these observations; do not infer an uncontended speedup
percentage or whole-wheel duration from them.

| Metric | Original | Covariant g lowering |
| --- | ---: | ---: |
| Generated derivative bytes | 43,685,629 | 26,657,437 |
| Reaped-worker maximum RSS, KiB | 14,256,840 | 7,432,664 |
| Completed sm80 CUBIN bytes | 100,726,176 | 61,231,776 |
| Completed sm120 CUBIN bytes | 465,824,224 | 282,093,976 |

The original CUBINs are completed architecture images retained before
the full-owner timeout, not an invented completed three-architecture
object. The genuine completed baseline native sm120 owner's resources
also match that original sm120 image. The new complete owner retains
all three real images and 27,702,903 bytes of compute80 PTX.

For all five launch kernels on the two directly comparable architectures,
register usage remains 255, static shared storage and local storage are
unchanged. Stack requirements increase by 48--144 bytes on sm120 and
96--112 bytes on sm80. Do not equate smaller source or compiler memory
with universally identical occupancy or method performance. Reaped-worker
RSS is also not a simultaneous compiler-tree or cgroup peak; concurrent
large units and the complete wheel remain resource-acceptance gates.

## Independent numerical and execution gates

- Host execution of the actual complete emitted header covers all 25 s--g
  shell pairs, five asymmetric/coincident/near/translated/distant geometries,
  6,125 primitive Cartesian pairs and all 27 derivative channels. Independent
  PySCF/Libcint references pass at `atol=2e-9, rtol=2e-11`.
- The permanent cached host probe compiles three representative real shell
  helpers, not another whole 26 MB header. It checks every component of
  s/g, g/p and g/g against independent Libcint A/B S/T/V derivatives, seven
  geometries including both sides of Boys T=20, independent SciPy Boys
  inputs, and invalid IDs including each first out-of-range component.
- Slurm jobs 17493 and 17499 on authorized node1 run separate baseline/new
  DSOs linked to genuine complete native gradient owners. Each passes
  3,750 complete weighted-gradient updates: 750 independently assembled
  vectors, four explicit schedules and implicit schedule 3. Coverage includes
  dense/channel-isolated/inactive/zero work, nonsymmetric weights, channel
  scales, three nuclei, nonzero accumulators and arena guards. The full-sum
  oracle gate is `atol=3e-8, rtol=3e-10`. Maximum independent-oracle error is
  9.01e-14; maximum old/new difference is 1.56e-15. The oracle is independently
  checked by fixed-weight finite differences for s/s, f/f and g/g.
- Timed complete gradient updates include output clearing and all launches,
  with semantic AO-pair, primitive-product and nuclear-visit counts retained.
  Five post-warm samples of 32 updates cover s/s, f/f, g/s and g/g under all
  five schedules. Observed new/old medians range about 0.978--1.048. These
  sequential, unlocked-clock jobs do not establish a method speedup or a
  universal performance bound; default cooperative g/g is approximately
  unchanged in this observation.
- A clean complete sm120 Release library at `cc0e4dd9a` also builds and passes
  Slurm job 17498: RHF/UHF, Cartesian/spherical Direct-HF energy/force endpoints,
  each cold/warm/moved/restored, against independent converged PySCF. Maximum
  energy/force errors are 6.22e-14 Hartree and 8.05e-11 Hartree/Bohr. Endpoint
  timings include preparation separately and actual iteration/Fock counts.
  This is a frozen earlier baseline qualification, not a newly rebuilt
  whole-library qualification of this g lowering.
- The subsequent complete sm120 Release build at `10541ecd8`, containing
  both this lowering and the DF Lambda pool, also succeeds. Slurm job 17500
  repeats all 16 independent Direct-HF endpoints. Maximum energy/force errors
  are 6.22e-14 Hartree and 8.04e-11 Hartree/Bohr; every endpoint retains its
  baseline iteration and Fock-build counts. The complete shared library shrinks
  from 1,380,611,768 to 1,329,973,496 bytes. Its incremental rebuild is not a
  cold timing comparison: retain its 21 cache hits and 77 misses, rather than
  comparing its duration to the original full build.

The focused suites pass 357 tests, with 57 unchanged explicit-GPU skips.
The 604-module compiler ownership check passes. Complete native-library
qualification does not replace manylinux link/repair or installed-wheel gates.
Another 203 catalog, dependency, CI-budget and compiler-structure tests pass.
The pinned CI checker `ty==0.0.82` passes after explicitly unpacking the
two-element angular pair; the complete emitted header remains byte-identical
to the compiled and GPU-qualified artifact in a one-off preservation proof.
Official snapshot regeneration changes only current source identities and
28 changed-input lists. Historical measurements, summaries, qualification
claims and acceptance records are unchanged; retain the added source inputs.

## Rejected alternatives

Partitioning the original large switches into noinline leaves of at most
32 cases preserves all 2,450 original case bodies but slightly grows the
header to 43,708,119 bytes. It also times out at 3,600 seconds, retaining a
reaped-worker maximum RSS of 13,578,604 KiB; its sm80 assembler alone exceeds
12 GiB. It does not solve the observed container limit and is not adopted.
Do not borrow the successful covariant object's evidence for this rejected
partition-only lowering.

Exact shared-ancestor CSE across all cases, retaining their ordered roots,
only reduces the source census to 40,884,913 bytes (6.41%). This alternative
was not compiled or numerically qualified and is not adopted.

Do not drop g channels, architectures, compute80 PTX, independent numerical
tests or scientific methods to hide this compiler hotspot. Do not merely
increase timeouts for a demonstrated memory-limit failure.

## Evidence and reproduction

The frozen local investigation directory is
`/tmp/gqc-wheel-one-electron-partition/`. It retains source inventories,
complete-object commands/stats, original completed images, generated
case-preservation proofs, independent oracle inputs, native-owner/DSO
hashes, Slurm reports and complete-gradient timings. The complete baseline
build and Direct-HF reports live in `/tmp/gqc-wheel-native-endpoint/`;
the original `cc0e4dd9a` artifact/report are preserved in its `baseline/`.

Generate the current source with
`python tools/generate_one_electron_kernels.py --derivatives --max-angular-momentum 4 --output /tmp/generated_one_electron_derivatives.cuh`.
The compiled header SHA-256 is
`c32387f1067ead7a8463efb09e832ffc563d71afe68cf421f4785e57f759c047`.
Run `tests/python/test_one_electron_g_shared_values.py` with the required
verified host compiler cache; retain independent full-header and scheduled
device gates when changing this covariance boundary again.

## Revisit when

Consider additional symmetry sharing only with an explicit geometry/center
mapping proof, complete-object resource evidence and independent complete
gradient/endpoint gates. The remaining 7.1 GiB single-worker cost is not
small enough to assume arbitrary concurrent large compilations fit 12 GiB.
