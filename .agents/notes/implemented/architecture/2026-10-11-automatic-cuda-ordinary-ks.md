# Decision: automatic semilocal CUDA ordinary-stream KS

Status: implemented ordinary energy endpoint; broader product qualification incomplete
Date: 2026-10-11

## Problem

The static automatic point bridge already reaches native fixed-density CUDA
XC, but public MethodIR/KS resolution and the ordinary KS owner still reject
automatic programs. Reusing a curated transport family would incorrectly
inherit its domain, scheduling and derivative capabilities. The legacy resource
query has only an LDA/PBE jet flag and cannot represent tau feature width.

## Decision

- CPU/CUDA resolve the same pure-semilocal MethodIR and J-only plan by structural
  capability and the existing explicit blacklist. Backend resolution is not an
  installed-artifact availability claim. Native preparation requires a static
  CUDA provider and compares its canonical expression, ingredients and work
  domain with the CPU registration. Missing providers report the opt-in build
  option, never a CPU/scalar/reference fallback.
- `CudaKsPlan` retains the static provider, uses its domain in diagnostics/final
  state, and admits only strict-FP64 device-fused all-electron pure-semilocal
  execution. Curated device chunks, graph replay, incremental J, host-unfused
  science, mixed precision, range/exact exchange and nonlocal composition are
  not inherited. Existing ordinary per-spin occupation, final audit and warm
  lifecycle owners remain authoritative.
- Shared detached-state validation authenticates the installed automatic CUDA
  provider against CPU code/science/ingredient/domain metadata, not merely a
  CUDA ordinal. Automatic RKS also enters the bounded physical F[D] closure:
  a DIIS-converged density alone can fail the detached physical-density gate.
  Both the native owner and public inventory retain four extra history slots;
  no convergence or canonicality threshold is weakened.
- Private resource inventory v2 uses the exact functional code and the native
  XC layout owner, including tau. V1 remains unchanged. Automatic requests are
  energy-only; they neither claim force storage/capability nor substitute a
  same-family LDA/PBE shape.
- Automatic Direct-J preparation retains no unused derivative source. Curated
  contexts keep their existing derivative reservation; the automatic energy-only
  owner must match its code-scoped Direct-J inventory rather than spend a force
  reservation it does not advertise.
- Explicit MethodIR/composition inputs derive automatic capability from their
  actual semantic plan, not the selector spelling. Native snapshots likewise
  use that plan's code/domain, preserving dynamic display/identifier choices
  without granting curated forces or misidentifying automatic state.

## Independent gates

The public CUDA suite covers the shared 3/5/3 LDA/GGA/tau-MGGA subset, both
spins and selector/explicit-MethodIR forms. Each runs cold, ordinary warm
recomputation and changed geometry. Pinned PySCF 2.14.0 / Libxc 7.0.0 compares
SCF energy/density and fixed-density full energy/Fock on the exported exact
grid. Source/binary-bound diagnostic receipts retain complete endpoint times,
ordinary physical J/XC counts, no unused K, no implicit iteration matrix
download, and stale snapshot invalidation. These are not graph-replay,
full-catalog molecular, force, installed-bundle or performance-win receipts.

Integer-occupation SCF density matrices make an arbitrary linear density
stencil leave the PSD domain. The independent Fock-direction gate therefore
uses a congruence curve `(I + h A) D (I + h A)^T`, with exact derivative
`A D + D A`. It preserves PSD without weakening numerical gates.
A separate CPU probe of the oracle/transport fixture agrees with native energy
to 2.23e-14 and directions to 1.43e-11; this is not CUDA qualification and did
not validate the complete H2 TPSS Fock matrix.

### Fixture and oracle pitfalls

The complete molecular suite uses asymmetric H4/STO-3G, with two occupied
spatial RKS orbitals and UKS alpha/beta occupations 2/1. An explicit Hcore
initializer isolates SCF matrix traffic. The public default MINAO initializer
borrows the CUDA eigensolver and legitimately downloads setup eigenvectors;
those transfers are not iteration-time host XC staging. Empty-spin and worker
boundary E/vxc gates remain in the separate fixed-density suite.

The previous H2 rank-one fixture hits the TPSS `min(tau_W/tau, 1)` boundary.
Independently rounded AO products choose different branch derivatives in
unoccupied directions. A Slurm probe reproduces full-Fock disagreement on
**both** native CPU (1.70e-4 RKS) and CUDA (2.34e-4 RKS), despite matching
energies and PSD-tangent directions. On the H4 interior fixture, the complete
TPSS Fock agrees with independent Libxc to 2.05e-14 on CUDA in both spins.
Do not cite tangent/energy agreement as complete rank-one/empty-spin TPSS Fock
qualification or replace the full-matrix gate with an occupied-space projection.
The test fixture avoids this nonsmooth comparison; the boundary remains
distinct evidence, not a silently resolved scientific discrepancy.

The independent oracle computes its own Hcore seed; it never seeds from the
native density. Pure LTA can stall a 200-cycle PySCF MINAO/DIIS oracle even when
the native fixed-density full Fock matches to 3.16e-14. The changed-geometry H4
case also exhausts a 200-cycle Hcore/DIIS and Newton oracle. A separate bounded
1000-cycle Hcore/DIIS reference converges with unchanged energy/gradient targets
and density agreement to 1.08e-8. The suite uses that finite reference budget
for every method, disables density-based grid pruning and authenticates exact
grid bytes. Native production remains bounded to 200 iterations plus its four
closure corrections. No endpoint numerical threshold or convergence flag is
bypassed to turn oracle failure into a passing receipt.

The ordinary batch wrapper previously exposed native context detail only for
correlated methods, losing automatic build/policy blockers as a generic
invalid-argument status. Automatic requests now retain their actual preparation
detail too. Negative real-device policies check those actionable blockers;
the remedy is not to weaken the test to accept a status-only error.

## Evidence and source lifetimes

- The final focused host resource/API regression passes 238 cases with seven
  existing real-device skips. The separate actual-C++ model/resource probe
  passes 71 cases, including 52 backend/science/domain/composition metadata
  mutations. These overlap and must not be summed as unique coverage.
  Compiler structure has 518 modules and no dependency errors.
- Full native CUDA compilation completed on n5 using verified sccache 0.16.0,
  NVCC 12.9.86, Release/sm_120. Byte-changed inputs were explicitly invalidated
  before rebuild, including the factory, shared final-state validator and CUDA
  closure owner; archive timestamps alone previously hid stale objects.
- Source-frozen v12 runs in Slurm job 7511 on node1/main/RTX5090, preserving
  scheduler-assigned device visibility. All 47 public CUDA KS cases, all 100
  fixed-density CUDA XC cases, curated native XC, and explicit TPSS/UKS/MethodIR
  memcheck pass; Compute Sanitizer reports zero errors. The runner still exits
  nonzero because of the separately controlled curated KS failure below.
  It retains 44 KS receipts / 132 lifecycle rows and 44 fixed-density receipts.
  Source and artifact manifests are verified before and after the run.
  The library SHA256 is
  `d97ab7cb5c90a04bcfa1f71ae2161ceb6f4d914c2010595497b646bebb9089d4`.
- Across those 132 rows, maximum absolute SCF energy, fixed-density energy,
  density, full-Fock, and direction errors are respectively 2.63e-14,
  2.80e-14, 1.08e-8, 3.16e-14, and 1.01e-10. Cold/warm-recompute/changed-geometry
  work counts are 530/106/602 Fock, J, and XC builds, with no unused K.
  Timed endpoint sums are 23.471/18.037/27.480 seconds respectively;
  configuration/resource estimation and preparation add 55.605 and 48.413
  seconds. The complete endpoint aggregate is 173.003 seconds, excluding the
  independent reference and fixture/receipt work. These are qualification
  measurements, not a speedup claim or isolated-kernel benchmark.
- v12 freezes prerequisite head `6eb9026e0b9ceed384d1467c6b53764e03ece06e`
  plus the ordinary KS patch. The final parent update to `5817dc51acc218c2adecf12470db999468865c0c`
  changes only `test_generated_registry_producers.py`, not native/compiler
  science; final evidence-note edits likewise do not relabel the measured
  source archive or binaries. Raw source/build/log/receipt evidence remains in
  ignored `.cache/1118-runtime/ks-v12/` and the matching n1 staging directory.
- The monolithic curated KS driver fails its shared final-model identity gate,
  including with the ordinary chunk/replay opt-outs. This failure reproduces
  with the earlier v3 library and with a matched native-consumer comparator
  restoring **all five modified native files** to prerequisite head
  `6eb9026e0b9ceed384d1467c6b53764e03ece06e`, retaining the same shared compiler
  and point inputs. Comparator library SHA256 is
  `d50d2478ec1413ab0bbc12dc860e91171aab13837563ccf408747c22408eb5f6`.
  This is a native-consumer control, not a complete historical checkout build.
  Both full and ordinary comparator invocations fail at the corresponding
  same sites. Do not hide that baseline defect, claim the whole curated KS
  driver passed, or weaken its identity gate as part of this automatic patch.

## Same-input kinetic-boundary correction

Subsequent independent diagnosis established that the retained TPSS discrepancy
cannot be dismissed solely as different rounded AO inputs or an arbitrary
nonsmooth subgradient convention. An exact dyadic, same-input physical rank-one
state gives the normalized `tau_W/tau` predicate one ulp above one; the selected
constant branch loses a valid inward PSD derivative. The full unoccupied Fock
entry is zero instead of the independently verified `-3.3455150091360384`.
Both trace-changing and fixed-electron inward PSD directions expose the error.

The shared canonical repair in PR #2279 is composed into this endpoint. It
changes only the exact kinetic predicate, retains raw branch derivatives and
work floors, and protects subnormal products. The independent native CPU tests
cover the exact full coefficients, the two valid directions, and empty/floored
spin controls; the full imported E/vxc/fxc fixtures and CPU/CUDA mathematical
identity checks also pass. The
[physical kinetic predicate decision](../numerics/2026-10-11-physical-kinetic-predicates.md)
records the proof and the rejected whole-ratio rewrite that introduced new
minority-spin failures.

The source-bound v12 GPU receipts above remain measurements of their original
source. They are not relabeled as GPU measurements of the corrected expression,
and the original H2 device full-Fock rows have not been rerun here. The retained
aggregate-driver exit 1 and matching native-consumer control likewise remain
historical failures. This correction resolves the demonstrated shared-source
predicate defect without claiming a new complete GPU campaign.

## Remaining boundary

Do not close remaining #1118 lanes or promote forces from this source patch.
The implemented ordinary energy endpoint has real-device endpoint, negative
policy and memory gates; the monolithic curated KS baseline remains a disclosed
failure rather than a passing regression claim. Rank-one/empty-spin TPSS
full-Fock boundary qualification also remains separate from the interior suite.
Bounded installed bundles, executable alias reuse, persistent dependency/cache
closure and the combined product/backend/domain evidence catalog remain separate
work. Keep the existing automatic AOT option default OFF.

## Rejected shortcuts

Do not infer the functional/domain from a native transport family, use a
positive method-name whitelist, reduce MGGA to a GGA capacity estimate, reuse a
stale snapshot across a solve/geometry change, or count a warm solve as a graph
replay. Do not hide compilation economics by disabling caches or changing
scientific/artifact identities to obtain hits.

## References

- #1118, #1119, #1121, #1123, #1124; prerequisite bridge PR #2261.
- `tests/python/test_libxc_automatic_cuda_ks.py`.
- `tests/python/test_libxc_cuda_ks_resources.py`.
- `2026-10-11-automatic-cuda-point-provider.md` in implemented architecture notes.
