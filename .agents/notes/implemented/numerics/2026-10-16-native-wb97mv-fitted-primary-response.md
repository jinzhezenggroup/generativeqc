# Decision: native compiler-AD full-rank fitted-primary response

Status: implemented (algebra component; molecular endpoint incomplete)
Date: 2026-10-16

## Problem

The Python WB97M-V Hessian bridge already differentiates the physical full-rank
fitted J/K energy, including the raw-metric inverse. Native nuclear response
cannot freeze that metric, substitute exact primary J/K, or borrow an occupied
projection that rejects signed density response. The incumbent source equations
also contain multioperand contractions whose literal unplanned execution would
amplify work despite bounded intermediate storage.

## Decision

`method/wb97mv_df_response.py` composes the existing stationary DF programs and
the incumbent whitening-orientation map, without creating a second scientific
algebra. Generic compiler `prepare_for_backend` contraction reassociation runs
before both affine CUDA emission and work accounting. The native owner executes
the resulting build-time maps without Python, JIT or host numerical contraction.
Reassociation changes reduction order deliberately, not the method or FP64
precision policy. The older induced-Fock consumer is not silently rescheduled.

Reference maps supply energy, spin Fock, raw three-center weights and raw metric
weights. Response maps supply frozen-density Fock motion and moving source
weights. The latter retain right dD, dA and the compiler implicit inverse JVP;
the former excludes dD to avoid double counting CPKS relaxation. The physical
inverse is W^T W for the actual out-auxiliary/source-auxiliary whitening layout.
It does not require a symmetric whitening gauge.

Preparation retains immutable inverse/reference results in a charged external
arena. Every action stages seven outputs privately, validates device finite and
symmetry domains, and reauthenticates before publication. Inputs may be signed
and indefinite but must be symmetric in physical matrix indices. Arena/output
aliasing, foreign streams/devices, capture, stale owners and late revocation
fail without publishing numerical results. CUDA faults during the final copies
remain explicitly outside an all-or-nothing publication guarantee.

The actual KS adapter borrows accepted D, raw integral source, W and a new
read-only metric-domain lease from the real full-rank DF owner. The separate
lease avoids changing the existing value-source capability/equality contract.
No eigenvalue or basis numerical data is reconstructed on the host. Retained
raw public A is generated once on the producer stream, and charged with all
algebra storage under one byte bound. Mutable J/K scratch and occupied projection
leases remain untouched. Cutoff separation is checked on CUDA against the
existing Hessian domain: lambda_min > max(1.1 cutoff,64 eps) lambda_max. A forward
full-rank factor alone does not authorize second response near the cutoff.

## Rejected alternatives

- Rewriting/factoring the DF equations by hand: the generic compiler already
  owns contraction reassociation and can schedule the original AD graphs.
- Emitting the incumbent graph without preparation: materialization is storage
  bounded but multioperand summation work is not acceptably bounded.
- Differentiating a whitening gauge or freezing V^-1: neither is the physical
  full-rank raw-metric source contract used by the stationary envelope.
- Pseudoinverse/rank-truncated differentiation or Direct fallback: changes the
  scientific contract and hides a missing spectral second-response rule.
- Borrowing occupied factors or primary J/K scratch: breaks signed response,
  lifetime isolation and accepted-state projection capabilities.

## Evidence

Independent compiler tests compare energy/Fock/source weights against expanded
NumPy fitted ERIs, their joint energy derivative, moving source-weight finite
differences and the distinct fixed-D Fock derivative. A nonsymmetric whitening
gauge distinguishes W^T W from W W^T. Behavioral schedule checks bound every
emitted node, including primal validation dependencies, rather than pinning
source spellings or hashes. Build-only generation rejects public runtime,
CuPy, Torch and PySCF imports and is deterministic across repeated generation.

Final integrated real-device qualification: Slurm 3127, node2/main/PRO6000,
assigned visibility 0; 27 tests passed after preserving concurrent compiler
signed-zero repair `9b83bd93e`. An earlier Slurm 3125 run passed 16 native
tests; Slurm 3126 passed 27 tests and 24 sanitizer runs before that integration.
Signed RKS/UKS synthetic shapes include
(N,A)=(2,3),(5,7),(9,13). Actual RKS/UKS DF sources agree with independent
libcint/NumPy primal and directional oracles; Direct owners are rejected.
Standalone subprocesses execute with PATH/PYTHONHOME/PYTHONPATH unavailable.
Final qualification includes 28 clean sanitizer runs: memcheck/initcheck/
racecheck/synccheck for two standalone shapes and actual RKS/UKS DF and Direct
rejection modes, plus memcheck/initcheck on nonfinite-whitening/eigenvalue
preparation failures. Racecheck reports zero errors and warnings. Actual KS
admission checks the exact total arena bound and rejects one byte less, including
retained raw A. Host-only admission checks extent/product overflow and byte/work
limits without initializing CUDA. The additional structure/AD regression suite
passes 234 tests; the concurrent inline-add/Fock regression passes 20 tests with
four optional GPU skips locally. Pinned ty 0.0.82 and all-files prek pass.

Standalone scheduled-work receipts (not endpoint timing or FLOPs):

| Spin / N / A | Arena bytes | Reference terms | Terms per action |
| --- | ---: | ---: | ---: |
| RKS / 2 / 3 | 9,728 | 684 | 1,775 |
| UKS / 5 / 7 | 91,648 | 24,837 | 58,706 |

These standalone arena sizes exclude borrowed inputs; the actual KS adapter
separately charges retained raw A. Source weights use full ordered public A/V
arrays, not an implicitly half-weighted triangular derivative ABI. A future
raw-integral consumer must preserve these multiplicities. Raw qualification logs, source-matched native
binaries, dependency records and cache statistics are retained under
`.artifacts/wb97mv-hessian-aot-20261015/df-*` (ignored local evidence).

## Consequences and remaining gates

This is a fitted-primary algebra source, not native raw nuclear integral AD,
stationary HVP/Hessian orchestration or a production CLI replacement. Synthetic
shapes are not numerical qualification of nontrivial spherical d/f/g molecular
transforms. Actual owner fixtures currently use s orbitals/auxiliaries. Raw
three-center/metric derivative streaming, auxiliary-g closure, residual ECP,
complete high-angular integral qualification, stationary assembly and removal
of the production Python/CuPy Hessian worker remain required. No full native
endpoint speed or GPU4PySCF comparison is established by these component tests.

## Revisit when

A qualified bounded GEMM/liveness schedule can replace the materializing affine
schedule while preserving complete consumers, numerical tolerances and literal
work/traffic evidence, or a separately qualified rank-truncated physical second
response is added. Do not reinterpret this full-rank rule as such a fallback.

## References

- `docs/developer/dft_hessian.md`
- `.agents/notes/implemented/numerics/2026-10-16-native-wb97mv-nuclear-jk-source.md`
- `python/generativeqc/df_hessian.py` (independent incumbent runtime contract)
- PR #2527
