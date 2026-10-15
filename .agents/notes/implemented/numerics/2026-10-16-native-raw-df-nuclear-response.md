# Decision: Native two-pass raw DF nuclear response

Status: implemented
Date: 2026-10-16

## Problem

The native fitted-primary algebra accepts raw three-center/metric geometry
directions and returns their fixed/moving cotangents. Removing its Python
integral consumer requires the actual normalized orbital and auxiliary bases,
including independent public Cartesian/spherical representations. The ordinary
orbital AO ABI cannot carry the six-term real-spherical auxiliary-g expansion.
Nuclear response must also retain every mathematical-center incidence, even
when multiple centers refer to the same physical atom.

## Decision

Borrow a new immutable, single-system DF basis capability from the actual CUDA
integral source. It exposes normalized radial and Cartesian component arrays,
integer shell packing and the producer's original public expansions, not the
full mutable DeviceBatch, J/K scratch, occupied factors or reconstructed host
numerics. Combined primitive/component packing is orbital, auxiliary, dummy;
public Cartesian indices remain basis-local. Counts exclude the dummy, while
alias protection includes its tail cells. Apply radial/component normalization
and public projections exactly once. The enclosing accepted-state owner must
authenticate the complete capability; a collection of matching pointer labels
alone is not lifetime proof.

Reuse existing compiler DF value/first/gradient-JVP leaves and the existing
compiler fixed-projection weighted-envelope AD. Admit orbital angular orders
through f and auxiliary orders through g separately. Do not add recurrences or
weighted derivative equations in native code. A native consumer accepts only a
complete, nonoverlapping linked AOT closure for each actual ordered shell class.
Missing closure fails before numerical work, without JIT or Direct substitution.

Use two explicitly counted source passes around method-owned inverse/weight AD
and the eventual CPKS consumer. The value pass supplies raw public A,V and their
geometry directions. The contraction pass supplies w I' and w I''R + w_R I'.
The caller's moving cotangents retain right density motion. Full ordered A and V
have no hidden triangular factors. The actual RKS/UKS adapter binds normalized
orbital/auxiliary topology and the complete fitted WB97M-V model to the canonical
accepted state and original stream, with an independent charged arena.

A mandatory bounded shell cache evaluates every primitive Cartesian component
once per pass, before public expansion gathers. It retains only that shell's
value, mathematical-center gradient and gradient-JVP. Deterministic axis-owned
scatter sums all mathematical-center incidences, not a left-coordinate source
traversal. No global coordinate-major derivative or fitted four-center tensor
is formed. This is bounded source work, not yet an optimized Hessian endpoint.

Work admission and receipts count primitive-component evaluations, radial
records repeated for each component tile, sparse coordinate projection bounds,
primitive-field summands, cache additions, projection sums and physical-center
scatter bounds. Radial tiling changes cache additions but not mathematical
primitive-component evaluations. A reduction bound is intentionally conservative
for either pass, not an instruction/FLOP count. Failed/revoked actions can submit
work without completing a pass.

All actions use caller/private staging and reauthenticate before publication.
Numerical/admission/revocation errors preserve published outputs; a CUDA fault
during publication may leave partial copies. Every submitted primitive record
must be initialized even after an earlier status failure: linked AOT leaves
report errors but do not consume an abort flag. Skipping those writes would let
a subsequent leaf read uninitialized records on a rejected path.

## Rejected alternatives

- Reuse the three-term orbital-f projection ABI for auxiliary g: truncates the
  six-term g m=0 column and silently corrupts values and nuclear derivatives.
- Reconstruct normalized basis numbers on CPU: breaks actual-source ownership
  and risks applying normalization/projection twice or in the wrong index space.
- Rerun primitive leaves for each public expansion or physical left coordinate:
  bounded scratch would conceal angular/coordinate work amplification.
- Retain all molecular derivative tensors: unnecessary coordinate-major storage
  when final raw cotangents are available for the second source pass.
- Freeze the metric, infer a triangular factor, or recover coincident centers
  by translational symmetry: changes the full ordered compiler-AD contract.
- Treat generation admission as full f/f/g qualification: the targeted linked
  inventory deliberately covers smaller, independently testable domains first.

## Evidence and scope

The independent oracle uses Cartesian libcint A,V and analytic first center
partials, independently transforms each basis with PySCF Cartesian-to-spherical
coefficients, and normalizes public columns with independent overlap diagonals.
It retains both orbital and metric incidences. Displaced weighted energies
validate the oracle's first derivative; two finite-difference steps of independent
analytic moving-weight gradients validate HVPs. Finite differences are test-only.

Raw tests cover contracted signed radial coefficients, ordered orbital s/p and
d/f classes with s auxiliary, and s orbital with two separated auxiliary g
centers (including noncoincident g/g metric derivatives). Orbital/auxiliary
Cartesian/spherical choices are independent, across radial tiles 1,7,64.
Actual accepted RKS/UKS fixtures compose raw nuclear directions with native
full-rank inverse/Fock/moving-weight AD and then contract native raw derivatives.
The right dD is an arbitrary symmetric test direction, not a stationary CPKS
solution. Re-solving the real KS owner revokes both source passes without
publication. Direct models and resource-short owners fail closed.

The targeted qualification inventory has complete selected classes, not full
production f/f/g closure: three-center {0,1}^3, (2,2,0), (2,3,0), (3,2,0),
(3,3,0), (0,0,4); metric {0,1}^2 and (4,4). Tiles contain at most eight
components. It preserves the earlier through-f one-electron and s/p exact
full/long/short-range closure. Admission of larger auxiliary roles is separately
tested without pretending those larger generated scalar graphs were all run.

Production CLI Hessian/HVP still uses the Python/CuPy worker. Residual ECP,
complete high-angular closure, stationary CPKS/Hessian assembly and full
Python-free endpoint qualification remain outstanding. Component receipts and
test durations are not GPU4PySCF endpoint speed comparisons.

Final integrated qualification preserves concurrent affine-add repair
`81219bc2e` and formatting commit `507c6efeb`. Slurm 3133 on node2/main/PRO6000,
assigned device visibility 0, passes 58 raw-source/fitted-algebra/actual-owner
tests and 36 sanitizer runs. Memcheck/initcheck/racecheck/synccheck cover
contracted Cartesian s/p, spherical orbital d/f, Cartesian orbital with
spherical auxiliary g, actual RKS/UKS source/algebra chains and Direct rejection.
Memcheck/initcheck additionally cover all four malformed/capacity source cases.
All seven racecheck runs report zero errors and zero warnings. Native children
run with PATH/PYTHONHOME/PYTHONPATH unavailable; native binaries link no
libpython. The test harness and independent libcint oracle still use Python,
outside the native numerical consumer.

Independent acceptance gates are 3e-10 for raw values, 8e-10 for raw geometry
directions, 2e-9 for weighted nuclear gradients, and 2e-7 for finite-difference
HVP comparisons (with relative tolerances in the tests). Both displaced steps
3e-4 and 1e-4 are checked. Local integrated compiler/SCF/AD/lifetime regressions
pass 291 tests with 86 optional/oracle skips; two host-only admission CTests,
pinned ty 0.0.82, 392-file CUDA ownership and all-files prek pass. A narrow
source-file dependency permission exports the immutable basis lease without
allowing general DF source dependencies on KS consumers or mutable plan state.

Representative raw-source admission receipts, radial tile 7 (per source pass;
not complete molecular endpoint timings):

| Domain / public representation | Arena bytes | Primitive components | Submitted radial records | Reduction bound |
| --- | ---: | ---: | ---: | ---: |
| s/p Cartesian | 17,920 | 430 | 117 | 119,072 |
| d/f spherical, s auxiliary | 32,768 | 2,892 | 390 | 286,449 |
| s Cartesian, g auxiliary spherical | 61,440 | 3,150 | 411 | 399,246 |

Reproduction artifacts are retained locally under
`.artifacts/wb97mv-hessian-aot-20261015/df-nuclear-*`, including the selected AOT
inventory, build/cache commands and source-matched binary checksums. The final
remote stage is
`n2:/data/jzzeng/wb97mv-hessian-native-response-20261015/df-nuclear-integrated-20261016`.
The host reports an unrelated NVML userspace/kernel version mismatch; CUDA
execution and all sanitizer gates succeed. No GPU speed claim relies on NVML.

## Revisit when

An endpoint profile justifies retaining raw derivatives or fusing a consumer,
with explicit capacity/lifetime fallback and matched complete-endpoint evidence;
or the compiler can schedule direct public-projected contractions with the same
single primitive-component evaluation and complete independent derivative gates.

## References

- `src/integrals/cuda_df_nuclear_response.hpp`
- `src/dft/cuda_ks_wb97mv_df_nuclear.hpp`
- `tests/python/test_native_df_nuclear_response.py`
- `docs/developer/dft_hessian.md`
- `2026-10-16-native-wb97mv-fitted-primary-response.md`
