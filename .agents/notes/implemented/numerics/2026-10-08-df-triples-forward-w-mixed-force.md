# Decision: expose an experimental mixed forward-W complete force endpoint

Status: implemented internal experiment; no public/default precision admission
Date: 2026-10-08

## Problem

The master baseline `4385f72751b829883407c01106186917c344317b` already binds the
compiler's qualified FP32 W contractions to standalone DF triples energy.
Complete molecular forces instead share strict W moments between the triangular
triples pullback and full-Fock resolvent response. Merely replacing the final
energy with a separately evaluated FP32 result would give mismatched E/F state
and duplicate staging/work, rather than testing a complete endpoint.

## Decision

Append an explicit `admitted_triples_w` internal control and a benchmark-only
`TRIPLES_W_FP32_0_OR_1` argument. The joint response reuses the existing
compiler-owned `WPlan`/`WExecution`, including its canonical operand casts,
prepared matrix tables, arithmetic identity and provider selection. Integral
panels remain FP64; their optional FP32 cached copies, T2 and ovoo casts live
in one admitted arena until the complete stream drain.

Energy, pullback seeds and full-Fock response use the same approximate W
moments. All reverse contractions, V algebra, denominator checks, Fock products,
CCSD, Lambda and orbital/nuclear response retain FP64. These derivatives are an
approximation to the original physical derivatives, not exact derivatives of
the discontinuous FP32 rounding operation. That distinction is part of the
internal API documentation and independent acceptance contract.

The additional W provider coexists with the strict reverse BLAS provider.
Charge both simultaneously, plus host binding objects and every cast/cache
buffer. If optional W storage cannot fit, retain the original strict page/panel
policy and publish an explicit resource-fallback flag. Do not reduce residency
just to fit an unqualified precision experiment. Scalar-tile fusion has a
separate optional-storage fallback; one byte below a fused request is not
necessarily below the actual unfused minimum.

## Invariants

- Public precision modes and method defaults remain unchanged.
- No CPU/PySCF mathematical fallback or reference amplitudes/orbitals enter the
  production execution chain.
- Preserve original numerical publication gates, denominator safety checks,
  source lifetimes and complete-budget admission before touching numeric inputs.
- Record actual arithmetic bits, FP32 calls, cast elements and the generated
  precision-schedule identity, rather than inferring execution from the request.
- Compare complete cold E+F calls with matched controls and semantic work;
  isolated SGEMM throughput is not an endpoint speedup.

## Rejected alternatives

- A handwritten float W kernel duplicates compiler-owned mathematics and
  precision semantics; reuse the already-audited compiler candidate instead.
- Evaluate mixed energy separately from strict forces: this fails to share the
  same approximate moments and measures redundant work rather than the intended
  complete force execution.
- Immediately lower reverse/Lambda/orbital arithmetic: these are distinct
  numerical decisions needing their own compiler schedules and independent
  residual/refinement qualification.
- Promote the experiment to AUTO/public FP64: small physical fixtures do not
  establish cancellation-heavy or near-degenerate safety.

## Evidence

Release CUDA 12.9, sm_120, portable_cuda AOT, compiler fast-compile disabled,
verified ccache 4.5.1 for C++ and CUDA, with retained before/after statistics.
Real execution uses finite Slurm allocations on n2 and preserves assigned
device visibility. The source is isolated from the original working branch.

Slurm job 2697: **90 passed, 8 skipped** across the complete mixed endpoint,
benchmark selectors, historical default policy, joint Fock/pullback response and
standalone W precision suites. Skips require optional provider/test hooks absent
from this qualification build. H2, water and LiH use independent same-Hamiltonian
PySCF 2.14.0 energies and physical directional energy finite differences at
`1e-4` and `3e-5` Bohr; no reference state is supplied to the native solver.
Energy-only and energy+force calls agree at `1e-12` Eh. Joint tests exercise
fusion on/off, bitwise strict-budget fallback and refusal one byte below the
unfused strict minimum without partial publication.

Across those small fixtures, maximum independent energy error is
`7.461e-12` Eh, maximum mixed-vs-strict force error is `6.153e-12` Eh/Bohr,
and maximum independent directional-FD force error is `7.312e-9` Eh/Bohr.
The retained gates are `1e-8` Eh and `3e-7` Eh/Bohr. Slurm job 2695 also runs
the complete mixed water endpoint under compute-sanitizer memcheck: zero errors.

### Matched large endpoint

Slurm job 2694 measures two fresh processes on the same RTX PRO 6000 Blackwell
GPU (`GPU-54595246-dbdc-a633-dc38-7bd8eea3831a`, 600 W): spherical ethane230,
aug-cc-pVTZ / aug-cc-pVTZ-RI, 230 AOs, 488 auxiliaries, 9 occupied / 221 virtual
orbitals, 64 GiB complete correlation budget. RHF remains conventional exact
native CUDA; only correlation uses DF. Both runs use identical selectors,
including DIIS history 8, Q batches 8, packed DIIS, no screening/preconditioning
or recycling, omitted epsilon cotangents and unfused scalar response.

| Complete phase | Strict FP64 (s) | Mixed forward W (s) |
| --- | ---: | ---: |
| Native cold E+F endpoint | 671.481674 | 664.223592 |
| Complete process wall | 671.79 | 664.55 |
| (T) energy + pullback + full-Fock response | 39.332587 | 31.589527 |
| Corrected Lambda | 274.267804 | 274.305019 |

The observed triples reduction is 19.69% (1.245x); complete native time is
1.08% lower (1.011x). **This is one process per mode, not a statistically
qualified endpoint speedup.** Compilation/cache and OS page cache are not
cleared; each native solver state is fresh. Lambda remains the largest phase.

All measured semantic work matches: RHF 19 iterations; CCSD 20 iterations /
38 evaluations; Lambda 21 iterations / 42 actions; Z 12 iterations / 13 actions.
Triples and Fock response retain respectively `9,224,865,365,040` and
`9,924,048,505,176` contraction summands. The candidate executes 4,860 FP32
GEMMs and records `56,315,690,067` logical cast elements; this is not a measured
memory-transaction count. Complete reported numeric capacity stays
`7,107,919,137` bytes because another phase sets the endpoint peak; this is not
sampled total GPU VRAM usage.

Mixed total energy is `-79.71851664322114` Eh, with `2.619e-11` Eh error versus
the retained independent same-Hamiltonian PySCF energy. All 24 force components
differ from the matched strict run by at most `2.504e-11` Eh/Bohr. Retained
independent energy finite differences cover C0-z and H1-x at two steps, not all
coordinates; maximum error is `3.099e-8` Eh/Bohr, below the unchanged `3e-7`
gate. Lambda residual is `6.115e-13`, Z residual `1.357e-13`, and stationarity
`7.673e-12`. No tolerance relaxation or iteration/work elimination causes the
observed timing difference.

Bulk compilation adds 468 cache hits and 5 misses, with no additional cache
errors. The completed library SHA256 is
`57cab37caa474774279726027e0f0903c9cf464be768d4be367d9eb94ce1909a`;
the benchmark SHA256 is
`253135273fda9c9630c75f3ddd37ce49c0192480a7cc4aaa79955cdc58b47c78`.
Production source digests match between the local patch and remote build.

Ignored local evidence lives in `.artifacts/mixed-qualification/`, including
scripts, raw endpoint JSON, finite-difference accuracy records, compiler-cache
statistics, allocation/binary identities and sanitizer output. The corresponding
remote compute root is
`n2:/data/jzzeng/qc-ccsdt-mixed-master-20261008-4385f727`.

## Consequences and revisit conditions

This adds a reproducible candidate, not a qualified default or promised complete
endpoint speedup. W is only one component of CCSD(T) E+F; corrected Lambda and
reference/orbital response can dominate complete timing. Revisit broader
precision admission only with repeated matched endpoint measurements,
near-degenerate/small-denominator and cancellation-heavy cases, independently
checked complete forces, and a strict refinement/fallback policy for any
additional response region.

## References

- `src/cc/df_triples_cuda.cu`, `src/methods/df_ccsdt_force.cu`.
- `docs/developer/df_ccsdt_gradient.md`.
- `tests/python/test_df_triples_mixed_endpoint.py`.
- `tests/python/test_df_occupied_triples_fock.py`.
- The earlier compiler-candidate rationale:
  `.agents/notes/implemented/numerics/2026-10-04-df-triples-mixed-precision-candidate.md`.
  This note extends the runtime/force boundary; it does not promote public precision.
