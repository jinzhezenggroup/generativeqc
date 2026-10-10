# Decision: retained Coulomb charges and bounded DF geometry admission

Status: implemented
Date: 2026-10-10

## Problem

The energy repairs in PR #2164 reduced the frozen 96-atom PBE0-DF energy
endpoint to 73.374362 s, but its complete force consumer still took
221.229604 s in the retained stage-two diagnostic (294.721926-s E/F endpoint).
The retained force baseline library is
`eebbe62a00def9603d3ca66cbb8105c8dbb7b41919f36db17768f381f9b499ac`.
This is an energy-repaired baseline, not a measured original-master force
population. The original frozen master was
`4e20f7a7bcfbff770fefd13fa2777f8ab309679c`.

A fresh Slurm cProfile baseline returned identical E/F and identified 74.543 s
in the native DF derivative provider and 142.368 s in geometry drains. Nsight
Systems attributed 134.204491 s to 4,608 cooperative geometry kernels. The
ordinary J response alone took 71.798 s: it repeatedly inverse-applied AO
factor panels and formed an exchange-style metric Gram despite zero exchange
coefficients. Its 100 fitted-panel projections perform 81,377,340,948,480
FLOPs and stage 87,691,100,160 unpacked bytes through 37,120 copies.

The geometry planner reserved the entire remaining consumer budget for every
DF provider. That was necessary for unknown providers but wasteful for the
known snapshot API: its DF response scratch has an independent resource
contract. Its additional consumer owns only paired one-electron work and
compact publication. Overreservation restricted geometry to 32 point workers,
prevented the phase cache, and selected the tiled cooperative Becke fallback.

## Decision

### J-only full-rank response

For the already validated resident whitened factor B=A X, with symmetric
X=M^-1/2, contract q=B:D once and then p=X q. Reuse the compiler's existing
Coulomb response kernels for bar_A=c D p and bar_M=-c p p^T/2. The compiler
owns the auxiliary-fast TensorIR charge equation and its direct-NN GEMM
lowering. Packed rows use the existing folded-density kernel: off-diagonal
entries read both physical density elements; diagonal entries are not doubled.

The three-center cotangents retain bounded dense panels and their established
strided consumer mapping. All arrays borrow the existing response workspace;
no complete raw/inverse-applied AO tensor or second B is allocated. Forward
metric gauge, derivative providers, center response, ordering and stream
lifetime are unchanged. Counters report one retained-factor pass per nonzero J
term, its pair/auxiliary elements, root/outer work and auxiliary panel count.

Admission requires the existing validated retained view, full metric rank,
BLAS algebra, and exactly zero exchange coefficients in every term. Borrowed
JK scratch, streamed occupied response, packed-response sinks, single-fitted
diagnostics and serial/scalar algebra retain their existing routes. The
diagnostic `GENERATIVEQC_DF_COULOMB_RESPONSE=auto|panels` defaults to auto;
panels selects the original bounded algorithm, and unknown values fail closed.

### Geometry resources

The known `NativeKsSnapshot` exposes a live-token-checked additional integral
reserve. The compiler bounds the paired provider's topology/host envelope,
both full FP64 D/W uploads even when resident weights avoid them, and compact
publication. This is deliberately conservative; it is not an estimate of the
independent DF response scratch. Unknown providers keep their entire previous
allowance, and tight budgets keep the bounded fallback. Reported additional
one-electron device usage must fit the admitted reserve.

For automatic known fitted grids with at least 48 atoms, prefer 256-point
tiles. A 512-point AO tile fits this endpoint but cannot also admit the Becke
phase cache; 256 points admits both point concurrency and the existing phased
math. Explicit tiles, other methods/providers, target admission and finite
grid work/submission windows remain authoritative. No grid point, partition
iteration, pruning setting or approximate amplitude threshold changes.

## Independent endpoint evidence

Three alternating fresh-process native/GPU4PySCF pairs run in one finite
15-minute Slurm allocation on node1, RTX 5090. Every real-device test,
profiling and ablation also runs through finite-time srun. Native Release
sm_120 compilation reuses the verified ccache 4.5.1 cache; launchers and cache
statistics are retained. The new native library SHA-256 is
`75f2a97a01f05b0d611f5c15763a478169f039d5a9c40217f7e14fe813047a70`.

The fixture is 32 waters / 96 atoms, spherical def2-SVP (768 AOs),
cc-pVDZ-JKFIT (3712 auxiliaries), FP64 PBE0-RKS, unpruned 48x16x32 quadrature
(2,359,296 points), energy tolerance 1e-12, and native density/reference
gradient tolerance 1e-10. GPU4PySCF is 1.8.1. Cold means fresh process,
owner and density with persistent disk caches retained. Timing includes
prepare through synchronized first public E/F, excluding imports, context,
native Calculator construction and post-result teardown.

| Engine | Complete E/F samples (s) | Median (s) | Force median (s) | Iterations / Focks |
| --- | --- | ---: | ---: | --- |
| Native | 99.025059, 99.337125, 99.442745 | 99.337125 | 25.480677 | 25 / 25 each |
| GPU4PySCF | 76.572512, 76.908620, 76.512974 | 76.572512 | 10.561966 | 37 / 38 each |

The dominant old force bottlenecks are removed, but native is still 29.73%
slower at the complete endpoint and its force phase remains 2.41x reference.
Do not normalize by Focks or substitute the earlier 84.631802-s reference
sample for these freshly paired measurements. The retained single stage-two
force trace is not a clean paired timing population.

All three native runs return all 288 force coordinates and pass independent
gates: maximum energy error 1.90994e-11 Eh, maximum force error 1.52610e-10
Eh/bohr, native physical residual 3.82286e-13. Gates remain 1e-8 Eh,
3e-7 Eh/bohr and residual <=1e-9.

The diagnostic run attributes J response to 0.820850 s and paired one-electron
plus J/K derivatives to 3.430203 s. Geometry drain is 16.802675 s. The J ledger
has one 1,096,138,752-element resident B pass, two 13,778,944-element
root/outer operations, ten derivative panels, no fitted AO projections and no
metric BLAS dots. Final occupied-U reuse still completes independently.
Geometry has 256 lanes, 9,216 phased batches and 39,755,392 phase-cache bytes.
All 2,359,296 XC points and the complete point/pair work remain present.
Reported additional consumer bounds are 399,228,960 device bytes within
536,870,912 and 192,229,472 host bytes within 268,435,456. Those bounds still
exclude the separately charged DF-provider response: they are not whole-force
allocation or transport measurements.

Single causal ablations preserve E/F and work settings: old J panels with new
geometry take 96.211714 s in force; retained J with the unknown-provider/legacy
geometry reserve takes 149.786629 s. These are diagnostic single samples, not
timing populations. A normalized-adjoint/zero-seed probe takes 25.211859 s:
one near-baseline sample is insufficient to promote that previously losing
experiment, so its production default remains off.

## Validation and retained limitations

The clean prospective PR tree passes 299 focused host compiler/lowering,
resource/fallback, token, budget, policy and benchmark-output guard tests.
Compiler structure checks 504 modules with no dependency errors.
The constrained 112-MiB native/GPU4PySCF test passes auto, panel ablation,
streamed dense, intentional host fallback and cold/warm force replay.
Expanded CUDA response/state/mixed-spin/metric tests and public DF tests have
83 passes, one pre-existing skip and eight old route/resource assertions.
Those same eight failures reproduce on the saved stage-two binary in the same
environment; they are neither suppressed nor repaired here. Initial public
tests also found seven stale non-PBE0 AOT identities; rebuilding the required
PBE/R2SCAN artifacts resolves them (29 public tests pass, one skip).

Raw receipts, Slurm commands, profiles, cache/build logs, all paired E/F arrays
and ablations are retained in ignored `.artifacts/pbe0-df-force-fix/` and the
remote qualification directory. Keep current-state behavior in
`docs/user/dft_density_fitting.md`. Preserve the earlier energy notes rather
than silently replacing their historical evidence.

## Revisit when

The remaining full-grid geometry work has a qualified exact work/schedule
improvement with complete E/F and independent error gates, or the DF resource
contract is unified and can replace the explicitly partial consumer bounds.
Do not repeat AO refits for J, spend unknown providers' allowances, change the
metric gauge/precision, or promote a Becke experiment from one microbenchmark.

## Current-master integration follow-up

PR #2164 merged on 2026-10-10 at 01:13:59 Asia/Shanghai and its old fork
branch was deleted while qualification continued. The force-only patch is
therefore submitted separately as PR #2170, based on master `b315d4056`.
Its initial source tree is `4b918b3d285afa2d5477e68584f9671be0281e05`;
only the fourteen force/code/test/documentation files differ from that base.
Other agents' dirty CPU-integral and direct-J worktrees are not staged.

The clean rebased tree passes 309 host tests and the compiler checker covers
505 modules with zero dependency errors. A complete ccache-backed Release
sm_120 rebuild plus a finite Slurm qualification passes the constrained
cold/warm/panel/host-fallback CUDA test again. One fresh integrated 96-atom
E/F endpoint takes 98.606486 s, including 18.500749-s prepare and
24.951002-s force, with 25 iterations / 25 Focks. This is a single integration
control, not a replacement for the earlier paired timing population.

Its independent energy error is 1.90994e-11 Eh, force error 1.52676e-10
Eh/bohr and residual 3.82286e-13. It retains one completed B charge pass,
zero repeated fitted AO projections / metric BLAS dots, a final-U hit,
256 geometry lanes, 9,216 phase batches and the same additional-consumer
399,228,960/192,229,472-byte device/host bounds. The integrated library SHA-256
is `c7265b4d71c673efeb2c62dc159089376be02416764e3cf5145e98272dc81176`.
Receipts are `force-followup-96.json`, `force-followup-96-df.jsonl`,
`force-followup-small-tests.log` and `followup-host-tests.log` in the ignored
qualification artifact tree.
