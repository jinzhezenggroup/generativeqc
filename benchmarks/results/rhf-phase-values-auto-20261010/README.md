# Bounded default-auto exact RHF phase values

This is a source-matched **cold ethane CCSD(T) energy-plus-force** qualification,
not a molecule-name dispatch rule, an online profitability model or a universal
performance-promotion claim. `validation.json.gz` contains every raw endpoint
output, paired timing, numerical gate and separate diagnostic profile. The
shared publication reader verifies the stored identities and decompresses it.

## Source and contract

- Base master: `15e052d800085c61373937ed9c81c06cc6ca3548`.
- Measured clean revision: `2db14f19fc1a460989785a0d2e21e45fa0a5dac5`.
- Linked library SHA256:
  `6d771d67542d82e47d4442b19f6a4b9ad74afd971d2f4ff32b80485b03c20ded`.
- Linked endpoint SHA256:
  `3d247bce7a3c6ea98a0af82c120523001f01406307f1e44d08904b816a381745`.
- Input SHA256:
  `9428f2b1d1db38ffa374387705099e8d57fde98e0e068faed2861b04604a1c6e`.
- Ethane: 230 spherical aug-cc-pVTZ AOs, 488 aug-cc-pVTZ-RI functions,
  all-electron exact conventional RHF plus correlation-only DF RCCSD(T),
  strict FP64 and all 24 analytic force components.
- Each timing is a fresh process including the native initial DF density guess
  and complete E+F. Compilation and installation are excluded.

The measured source can be reconstructed from the durable base commit by
decompressing and applying `measured-source.patch.gz`. Subsequent publication,
inventory and evidence-test changes do not change the measured production code.
The preceding opt-in evidence at base `4444d0376` is a distinct historical
campaign and is not pooled or relabelled here.

## Routing and acceptance

Unset/`auto` admits only a fresh single restricted unscreened physical-reference
bucket with uncovered generic f-shell Fock work, at least 64 public/128 Cartesian
AOs and an iteration limit of at least eight. The limit is not a forecast of
eight actual iterations. A fresh supplied seed can already be converged; only
known reused buckets are excluded as warm. Fully generated/lower-angular,
smaller, short-limit and resource-refused routes retain their exact evaluator.
`0` disables values; `1` requests them without bypassing scientific/resource gates.

Optional immutable canonical FP64 values have an 8-GiB ceiling, complete numeric
and host-preparation admission, and a 256-MiB free-device reserve. Compensated J/K,
shared `h + J - K/2`, the final physical Fock and canonical audits remain unchanged.
Admitted graphs/values retire before correlation. Refused/warm calls reuse
ordinary bucket graphs rather than recapturing on every call.

Three distinct RTX 5090 UUID strata on n1 run A-B-B-A, six samples per side,
with A=`0` and B=**unset**, not forced `1`. All solver/operator/work counts and
precision gates must agree. The n4 graph-node profile is separate and is never
pooled into clean ratios. Sampled one-second NVML total-device occupancy is not
an exact allocator or owned-buffer peak. Cached build wall is not an uncached
compile cost. The common formal performance envelope remains `not-run` because
these clean samples do not expose every per-iteration solver history; the
retained complete comparisons are scoped descriptive evidence.

Ethane energy and the two-coordinate/two-step independent FD checks are retained
same-Hamiltonian references; the 24-force comparison is a retained qualified
native vector, not a newly generated independent full force oracle. Five new
allocated-device tests generate fresh independent PySCF RHF original/displaced
energies for water, methane and methane-SPD and check forced/cold/warm,
changed/restored geometry, budget refusal and ordinary graph reuse. The separate
methane memcheck covers automatic admission and refusal, geometry/lifetime and
forced/disabled routing without changing Slurm visibility.

## Reproduce

1. Restore the base plus measured patch in a `source/` checkout. Select a common
   enclosing root for source, build outputs, input and scripts; adjust the
   retained recipes' `root`/`deps` paths for the local environment.
2. Run `build.sh` with verified ccache, CUDA 12.9, GCC 11, Release, sm120,
   `portable_cuda` AOT and `GENERATIVEQC_CUDA_FAST_COMPILE=OFF`. The library and
   endpoint are source/binary-identified; recompilation may change binary hashes
   and must freeze new identities, not relabel these observations.
3. Run each `run-abba.sh` through finite `srun` on n1/n2/n4/n5 with a compatible
   GPU request. Do not use n3 or override `CUDA_VISIBLE_DEVICES`. Repeat on three
   distinct UUIDs; the manifest records the allocation command.
   This campaign uses n2 only for CPU compilation because of its driver/NVML
   mismatch; real-device qualification runs on n1 and n4.
4. Run `run-qualification.sh` through a finite GPU allocation to generate fresh
   small-case oracle tests and a node-level CUDA-graph profile. For memcheck,
   reuse its methane probe/fixture and run `run-memcheck.sh` via finite `srun`.
5. Recompute all error blocks against `references/` and compare each paired
   endpoint's complete work and timing. Repository host evidence tests also
   recompute the retained numerical gates and verify source/publication scope.

Raw compiler logs, sampled NVML streams, pytest journals, Nsight report/SQLite
and sanitizer output remain in the ignored/local audit campaign
`evidence-rhf-auto-20261010/`. No external archive or Release is published.
