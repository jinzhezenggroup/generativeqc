# Decision: default to producer-proven restricted PBE0 force points

Status: implemented; clean source 99ebd196d default-route endpoint qualification passes
Date: 2026-10-10

## Decision

Default the private stationary point request to on. The fast kernel is still
selected only by exact mathematical composition, an owned same-generation
identical-spin rho/gradient witness, precomputed phased storage, sufficient atom
scratch, and absence of external seeds. Explicit `off` preserves the general
route for ablation and diagnosis. Older producers/consumers keep their v1
fallback; the public task-view ABI does not change.

This promotes the algorithmic result rather than shipping a permanently disabled
experiment. General/response/HVP algebra, grid, precision, SCF stopping rules,
direct/DF choice and complete force assembly stay unchanged. First derivatives
use four common channels and one exchange evaluation. Spin-antisymmetric second
derivatives cannot be recovered from this restriction and remain outside scope.

Remove the prototype's unused guarded generic `evaluate_restricted` entry.
Only the composition-qualified PBE0 bound entry remains. Its numerical body and
the eight-direction general/response body are unchanged from the qualified
header; the compile-time channel invariant remains intact. The PBE0-only
subnormal exchange ratio avoids division by quantized rho^(4/3), without a
cutoff, changed acceptance gate or independent production formula.

## Integration and evidence boundaries

Merge master `4444d0376` into the owned branch, preserving fitted reserve/tile
policy and stationary residency assets. The PR diff contains this optimization,
tests, documentation and evidence, not unrelated reversions from the old branch.

Prior snapshot evidence remains in
`../../proposed/2026-10-10-rks-point-producer-binding.md`: routing/parity,
memcheck/initcheck, all 87 independent points, and official complete warm/
moved-warm E+F comparisons at 48/96 atoms. Those measurements are the #2185
snapshot, not current master. Independent arms' frozen seeds differ; actual
SCF/AO/force work matches. Bootstrap timings are not clean cold/reconvergence
performance evidence.

Master-aligned promotion passes 583 focused CPU tests, with three real-GPU
opt-in tests skipped outside Slurm. This includes all emitted point references
under the unchanged strict gate, binding, launch-failure, ABI and fallbacks.
No failing general PBE/PBE0 oracle row is removed. General PBE/UKS/HVP numerical
qualification is not implied.

CPU-only Slurm 7098 builds authenticated native/grid/PBE0 AOT artifacts from a
clean Git archive of `99ebd196df529bdd45107af44b86b7879d3bb69f`, with verified
ccache 4.5.1 launchers and unchanged full-source manifests. Slurm 7106 completes
same-binary explicit-off versus unset/default-on endpoints on RTX 5090, with
five interleaved samples per arm/geometry. This source is aligned to master
`4444d0376`, not subsequent `4b330ff3d` or its new MD-J source reuse.

| Atoms | Replay | Off median (s) | Default median (s) | Reduction |
| --- | --- | ---: | ---: | ---: |
| 48 | warm | 5.532675 | 5.271843 | 4.714% |
| 48 | moved-warm | 5.559617 | 5.323170 | 4.253% |
| 96 | warm | 16.203113 | 15.665374 | 3.319% |
| 96 | moved-warm | 16.166703 | 15.627425 | 3.336% |

All four independently recomputed comparisons pass >2% and robust-noise gates.
Every measured call includes complete synchronized E+F/host return and performs
one SCF iteration/Fock build. Actual default restricted work is 2304 batches /
1,179,648 points at 48 atoms and 4608 / 2,359,296 at 96; zero general points
execute in those admitted default calls. Semantic SCF AO/force work matches.
All setup, priming and measured calls satisfy unchanged independent 1e-8 Eh
and 1e-7 Eh/Bohr gates; maximum errors are 1.0914e-11 and 3.7541e-11 respectively.
Each arm's seed is immutable, but cross-arm seeds are not bit-identical (maximum
density difference 1.2661e-11); coordinates match exactly.

The native/AOT/manifest hashes are respectively
`db17efc99ef6fc73a557b5b39be7091417a32ebf6f9b21fa278f4ca69f54c212`,
`6842f8f5dd37d5c71353faf975b2a1b420b9c628db682b1332a349a960a8ab0a`, and
`abe179c2bb252ef5cea7e06607f85f880427468a18d61e76cb96032228ad1f9a`.
The source archive hash is
`c0e7b52134b157765f0ffd469935a5ea66fcf403605ff3b31784f95636aec190`.
Within that official module, the general/bound point kernels use 254/148
registers and 96/0 stack bytes; these are static resources, not timing evidence.

Compact source reconstruction, original samples/references, independent error
checks, receipts and recipes are retained in
`benchmarks/results/pbe0-restricted-point-default-20261010/`. Its envelope accepts
numerical qualification and separately retains the passing scoped timing gates.
Resource-complete schema promotion is not asserted because global peak memory
and complete build duration were not measured; free-device snapshots are not
substituted. Raw NPZ seeds and binaries remain ignored. No cold/reconvergence,
GPU4PySCF-relative or latest-master acceleration claim is implied. Unrelated
master commits do not justify repeating older GPU matrices.

At the submission boundary, integrate master `4b330ff3d`, including its existing
lossless evidence compaction to remain inside the unchanged 64-MiB checkout
budget. The restricted header, native geometry owner and ordinary/compiler
stationary consumers remain byte-identical to the measured `99ebd196d` source.
The merged MD-J work is separate upstream SCF code; timings remain attributed
only to the frozen source. No additional GPU matrix is run for the CC/tensor
updates. The new compact-evidence/reference-reader/retention CPU checks pass.

## Revisit when

The mathematical domain, producer ownership/generation, planner or consumer
changes; a producer loses the proof; or complete endpoints regress instead of
using the bounded general fallback. Register counts alone cannot justify this
promotion.
