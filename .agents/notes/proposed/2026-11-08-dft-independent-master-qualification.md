# Proposal: ship DFT streaming independently of the old HF draft

Status: proposed; latest-master native qualification in progress
Date: 2026-11-08

## Problem

The qualified normal native DFT predecessor is recorded in
`../implemented/performance/2026-11-08-dft-streaming-shared-rys-xy.md`.
Its source identity7fbc and measured7.30/7.67% A100 improvements are historical
evidence, not qualification of newly merged master2bf3fd98d (#3044), which
changes production DFT derivative/AO owners.

Merging the entire old HF foundation into that master would introduce unrelated
Libxc doc/binary snapshot conflicts and375558 excess benchmark bytes. Neither
blindly selecting qualification snapshots nor enlarging the evidence budget is
an acceptable way to ship the DFT-only optimization.

## Decision under qualification

Reuse a clean DFT worktree and create `perf/dft-streaming-xy-master` directly
from2bf3fd98da9c6cfba46ca83345e63e1d934e4780. Apply only the isolated DFT commit,
not the old HF branch ancestry, adaptive/prepared-Fock WIP, expanded paged-HF
default capabilities or prior Libxc snapshots. The old dirty worktree remains
untouched. The compiled streaming capability and actual FP64/resource policy
remain independent of paged-HF profitability rules.

Latest-master normal Release source identities are:

- Control: `bda7f76be67202a57eb1001cfbe29bd4d4d844a85c144b1854a264a96ffdb208`.
- Candidate: `75d94b32dd3e0b13be41bcc7a75f8634ebb15a1e3acbaa6f74be9d4c1afbce9e`.

Source archives exclude only historical benchmark-results data, not build inputs.
Builds16857/16858 are finite Slurm CPU allocations on node1, with the verified
sccache0.16.0 launcher and exact borrowed-source/runtime restoration. GPU job16859
is explicitly dependent on successful build16858, reserves a5090, and runs its
endpoint through `srun` without overriding visibility. These submissions alone
are not completion or numerical/performance evidence.

The independent source split passes85 focused host tests. Historical artifact
replay continues to authenticate its original images. Image identities belong to
each data cohort rather than fixed Python constants, and cold kernel census is
checked against the actual measured cold Fock count, not an assumed nine steps.
Fresh normal native A100/5090 complete-endpoint, independent scientific, work,
per-warm and resource/default-execution gates are still required for this source.

## References

- Issue2976, old draft PR3048 and DFT commit31da3e16a.
- `benchmarks/results/dft-shared-rys-streaming-20261108/` is explicitly the older
  qualified image; do not relabel it with the new source identities.
