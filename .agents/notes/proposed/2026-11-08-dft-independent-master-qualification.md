# Proposal: ship DFT streaming independently of the old HF draft

Status: implemented; superseded by the source-matched 2026-11-09 qualification
Date: 2026-11-08

## Problem

Completion and remaining gaps are recorded in
`../implemented/performance/2026-11-09-dft-streaming-master-qualification.md`.
The pending-job discussion below preserves the original investigation, not
current qualification status.

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

## Independent PR and qualification continuation

The exact independent tree was published as draft PR #3051, remote commit
`3ad8c2dc6b758d16e240f7b5af0a3f42d7534587`. Authenticated Git data API publication
was needed after fork SSH write refusal and stalled smart HTTPS. Its tree equals
the original local 1263ddd0c tree exactly; parent and attributed message are also
verified. Only commit representation differs, not scientific source identity.

Control build16857 reached the internal1500-second compile timeout (exit124),
after271/710 tasks, and restored the original source/runtime exactly. Its never-run
dependencies16858/16859/16860 were cancelled. The cache-preserving finite CPU
continuations are16861/16862, with dependent5090 endpoint16863 and complete-ELF
transport16864. A failed first resubmission used SSH alias n1 as a Slurm node name;
the scheduler rejected it before creating any job. The corrected scheduler name is
node1. These are compilation/transport facts, not new GPU performance evidence.

The same forthcoming A100 allocation also retains one fresh PBE0/B3LYP DF96
population per engine to check the other remaining DFT cold cases. These single
paired comparisons are not balanced ABBA optimization qualification. Separate
PBE0 and B3LYP cold-only kernel attribution is collected without counting warm
kernels. All endpoint populations still include their own three measured warms.

Initial hosted core-a stops at the Libxc catalog's checked input identities;
the exact live summary and stale campaign state are unchanged. The official
metadata-only reconciliation is recorded in
`../implemented/compatibility/2026-11-08-dft-streaming-catalog-binding.md`.
It does not turn historical Libxc or DFT measurements into current qualification.
