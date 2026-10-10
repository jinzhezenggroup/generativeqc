# Bounded DF-CC BLAS workspace

Shared reader: validation.json.gz embeds retained_samples (4 ABBA, 2 pilots,
1 integration) and reproduction_recipes (name-to-script map), avoiding duplicate
metadata. Restore #2191/#2197 via sibling Git manifests; apply patches in
source_reconstruction order with git apply --unidiff-zero. Copy pinned repo test
sources, extract recipes, adjust paths/hash binaries, and use finite srun with
Slurm visibility and ccache. Current-header ownership differs from frozen timing.
No formal/global/force/latest-master promotion, RSS claim or external archive.
