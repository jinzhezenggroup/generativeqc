# Automatic energy-only Q32 cap

Frozen #2191, fresh ABBA; not latest-master/global/force promotion.
Shared reader: samples retain four ABBA/four pilots; validation owns
gates/work/provenance/source reconstruction. Recipes: gzip JSON name-to-script
map; extract into scratch, adjust paths and create fresh hashes.
Restore #2191 via its sibling manifest; apply three pinned patches to the frozen
parent using `git apply --unidiff-zero` in order. Finite `srun`, Slurm visibility,
ccache. Rationale: the energy-auto-q32 Agent Note. No Release/archive published.
