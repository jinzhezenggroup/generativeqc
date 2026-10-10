# Incoming-residual mixing, CUDA DF energies only

All 48 CPU, 32 native CUDA and 48 stronger CUDA arm records, two budget outputs,
every binary supplied input/output, 4 frozen ABBA, 2 pilot and 2 current-master
complete endpoint samples are retained losslessly in validation.json.gz and
its hash-bound raw-receipts.json.xz companion (stdlib lzma + JSON).
The shared reader and offline tests validate hashes, original numerical gates,
scope, sample ordering, graph-work arithmetic and matched capacities.

Use tools.generativeqc_validation.record.load_publication_record to read it.
Extract reproduction_recipes to an ignored directory; raw-receipts.json.xz has
UTF-8 text or base64 binary data with exact bytes/SHA-256. Restore the frozen
#2200 baseline through its sibling source-reconstruction receipts, then apply
prototype-source.patch.gz. That library changes CUDA only; CPU feasibility is
a separate standalone build. Current integration uses git archive of the exact
integration_base plus production-source.patch.gz (gzip -dc | git apply).
Recreate master-source.tar.gz and production-overlay.tar.gz as specified by
integration-gpu.sh, or configure each source directly with its retained flags.
Adjust artifact paths and binary hashes to the fresh build; use finite srun,
the Slurm-assigned visibility and ccache. No external archive is required.

pilot-measured.py is the actual pilot runner; pilot.py is the later ABBA runner.
Their original scope text and receipts are preserved, not rewritten as a
current-master result. The integration pair qualifies actual scoped dispatch
after RHF master changes; it is not included in frozen ABBA medians. Observed
speedups are not formal/global, force/response or process-memory claims.
