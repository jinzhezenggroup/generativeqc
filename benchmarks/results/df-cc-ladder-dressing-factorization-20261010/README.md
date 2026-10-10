# One-sided DF ladder dressing

Four fresh energy-only ethane230/o9/v221/Q488 endpoints run ABBA in Slurm
2814: FP64, Q8, full DIIS8, 64 GiB. Baseline: frozen copy-elided #2184,
not latest master. Complete wall includes RHF/source/replay/(T)/teardown.

Median complete wall: 160.801392 -> 157.290694 s (**−2.18%**).
Complete CCSD: 64.686691 -> 61.154653 s (**−5.46%**).
Two samples/selection: no statistical/global or force/response promotion.

`validation.json.gz`: provenance, identities, work, gates and qualification.
`samples.json.gz`: every ABBA sample. Use the shared publication reader.

Reconstruct the frozen parent; apply the pinned baseline patch, then this
`measured-source.patch.gz` with `git apply --unidiff-zero`. Recipes require roots
and fresh pilot provenance/binary hashes for new runs. Use finite Slurm and
assigned visibility. No Release/archive is published.

Rationale and fallback policy: the implemented Agent Note.
