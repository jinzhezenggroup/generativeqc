# Lossless evidence: rejected screened-HF handoff strategies

Three related experiments retain their own identities, controls, cohorts,
summaries and replay entry points in the sibling directories:

- `hf48-screened-proposal-rejection-20261108`: rejected 384-AO admission.
- `hf-fp64-predictor-rejection-20261108`: rejected complete old-operator DIIS ring.
- `hf-earlier-handoff-rejection-20261108`: rejected four-step provisional cap.

Their raw bytes share compression by semantic responsibility, not arbitrary size
shards: `endpoint-observations.json.xz` contains molecular endpoints, independent
references and their producer records; `execution-provenance.json.xz` contains
drivers, source patches, controls, scheduler/device/build/cache/restoration
receipts and complete compiler stdout. Each stream is namespace-separated by
experiment. Each cohort pins both stream hashes and the exact population size;
the endpoint stream also retains every original per-file hash in its integrity
manifest. The loader checks the whole population and every member hash, even
under `python -O`. No numerical observation, failed tooling attempt or original
compiler-output byte is discarded. Purely redundant base64 transport stdout is
omitted with its checksum and decoded archives/closure receipts retained.

This repacking keeps correlated evidence below the unchanged repository storage
and large-file review budgets. It does not introduce new GPU qualification,
rerun any rejected strategy or relabel historical images as the current master.

Checked-in summaries and CLI output keep every arm, its work counts, numerical
pass counts/maxima and every paired-warm ratio. They avoid duplicating raw warm
times and the derived per-sample error table. Each `replay()` function still
returns the full per-sample calculation, and every raw input is retained.
