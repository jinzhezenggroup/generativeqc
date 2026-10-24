# Shared FP32

WB97M-V complete E/force medians; RTX5090, grid48/16/32, three repeats:
- 24 AO: FP64 6.783615s / Fast 4.987146s = 1.360220x.
- 96 AO: FP64 24.434694s / Fast 18.176775s = 1.344281x.

Summary: identities, samples, errors, work, slower baseline and sanitizer failures.
96 AO means 12 atoms, not 96.
Raw: `../evidence-vv10-range-fp32-20261015/`.

These are historical `ed4eff117c758467a023d33e3355cc8b436b527a`
measurements, not integrated-revision qualification. The compressed script is an
immutable record of that campaign, not the live rerun entry point.
Use the source checkout with `PYTHONPATH=python:.`.
For new work use `python -m benchmarks.shared_vv10_range_fp32 --engine reference
--case CASE --output RAW_ORACLE`, then `--engine native --case CASE --oracle
RAW_ORACLE --output RAW_RUN --repeats 3`, each inside a finite Slurm allocation.
Choose new ignored output paths. The live runner authenticates full scientific
inputs, actual selected libraries and source bytes, and retains failed attempts.
Existing independent reference receipts require an authenticated complete
protocol; historical case/grid labels alone do not qualify a reused oracle.
Missing historical raw data is not evidence of a new native failure.

Live timings retain continuous span and the excluded provenance gap separately;
the compared seconds sum construction/solve/force/export/teardown work. Ratios
are observations; performance acceptance is unassessed without a declared gate.
