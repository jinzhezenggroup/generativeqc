# Hückel seed preparation qualification

Run `python benchmarks/results/hf-seed-preparation-20261025/reproduce.py` from any
checkout to verify artifact hashes, replay all 44 complete energy/force checks
and recompute the paired timing medians. This replay needs no GPU or NumPy.

The three paired rounds compare base `98531471e` plus the retained exact patch
against the same baseline, using default FP64 Direct HF48/96. There is no experimental
screening in these measurements. All original and displaced references are
independent GPU4PySCF calculations. Timing includes construction, preparation,
SCF and host-visible analytic forces. Warm calls use each population's own
frozen density.

`paired.json.gz` and `moved.json.gz` retain every sample. `provenance.json` binds
the source files, base revision and patch; `measured-change.patch.gz` reconstructs
the measured source. `launchers.json.gz` retains the executed launchers and
qualification scripts. Build receipts, test logs and resource/memory-check logs
retain the scheduler, compiler cache and failure evidence. The sanitizer's first
run encountered the existing negative graph-capture probe; the memory-only
rerun disables CUDA API-error reporting, not memory checking. Large exact-byte
resource admission remains unsupported and is explicitly recorded. The
`hf-seed-production` receipts retain these earlier failure diagnostics; final
timings, tests, resource cases and memory checks use `hf-seed-final` after the
SCF adapter boundary cleanup.

The [decision note](../../../.agents/notes/implemented/performance/2026-10-25-huckel-seed-preparation.md)
explains the numerical, scratch-lifetime and fallback boundaries.
