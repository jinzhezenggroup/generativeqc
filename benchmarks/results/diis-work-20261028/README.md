# Production DIIS work qualification

This is a bounded instrumentation qualification for #1874, not an algorithm or
performance promotion. The compressed receipt retains 60 complete E+analytic-force
observations: RHF, ordinary PBE RKS and Graph-replayed PBE RKS; diagnostics off/on;
two independent histories; cold, warm, moved, rollback and cleared-warm replay.
Independent CPU PySCF SCF and analytic gradients use the same public basis/grid
inputs. Native and oracle computations remain separate.

Run the offline numerical/work checks:

```bash
python benchmarks/results/diis-work-20261028/verify.py
```

Production source base is `f1f910aead420f19d26e60aa5d1244da3900b44a` plus the
file digests in `qualification.json.xz`. Measurement preceded the commit; the
receipt explicitly identifies changed production files and the final test
harness. The qualified Release library SHA256 is
`c8d6ff8ef5cedd3f655ae3fc5faa02eec7d762c89dd63958eae29453c49cd5c0`, unchanged by the
final no-op build. All 464 C++/CUDA compilation commands use verified sccache
0.16.0. Cache statistics are retained as provenance, not a speedup measurement.

Slurm jobs 14808 (final observations) and 14807 (memcheck) ran on node1's RTX 5090,
CUDA 12.9, sm_120, with finite ten-minute allocations and scheduler-assigned device
visibility. Seven production endpoint/control tests pass; memcheck reruns all
seven with zero errors and zero leaked bytes. The final test harness additionally
checks that each KS attempt adds exactly 24 scalar D2H bytes and one terminal
synchronization, with unchanged matrix transfers. Graph submissions are observed
from the actual owner, independently of device Gram counters.

The existing native cached-Gram suite passes wrap, retirement, reset, invalid
history, inactive-item and staged-route checks. The native KS capacity suite runs
with diagnostics enabled, including shared limits, failed admission/retry and
complete E/F; every owner releases its charges. Native HF layout checks observe
16 extra device bytes and 24 host bytes per history, restoring the original
numeric allocation when disabled. Ordinary native checks ran in Slurm job 14805;
staged 512/513-AO checks ran separately in job 14809. The metadata-only HF layout
check requires no GPU.

The first endpoint trial is retained in ignored logs: its RHF oracle used
differently rounded STO-3G coefficients, and its paired comparison incorrectly
required bitwise equality. Final gates use identical coefficient inputs and
1e-12/1e-10 paired energy/force tolerances, plus independent 1e-8/1e-6 physical
gates. One existing resource test assumes native CUDA force profiles are absent;
that test is excluded in this force-enabled build, while the other 16 resource
tests pass. No production tolerance was changed.

Full raw journals, failed trials, build logs, commands and sanitizer output remain
under ignored `.cache/1874/remote-evidence/` in the task checkout and
`/home/jzzeng/tmp/qc-1874-diis-20261028/evidence/` on node1. This tracked receipt
contains the complete endpoint values, independent references, observed DIIS
counters, failure/fallback controls and source/binary/build/Slurm identities.

For a fresh real-device run, build the native library with CUDA and native
stationary `pbe_rks;pbe0_rks` profiles, then run inside an explicit finite Slurm
allocation with `GENERATIVEQC_LIBRARY` pointing to that build:

```bash
GENERATIVEQC_CUDA_DIIS_WORK_TEST=1 PYTHONPATH=python:. \
  python -m pytest -q tests/python/test_cuda_diis_work.py
```

Logical dot/vector-element counters are not kernel counts, physical bytes, or
endpoint timing. Uninstrumented fallback work remains unavailable.
