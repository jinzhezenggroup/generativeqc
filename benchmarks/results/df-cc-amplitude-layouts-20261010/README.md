# DF CCSD amplitude-layout epoch experiment

This is a rejected performance-promotion experiment, not an accepted public
speedup. The production change is not retained. The reconstruction patch applies
to `ec71ef7fea6f705a3623b45c7bfb51105c43337a`; the compact records preserve every
completed clean observation.

## Scope and qualification

Native molecular energy-only DF CCSD(T), ethane230, 230 spherical AOs,
9 occupied / 221 virtual orbitals, 488 auxiliary functions, all electrons active,
FP64, 64-GiB correlation budget, DIIS history 8 and Q batch limit 8. Each fresh
process performs RHF (including its DF preconvergence guess), correlation-source
construction, CCSD, original expanded physical residual replay and standard (T).
No force, Lambda or orbital-response result is inferred from unrequested fields.

Finite n2 Slurm job 2795 uses one RTX PRO 6000 Blackwell Workstation GPU,
CUDA 12.9.1 sm_120 Release, driver 595.91.07, 600 W, and one thread per numerical
library. The order is baseline / candidate / candidate / baseline. GPU visibility
is scheduler-owned and is never overridden. The two frozen libraries and
executables are content-addressed in `provenance.json`.

The native helper timer includes inner phase-owner teardown; the fresh-process
wall additionally includes startup and all final caller/reference cleanup.
Independent oracle checks run outside both timers. All four observations pass
the pinned independent PySCF 2.14.0 total-energy / separate-(T) gates of
`1e-8` / `1e-10` Eh and physical replay norms at most `1e-10`. Maximum errors are
`2.203e-12` and `8.448e-14` Eh. Solver trajectory, Q work, GEMM work and contraction
summands match. The corrected prototype also passes 34 native GPU regressions,
33 focused host regressions and the targeted aggregate-order regression.

## Result and decision

| Median | Baseline | Candidate |
| --- | ---: | ---: |
| Fresh-process complete wall, s | 195.954365 | 195.629638 |
| Native helper, s | 195.641453 | 195.311958 |
| Complete CCSD solver, s | 99.847012 | 99.368463 |
| Logical packing bytes | 5,578,493,904,992 | 5,001,216,728,672 |
| CCSD numeric capacity, bytes | 4,130,767,416 | 4,162,416,184 |

Packing traffic falls 10.348%; these are logical reads/writes, not measured DRAM
traffic. Complete wall improves only 0.166%, versus a 0.566-s spread between the
two native baseline observations. The small CCSD improvement does not justify
promoting a new retained-storage policy as an endpoint performance win. Keep the
original owner and pursue a larger measured bottleneck instead.

## Reproduction and retained artifacts

- `measured-source.patch.gz`: corrected production-source delta against the
  frozen parent, uncompressed SHA-256 recorded in `provenance.json`.
- `prototype-tests.py.gz`: layout, refreshed-amplitude and C++ aggregate-order
  regressions for the reconstructed prototype.
- `samples.json.gz`: all four clean records, argv, full process wall and gates.
- `measure.py.gz`: exact alternating driver; its machine-local paths identify the
  frozen experiment directory, not a portable installed benchmark interface.
- `summary.json` and `provenance.json`: medians, scientific gates and identities.

Full archives, libraries, compiler-cache receipts, raw logs and failed prototypes
remain ignored artifacts at `n2:/data/jzzeng/qc-cc-layout-20261010/` and locally
under `.artifacts/df-cc-layout-20261010/`. Real-device tests and timings must run
through finite Slurm allocations (`main`, `--gres=gpu:pro6000:1` on n2).
Compilation uses verified ccache. Standalone frozen-library probes must use the
system C++ linker, retain the ELF `.so.0` SONAME link and expose their matching
generated-header directory; the initial setup failures are not scientific gates.
