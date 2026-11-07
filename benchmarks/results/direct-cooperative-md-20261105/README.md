# Compact cooperative MD radial: A100 and 5090 qualification

Issue: #2976. Date: 2026-11-05. Scope: one static 96-atom water geometry,
r²SCAN-3c Direct, 1088 spherical def2-mTZVPP AOs, full energy and analytic
forces including D4/gCP. This is not a grid-converged production ranking or
a claim that the entire remaining A100 gap has closed.

## Complete endpoints

Each arm runs one cold constructor/preparation/SCF/force/host-export endpoint,
then three same-owner endpoints from its own frozen cold density. Each arm is
a separate process; baseline runs before candidate. Compiler/runtime artifact
caches are reused, not cleared. Tracing and force census are disabled in timed
processes. Both arms retain unmodified production defaults.

| Device | Arm | Cold (s) | Warm median (s) | Warm range (s) |
|---|---|---:|---:|---:|
| A100 PCIe 40 GB | baseline | 18.829708 | 8.462728 | 8.458465–8.567645 |
| A100 PCIe 40 GB | compact MD | 18.356019 | 7.974124 | 7.965012–8.123809 |
| RTX 5090 | baseline | 29.551986 | 11.473441 | 11.429283–11.544192 |
| RTX 5090 | compact MD | 29.451527 | 11.283032 | 11.263419–11.333649 |

Warm latency decreases **5.774% on A100 (0.489 s)** and **1.660% on 5090**.
All three corresponding 5090 warm ratios are <1, passing the predeclared ≤1.05
non-regression gate. Single cold observations are retained but do not establish
a robust cold-speedup distribution. Raw values are not rounded in the
losslessly compressed `.json.xz` populations.

The fresh same-5090 GPU4PySCF population completes at 44.275224 s cold and
15.174512 s median warm. Its electronic work uses GPU4PySCF; its documented
independent D4/gCP corrections execute on CPU inside each timed endpoint.
Native execution rejects PySCF/GPU4PySCF imports and uses native corrections.
The earlier A100 reference attempt is incomplete, so there is **no new
same-A100 GPU4PySCF timing comparison**. Cross-device comparisons below are
strictly scientific observable gates, not performance comparisons.

## Correctness and semantic work

- Native/native gates: energy per atom ≤1e-9 Eh, force maximum ≤1e-7 Eh/Bohr,
  force RMS ≤3e-8 Eh/Bohr. All 16 cold/warm cross-pairings per device pass;
  worst force difference is 1.005e-11 Eh/Bohr.
- Independent GPU4PySCF gates: energy per atom ≤1e-5 Eh, force maximum ≤1e-4
  Eh/Bohr, force RMS ≤3e-5 Eh/Bohr. All 64 pairings of both native arms/devices
  against the four fresh 5090 reference samples pass. Worst force error is
  1.350e-6 Eh/Bohr; worst force RMS is 4.497e-7 Eh/Bohr.
- 24 real-5090 Libcint J′/K′ Separate/Combined tests pass, covering RKS/UKS,
  Cartesian/spherical basis representations, and density-contraction enabled
  and disabled, including the existing bounded fallback.
- 66 focused generated-body/barrier/canary/compiler host cases pass; 1378
  additional MD math/reference/emission cases pass. The 73 skipped GPU/upstream
  tests are not counted as GPU acceptance. The 66 focused gates pass again
  after final master synchronization.
- Both native arms/devices retain 10 SCF iterations and 10 Focks cold, then
  1 iteration and 1 Fock per warm sample. The anchor/delta/closure counters and
  complete recorded settings agree for corresponding native samples.
- Default XC grid: 1,078,336 points; actual SCF/force tiles: 1024/1024;
  screening: 1e-12. No tolerance relaxation or Direct→DF substitution.
- The source schedule reduces 1991 interpreted DAG instructions / 24 levels
  to 322 recurrence destinations for degree seven or 486 for degree eight,
  using the caller's unchanged Boys ladder. Radial storage decreases from
  1991 to 339 doubles. These are generated work/storage counts, not additional
  independently timed kernels. Intrusive diagnostic work counters are retained
  separately and never treated as clean endpoint times.

## Identity, synchronization, and failures

Measured base commit: `a970014e8de544aa54c6e0662eda32ebc5e5388f`.

| Image | Native source identity | Core ELF SHA256 |
|---|---|---|
| baseline | `cd80f80304bae111bd98f6627167d7f1b119eac0f588ac9e653da06ca5f29153` | `d1d382be8bb039597237b72b71e7ca18b6bb4e1cacf23ba868503be4449337cc` |
| candidate | `b78434b9416f124db8bf73b3b339bae894027959ed0220f6731e1bb1b10c2293` | `05319ea1eada9df72daa64fe05d131c553de582211b6b594d4d2f44f5fba8871` |

The same authenticated normal Release SM80+SM120 ELF executes on both GPUs.
Builds reuse sccache 0.16.0 and the existing cache. The initial 1800 s candidate
compile times out (exit 124); a finite incremental continuation succeeds.
Both build/cache receipts and compressed complete compiler logs are retained.
The failed compile launches no GPU endpoint.

Before publication the branch synchronizes first to
`30657763a1f79c44e1b897b524ae9f54c96928cd` (#2991), then to latest master
`051a157c64061eac4fd9ee90d7becfa982f80161` (#3011), and then
`1e3169c06` (#3009). These updates concern CPU
UKS final closure, ECP qualification, and public CPU full-range Hessian admission,
not the measured CUDA force sources, generated math, or endpoint runner. API
comparisons are retained under `build/master-compare.json.xz` and
`build/master-second-compare.json.xz`; the subsequent CI-only update is audited
in `build/master-third-compare.json`. Timing images remain labeled with their
actual older source/base; they are not represented as newly rebuilt images.

- Clean A100 pair: LSF **225682**, exclusive GPU, finite 18-minute limit,
  terminal DONE. No visibility override or invented Slurm identifier.
- Clean 5090 pair/reference/Libcint: Slurm **16004**, finite 30-minute limit,
  terminal COMPLETED / exit 0:0.
- `diagnostic-a100/` retains intrusive job **225599**: native completes, but
  reference exits 137 after its finite cap with zero complete samples. It is
  not reference timing evidence and its cause remains unresolved.
- Bootstrap **225596** fails before any endpoint (Bash empty-array/nounset);
  **225598** fails HTTP 403 before endpoint because its transport filename is
  outside the allowlist. Neither is treated as a completed population.

## Replay

Immutable execution scripts, API comparison responses, and the independent
pytest XML are retained as lossless `.xz` files. Process/compiler/test logs
are retained as deterministic `.log.gz` receipts, with exact-byte, scoped
justifications in `benchmarks/evidence-policy.json`; neither general storage
limits nor scientific acceptance gates are relaxed. Decompression reproduces
the executed producer and original output bytes, rather than formatting them
after execution. Failed attempts remain separate from accepted populations.

From the checkout root, without a GPU or either electronic-structure engine:

```bash
python benchmarks/results/direct-cooperative-md-20261105/replay.py
```

Replay requires complete process populations, authentic source/core receipts,
matching semantic work/settings, independent numerical gates, the 5090
non-regression gate, and 24 non-skipped/non-failed Libcint XML cases. Output
must match `summary.json`. Validate retained bytes with:

```bash
cd benchmarks/results/direct-cooperative-md-20261105
sha256sum -c SHA256SUMS
```

`a100/runner.sh`, `5090/runner.sh`, and `build/` preserve real scheduler,
environment, authentication, compilation and endpoint commands. Replaying
real GPU work still requires a fresh finite allocation; these task-local
absolute paths are provenance, not a promise that historical environments
remain available. Scheduler/hardware text receipts with original trailing
whitespace are retained losslessly as `.txt.gz`, rather than rewriting their
bytes for a source-code whitespace check. No binary, Release, or external
backup is published.
