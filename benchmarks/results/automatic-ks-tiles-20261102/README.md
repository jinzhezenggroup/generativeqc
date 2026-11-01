# Automatic KS SCF and force tiles: PR qualification snapshot

Measured production source:
`d243aaf78c82f870d8110b40eeddc2950489cfbe866f586d070487f1e695562f`.
The capsule binds exact source reconstruction, executed driver bytes, main and
companion libraries, scheduler visibility, owner receipt hashes and lifetimes.

The public no-argument path selects SCF and stationary-force tiles independently.
For these large fitted strict-FP64 RKS tasks, A100 selects SCF 1024 (WB97M-V 512)
and force 1024; RTX 5090 selects SCF 256 and force 1024. Explicit SCF 256 retains
force inheritance at 256. Product names are not used by the policy.

## Protocol and measured scope

README water-96, stock grids, method-specific bases and full energy plus analytic
forces, including moving-grid response and r2SCAN-3c corrections. Each of three
arms has a conditioning owner, then auto / 256 / GPU4 / GPU4 / 256 / auto,
with two cold and six warm samples per arm. Cold includes setup and the first
complete endpoint; warm reuses the same geometry. Compilation caches are warm;
math libraries use one thread. All samples and work counts remain in the capsule.
Native and GPU4PySCF retain their respective convergence rules and pass the
independent README observable gates before any successful-allocation timing ranks.

Positive percentages mean slower automatic defaults. The columns compare
complete median endpoints. This snapshot has 19 qualified comparison groups;
no universal GPU non-regression or all-functional cold-speedup claim is made.

| GPU | Method | Cold vs explicit 256 | Warm vs explicit 256 | Cold vs GPU4 | Warm vs GPU4 |
| --- | --- | ---: | ---: | ---: | ---: |
| a100 | pbe | -22.93% | -34.10% | -17.86% | -9.35% |
| a100 | pbe0 | -17.07% | -26.62% | +4.74% | -9.57% |
| a100 | b3lyp | -17.52% | -27.39% | -0.16% | -7.82% |
| a100 | r2scan | -25.09% | -34.78% | -30.24% | -27.39% |
| a100 | r2scan-3c | -15.34% | -23.89% | +1.68% | -6.11% |
| a100 | m06-2x | -18.83% | -26.24% | -9.69% | -21.17% |
| a100 | wb97m-v | -4.00% | -12.92% | +33.43% | -50.65% |
| 5090 | pbe | -7.39% | -18.40% | -9.87% | -8.07% |
| 5090 | pbe0 | -4.39% | -12.39% | -9.65% | -42.15% |
| 5090 | b3lyp | -4.58% | -13.96% | -13.29% | -41.03% |
| 5090 | r2scan | -10.49% | -25.45% | -17.27% | -29.92% |
| 5090 | r2scan-3c | -6.27% | -16.68% | -3.90% | -31.01% |
| 5090 | m06-2x | -4.60% | -14.78% | -13.08% | -47.98% |
| pro6000 | pbe | -7.91% | -19.81% | -19.71% | -8.77% |
| pro6000 | pbe0 | -4.46% | -13.28% | -15.20% | -41.79% |
| pro6000 | b3lyp | -5.02% | -15.10% | -17.74% | -41.22% |
| pro6000 | r2scan | -10.85% | -27.04% | -23.48% | -29.60% |
| pro6000 | r2scan-3c | -6.15% | -17.44% | -7.10% | -29.80% |
| pro6000 | m06-2x | -4.83% | -15.80% | -17.05% | -47.85% |

## Unfinished qualification and retained failures

- A100 WB97M-V still takes 28 cold SCF iterations/Fock builds, versus GPU4's
  10 iterations and 11 effective-potential builds. The native initial guess also
  has separate work. Its cold endpoint is 33.43% slower, despite warm being
  50.65% faster. Investigation of the convergence difference follows this PR.
- RTX 5090 WB97M-V is still running at the snapshot boundary. PRO 6000 WB97M-V
  was cancelled at the user's request; its partial receipts and cancellation
  proof are retained and never ranked. No further PRO 6000 tests are requested.
- Larger/moved 120-atom PBE passes independent observable and reference-grid
  stability gates on all three measured devices. The r2SCAN-3c level5/6 reference
  difference fails the stricter quarter-gate grid-stability prerequisite;
  these runs remain failed qualification, not passing production evidence.
- Public options/resources tests pass (163 on RTX 5090). Both resource probes
  preserve six force-lifetime samples, but the combined qualification is
  incomplete. Version 1 incorrectly expected public DF NumericCap planning to
  be supported. Version 2 incorrectly treated `density_fitting_memory_budget_bytes`,
  a DF-provider workspace hint, as a whole-calculation XC capacity promise.
  The existing API documents those distinct contracts. Neither failed probe is
  silently relabeled as complete; corrected end-to-end resource qualification
  remains outstanding. Failed A100 public-v2 job 224498 has controller evidence
  but no returned output archive at this snapshot.
- Some later failed Blackwell controller records were purged before collection;
  their outputs remain retained but cannot qualify successful allocation timing.

Host policy/force/precision checks: 213 passed. Inventory tests: 134 passed.
Final-library CPU checks: 132 passed, 33 GPU skips on the CPU-only worker.
These are separate from the incomplete GPU promotion obligations above.

## Reproduce

From the repository root:

```bash
PYTHONPATH=.:python python benchmarks/results/automatic-ks-tiles-20261102/report.py > /tmp/automatic-ks-report.json
PYTHONPATH=.:python python -O benchmarks/results/automatic-ks-tiles-20261102/report.py > /tmp/automatic-ks-report-optimized.json
cmp /tmp/automatic-ks-report.json /tmp/automatic-ks-report-optimized.json
```

`--require-complete` additionally requires all seven comparisons, non-regression
against explicit 256, and larger/moved/resource validation on A100 and RTX 5090;
**it intentionally fails for this partial PR snapshot**. `--devices` can select
an explicit device matrix; adding PRO 6000 does not waive its cancelled owner.
The reader authenticates the exact-FP64 v3 streams and independently recomputes
observable gates, actual resolved tiles, strict precision and scientific work.

The whole final-source campaign is retained, including conditioning, slower,
failed and cancelled observations. Entire earlier source campaigns remain in
the ignored investigation capsule identified by checksums inside this capsule;
there is no outcome-based sample selection. Large binaries and raw profiler
payloads remain task-local with hashes. No release assets are used. The fixed
80 MiB tracked-evidence ceiling is unchanged.
