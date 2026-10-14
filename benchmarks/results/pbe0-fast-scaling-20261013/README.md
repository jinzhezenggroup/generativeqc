# PBE0 Fast size scaling: distinguish mixed work from FP64 fallback

This campaign tests whether the current XC-only warmup becomes substantially
faster as the AO count increases. It does **not** expand public Fast admission.
The production limit remains 24–256 AOs; larger domains request an FP64 audit.
The ordinary FP64 comparator already includes exact resident J/K reuse.

## Complete endpoints

RTX 5090, Release/sm_120, CUDA 12.9.1, fast-compile disabled. PBE0/RKS,
spherical def2-SVP, exact full-range J/K, no DF, moving grid `(48,16,32)`,
energy/density/screening tolerances `1e-10 / 1e-8 / 1e-12`, maximum 180 iterations.
Each native endpoint starts from Hcore; these are **not automatic-MINAO results**.
Construction, SCF/refinement, all analytic forces, host export, cache release
and CUDA synchronization are timed. Process caches are warm; converged native
densities are not reused between endpoints.

The 192-AO geometry is the WATER27 S4 octamer. The 384- and 768-AO cases are
synthetic translated arrays of two and four such octamers, **not optimized
16-/32-water clusters** or a chemically diverse large-molecule qualification.

One first call per mode is retained separately from three alternating-order
measured repeats. The experimental column uses ordinary AUTO with diagnostic
XC selector `3` and bounded DIIS-history retention, not the public Fast facade.
It retains strict final SCF and FP64 analytic forces.

| AOs | FP64 median (s) | Experimental XC median (s) | Ratio FP64/experimental | Experimental mixed + strict Focks; retries |
| ---: | ---: | ---: | ---: | --- |
| 192 | 8.514682 | 8.202226 | 1.0381 | 10 + 9; 0 |
| 384 | 24.066814 | 23.546930 | 1.0221 | 1 + 20; 1 |
| 768 | No complete endpoint | No complete endpoint | Not accepted | Both generated-force paths fail out of memory |

FP64 uses 17 and 22 Focks respectively. Every listed endpoint is independently
accepted at `1e-7 Eh` energy and `1e-6 Eh/bohr` maximum force-component error,
including first calls. All measured/first-call work counts and samples belong
in `summary.json.xz`; neither outliers nor retries are discarded.

**The 384-AO row is mostly FP64 retry work, not sustained FP32 acceleration.**
Every sample submits one mixed Fock, rejects it and finishes strictly. Its small
timing difference also accompanies one fewer total Fock than ordinary FP64.
It cannot establish that enlarging the mixed domain improves performance.
The 192-AO gain is modest, not a multi-fold endpoint improvement. Three repeats
are descriptive medians, not statistical confidence bounds.

## Retained 768-AO failures

The 768-AO experimental diagnostic (job 8045) fails during generated analytic
forces with `GENERATIVEQC error 7: out of memory`. Ordinary FP64 (job 8046)
also fails during generated forces with `out of memory`. The latter allocation
reports 32607 MiB total GPU memory and 18 MiB used before the endpoint. There
is **no valid complete-endpoint timing, force result or speed ratio** at this
size. The exception does not export a complete precision/work census.

The independent ordinary PySCF reference (job 8035) also fails its unchanged
SCF convergence criteria after 180 cycles. A bounded second-order Hcore retry
(job 8044) is cancelled after the experimental native force OOM removes the
prospect of an accepted comparison. Its last `running` checkpoint is retained
unaltered, with cancellation recorded separately; it is not a converged oracle.
The native comparison waiting for that reference is stopped before allocation.
We do not loosen tolerances, remove forces, switch grids/models or bypass GPU
visibility to manufacture a large-system result. These failures limit the
measurement; they do not prove that every future mixed-precision implementation
must fail or become slower on large systems.

## Public fallback and precision ablations

A separate **single first-call diagnostic** of public Fast at 384 AOs takes
53.264871 s: two FP64 solves, 22 + 23 Focks and two force evaluations. Its report
states `budget_or_domain_requires_fp64_audit`; the returned candidate submits
zero mixed Focks. This is not the experimental route above and is not a repeated
speed comparison. Its independent errors are `1.069e-11 Eh` and
`6.525e-11 Eh/bohr` maximum force component.

At 384 AOs, single-endpoint diagnostic ablations with the same Hcore seed give:

| Experimental selector | Narrowed work | Mixed + strict Focks | Retries |
| --- | --- | --- | ---: |
| 1 | PBE correlation point arithmetic | 12 + 9 | 0 |
| 2 | Points plus AO arithmetic | 12 + 9 | 0 |
| 3 | Points, AO and supported contractions | 1 + 20 | 1 |

All three pass the independent endpoint gates. Adding mixed contractions is
the observed trigger in this fixture. The exact device failure flag / numerical
root cause is **not established** by this ablation; do not assert a negative
density, clip density tails, or weaken checks on this evidence. The rejected
mixed step's metric electron counts remain correct. These single samples are
not a speed ranking of the selectors.

## Profile scope

An instrumented ordinary FP64 384-AO endpoint sums to 21.623403 seconds of GPU
kernel durations. Canonical/resident J/K kernels account for about 35.9%, AO
radial kernels 11.1%, XC matrix products/potentials/scatter 27.2%, and equal-spin
XC point kernels 3.3%. Exact name-substring selectors and sums are retained in
`summary.json.xz`. These are **kernel-sum shares, not endpoint wall-time shares**
or a claim that every operation in an XC family can use FP32. They include
strict work; the profiler endpoint is not a speed sample.

J/K, the eigensolver/DIIS, strict refinement and all analytic forces still use
FP64. Simply increasing system size does not remove these arithmetic boundaries
or guarantee a more favorable balance of work.

## Provenance and reproduction

Native source identity:
`a514e417d6ae21b91fde604baaaf2092db62f13f8bd17cb35a373bf556e858c4`.
Library SHA-256:
`6cf7a535f35e7579ea56d9efa34b777436a1d732587cf8eeeee08e444b9bdf60`.
This is the same native implementation as the
[earlier small-system qualification](../application-budget-fast-20261013/README.md).
The source manifest, driver/reference hashes, assigned device visibility, Slurm
jobs, endpoint work and oracle metadata are retained in `summary.json.xz`.
Host benchmark/admission checks: **11 passed**; Ruff check/format, diff whitespace
validation and the warnings-as-errors Sphinx build pass.
Raw JSON, logs, diagnostic scripts and the profiler report remain in ignored
`.artifacts/pbe0-fast-scaling-20261013/` and on node1 under
`/data/jzzeng/pbe0-fast-scaling-20261013/`; no Release or external archive is used.

With source `python/` and the repository on `PYTHONPATH`, an independent PySCF
installation and a matched tuned native library, select one of the cases in
`benchmarks/pbe0_fast_scaling.py`. Generate its reference with `--engine reference`;
then run the matched native comparison, preserving Slurm-assigned visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:30:00 \
  python -m benchmarks.pbe0_fast_scaling --engine native --case CASE \
  --reference REFERENCE.json --output .artifacts/scaling.json \
  --repeats 3 --modes fp64 experimental_xc
```

Leave precision/residency overrides unset; the driver scopes experimental
controls to the diagnostic mode and refuses preexisting overrides. `--modes
fp64 fast` measures the actual public policy including audits, not a renamed
experimental route. Every accepted timing requires an independent reference
with the identical scientific protocol.
