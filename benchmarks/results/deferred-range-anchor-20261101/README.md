# Deferred range warm-energy anchor

This experiment defers an otherwise unused incumbent-provider warm-energy
anchor, while retaining the accepted resident-source energy. It is **opt-in**
with `GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY=1`; the default remains eager.
An imminent force-allocation envelope must fit free device memory before
deferral. Later policy changes or unexpected eviction reconstruct the frozen
seed's own incumbent anchor in a counted preparation phase. No GPU-name gate,
quadrature, precision or physical convergence threshold changes.

The controls use README water96/WB97M-V, spherical def2-SVP, fitted full-range
J/K plus exact range correction, stock XC/VV10 grids and explicit force tile1024.
Each finite scheduled allocation reserves eight CPU slots with math libraries
single-threaded. One separate cold/one-warm population conditions caches, then
off/on/on/off owners each perform a complete cold E/F and three frozen warm E/F
calls. Cold includes setup. Every original sample and semantic work count is
retained; no external density or reference work enters production timing.

## Guarded version

Native source identity:
`0a9dda3c991cede2030fe3499bf730e51443701785527dda63199a61d671b59d`.
Receipts name base commit `1a3ba563dba0c16ca4ecef31cb61aef75e3c04b7` and explicitly
record dirty source. That base commit does not contain the prototype. Original
native identities and binary hashes are preserved rather than relabelled.

Negative percentages mean faster than the eager control on the same device.

| Device / allocation | Cold change | First warm change | Pooled warm change | Cold + three warm change |
| --- | ---: | ---: | ---: | ---: |
| A100 PCIe 40 GB / LSF 224156 | -8.66% | -0.16% | -0.11% | -5.78% |
| RTX 5090 / Slurm n4 1410 | +0.019% | -0.022% | -0.021% | +0.002% |
| RTX PRO 6000 / Slurm n2 3836 | -6.00% | -0.027% | -0.012% | -3.80% |

A100 cold medians are 140.271137 s eager and 128.121426 s deferred. All cold
solves execute 28 physical iterations: A100/PRO6000 use 29 eager versus 28
deferred Focks, while the 5090 force envelope retains 29 in both arms. Every
guarded-version warm call executes one physical iteration and one Fock.
The tiny 5090 variations establish no speedup; the previously transferred
first-warm cost is absent in these controls.

These are comparisons to eager GenerativeQC. A100 WB97M-V still trails the
retained same-device GPU4PySCF cold population from the preceding XC-unroll
campaign. This experiment does not establish all-functional parity or universal
performance on unmeasured GPUs.

## Rejected unconditional deferral

The first prototype has native source identity
`8532cf5a883047f83c6e4ab1f5c3e66d416f44a32d6251aabc72372ff94a0755`.
Its independent scientific checks pass, but the 5090 force consumer evicts the
range source and transfers the saved cold action into the first warm call.

| Device / allocation | Cold change | First warm change | Pooled warm change | Cold + three warm change |
| --- | ---: | ---: | ---: | ---: |
| A100 / LSF 224150 | -8.74% | -0.19% | -0.04% | -5.84% |
| RTX 5090 / Slurm n4 1409 | -6.32% | **+26.37%** | +0.034% | +0.14% |
| RTX PRO 6000 / Slurm n2 3835 | -6.00% | -0.031% | -0.009% | -3.80% |

The first 5090 candidate warm call takes about 73.11 s instead of 57.85 s and
executes two Focks for one physical iteration. Subsequent calls hide this cost
in the pooled median. This is negative evidence, not a promotion result.

## Correctness and remaining qualification

The reader replays the unchanged independent README energy, full-force and
reference-grid-stability gates for all 30 complete populations / 108 E/F calls.
It authenticates the original oracle files and their previously audited reuse
contract, rather than relabelling oracle producer identities. Source pairs
also retain physical iterations and pass tighter paired E/F gates. The largest
guarded-version paired errors are below 1.82e-12 Eh and 1.49e-12 Eh/Bohr.

Both versions pass 65 GPU cases per device: 12 frozen-seed invalidation controls,
two final-token/export/import controls, and 51 range-residency controls. They
exercise warm updates on/off, replay/capacity/bucket/factor changes, eviction,
failed begin/retry, device/host/disk sources and moved geometries. New small
systems use independently reconverged GPU4PySCF energy and full analytic force
references. Four eviction/failure/snapshot cases pass Compute Sanitizer memcheck
with zero errors in each version. Host controls separately cover the exact
force-headroom boundary and driver failures.

The 120-atom / 8 GiB eager controls timed out before any complete endpoint:
LSF 224154 at 1200 s and Slurm n1 15037 at 1650 s. They remain incomplete and
unranked. At a separately declared 32 GiB budget, the 120-atom 5090 eager arm
(Slurm15038) completed cold, retained, evicted-fallback and second calls before
its 1950 s limit interrupted the moved call. That is also an incomplete
lifecycle; the deferred arm did not run. The guarded A100 120-atom / 32 GiB control (LSF 224157) completes both
six-call lifecycles, including forced eviction and moved geometries. Its force
envelope keeps the eager anchor in both arms (cold 28 iterations / 29 Focks,
moved 20 / 21); every warm call uses one iteration / one Fock. Paired energy and
full-force errors stay below 9.10e-13 Eh and 1.78e-12 Eh/Bohr. A single larger
pair is a differential/lifetime control, not a ranked speedup. The separate 5090 108-atom / 32 GiB control (Slurm15039) also completes both
six-call lifecycles with eager work retained: cold 31 iterations / 32 Focks,
moved 20 / 21, and every warm call 1 / 1. Paired errors stay below 9.10e-13 Eh
and 7.00e-13 Eh/Bohr. Its complete lifecycle takes 898.551 s eager and 901.854 s
deferred-control (+0.37% for this single unranked pair). This smaller control
does not replace either incomplete 120-atom 5090 population. The PRO6000 120-atom / 32 GiB control (Slurm3838) also completes both six-call
lifecycles. It retains eager work (cold 28 / 29, moved 20 / 21, all warm 1 / 1);
paired errors are at most 4.55e-13 Eh and 1.35e-12 Eh/Bohr. Its complete lifecycle
takes 1778.698 s and 1778.778 s (+0.004% for this
single unranked pair). Neither changing the budget nor a smaller control erases
the original timeouts. Default promotion remains separate from these bounded
differential controls and incomplete 120-atom 5090 lifecycles.

Deployment failures also remain explicit: n5 job1942 started before its binary
copy completed and exited with SIGBUS/135. Its retry waited for all copies,
verified library/manifest hashes and passed in job1943. It supplies no timing
or numerical result. Full logs, builds, retries and profiler data stay in ignored
local artifacts; the compact receipt retains numerical samples, provenance,
validation summaries and negative-control outcomes.

## Replay

From the repository root, with NumPy and the benchmark dependencies:

```bash
PYTHONPATH=python:. python benchmarks/results/deferred-range-anchor-20261101/report.py
PYTHONPATH=python:. python -O benchmarks/results/deferred-range-anchor-20261101/report.py
```

Both commands verify checksums, recompute independent gates, check the complete
Fock census and print every warm observation, paired errors and full-lifecycle
medians. Gates remain active with Python assertions disabled. To rerun endpoints,
invoke `python -m benchmarks.readme_defaults standard --method wb97m-v --mode df
--atoms 96 --warm-repeats 3 --force-tile-points 1024 --inputs <qualified-inputs>
--source-revision <full-source-revision> --output <ignored-path>` in a finite
allocation, with the opt-in control set in the declared off/on/on/off order.

The implementation rationale and admission boundary are in
[the Agent Note](../../../.agents/notes/proposed/2026-11-01-deferred-range-warm-anchor.md).

Two historical recovery manifests have only JSON formatting whitespace removed
to retain these new samples within the existing checkout budget. Their data
tokens, recovery paths, payload hashes and byte counts are unchanged.
