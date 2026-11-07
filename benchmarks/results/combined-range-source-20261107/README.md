# Complete cold coefficient-bound range source

This is a source-matched full-Direct WB97M-V/96-atom standard-FP64 qualification,
not a fresh gpu4pyscf latency ranking. Each arm is a fresh process, with one cold
constructor + preparation + SCF + analytic forces + host export and one first
same-owner warm endpoint from its own frozen cold density. Default precision,
grids, convergence, force response and exact Direct semantics are unchanged.
Production endpoints reject PySCF/gpu4pyscf imports; independent references are
used only outside those timed endpoints.

## Decision and source

Enable coefficient-bound exchange inside the existing automatic repeated cold
range-source domain, with complete GPU LR residency, strict restricted full
Direct, immutable term identity and an independently bounded transient source.
Retain `LR + (c_full/c_LR) Full` in the existing allocation; one K contraction
then replaces separate full and LR K actions. The original energy/Fock assembly
applies the range coefficient once. Persistent quartic storage does not grow;
the temporary full source is audited and freed before SCF. This is not a fused
radial source-building recurrence.

No GPU-name specialization, CPU production solver, precision/grid relaxation,
Direct-to-DF substitution or convergence change is involved. Capacity refusal
preserves raw LR replay. A failure after mutation retires the source rather than
falling back to corrupted raw data. Analytic-force ownership stays unchanged.

The [decision note](../../../.agents/notes/implemented/performance/2026-11-07-coefficient-bound-range-source.md)
records identity, lifetime, ownership, memory and rejected-alternative details.

Final native source:
`b9c0e6d609dfae6ef2dc9721e896c13e19a4d239b3d029ae264de4c59f67e83b`.
SM80/SM120 Release library SHA256:
`2b867f8aba56371f155af043c3b34bdec7b75c73321ba1dfd44ef52ea4741fc9`.
Integration base: master `54d5eeadf248bc556d4b934ba59ffebc2972624b` (#3028).
Before submission, fast-forwarded to `17babea1beeced50a11254684b879cbf26157d72`
(#3025); it adds only unrelated benchmark documents. The integrated native
source identity remains exactly the qualified identity above.

Subsequent PR review added an optional refusal for compact bucket masks that
cover all populated buckets but omit empty ones. Such leases now retain usable
raw LR values instead of failing combined-source admission. The qualified
all-mask producer and contraction path are unchanged. This guard changes the
native source identity; the timings and GPU receipts below belong only to the
historical image identified above, not to the repaired source. The repair has a
passing compiled host regression; its added compact-mask numerical oracle has
not yet run on a GPU.

## Final complete endpoints

Each pair uses the same final image: explicit
`GENERATIVEQC_KS_COMBINED_RANGE_SOURCE=0` versus the control genuinely unset.

| Device / allocation | Raw LR cold | Default cold | Raw LR first warm | Default first warm |
| --- | ---: | ---: | ---: | ---: |
| A100 / LSF226273 | 96.189766 | 89.776354 | 15.019811 | 13.879518 |
| RTX5090 / Slurm16631 | 178.361599 | 178.619370 | 13.620170 | 13.624872 |

Times are seconds. A100 saves **6.413413 s / 6.67% cold** and **7.59% warm**.
RTX5090 changes by +0.14% cold / +0.03% warm; its resource envelope retains the
incumbent raw source route, not a slower forced composite. This is a qualified
paired cohort, not a universal GPU ranking or a statistical distribution claim.

Both arms keep 16 cold iterations/Focks (11 provisional + 5 target), four actual
cold VV10 sweeps, one warm Fock and zero new warm sweeps. A100 actual K
contractions decrease 32 -> 16 cold and 2 -> 1 warm; the two scientific radial
terms remain represented. RTX5090 keeps 32/2 K actions. The offline gates require
A100 cold ratio <0.98 and warm ratio <=1.03; RTX5090 cold/warm ratios <=1.03.

Independent signed raw Full/LR matrix oracles cover screened/unscreened sources,
original/displaced geometries, f shells, coefficient/operator/spin identity,
insufficient transient capacity, idempotence and unchanged retained source bytes.
Final 5090 qualification passes four fixtures with zero skips and memcheck with
zero errors. A100 independently passes the 96-AO composite probe. All exported
force components and energies pass both tight native paired gates and the
retained independent complete E/F oracle.

## Pilot and rejected results

The prototype image is retained separately, never relabeled as final:
native `d2172162b5a713a1d301ce4d0aefa100b1325a5d64f2ef3210f8cb17d2f9398c`,
ELF `68912cf573d10259a01c0b6c975833393925673742b43b5157a90e92f1742869`.

| Allocation | Raw LR cold | Explicit combined cold | Raw LR warm | Combined warm |
| --- | ---: | ---: | ---: | ---: |
| A100 / LSF226270 | 96.271742 | 89.834377 | 14.961188 | 13.908842 |
| RTX5090 / Slurm16623 | 179.062530 | 179.437876 | 13.704064 | 13.695091 |

Slurm16627 additionally passes both raw and composite f-shell fixtures. The
earlier Slurm16625 skipped due to missing compiler/cache configuration and is
not a passed gate. Source5573279078e2 built but failed vendor ownership before
GPU qualification; the final implementation uses a narrow shared Tensor
vector-accumulation provider. Slurm16630 failed host source authentication before
any oracle or endpoint because its source archive was not extracted. Superseded
builds and the corrected host/ownership harness failures carry no speedup claim.

## Offline replay

Run `python benchmarks/results/combined-range-source-20261107/replay.py`.
`evidence.json.xz` retains original complete endpoint records, scheduler and
source/ELF receipts, native reconstruction patches, oracle semantic values and
logs, without redundant trace timestamps or binary libraries. `summary.json`
is derived from that reader. Numerical/work/image/control and complete timing
gates remain active under Python `-O`; corruption tests exercise their failures.
Independent reference latency is deliberately not used for a new comparison.
