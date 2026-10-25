# Strict CBS preconvergence, independently of Fast convergence

Measured source: `7b3dc64f96cee5b805f1e8c958cbe6f22d13336f` plus `candidate2.patch.xz`. Dirty-source reconstruction and both candidate libraries' identities are retained; these are new device measurements, not relabelled historical curves.

CUDA full DF-HF/RI-JK CCSD(T)/CBS: cc-pVDZ/cc-pVTZ MP2 extrapolation, cc-pVDZ CCSD(T)−MP2 correction, matching JKFIT auxiliaries, all electrons and unchanged basis/geometry. Complete fresh-calculator energy + analytic-force endpoints on Slurm RTX 5090 cards, one math thread and 32 GiB numeric budgets. Process/CUDA startup is separately retained; no converged density or reference seed is supplied to a measured call. Basis snapshots are prepared outside timing.

## Complete endpoints

Seconds below are medians of two alternating calls per arm for C1/C2/C4, and a single call per arm for C6. Each comparison stays on one card and one binary. All individual times, full precision energies/forces and semantic work remain in `samples.json.xz`. These observations do not establish a universal or statistically certified speedup.

| QM7 case | Strict FP64 control | New Standard (auto) | Fast + FP64 | Fast (auto) | Standard time reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 3.280 | 3.298 | 3.177 | 3.243 | No demonstrated gain; same FP64 work |
| C2 | 9.373 | 9.072 | 8.745 | 8.476 | 3.2% |
| C4 | 86.066 | 77.210 | 73.658 | 67.153 | 10.3% |
| C6 | 635.207 | 560.766 | 542.374 | 466.167 | 11.7% |

Automatic strict E+F warmup starts at 9 occupied / 48 virtual orbitals. C1 remains FP64. Energy-only admission retains the existing 9/128 domain: C2 remains FP64; C6 changes from 141.687 s to 131.974 s (two calls per arm). Explicit `preconvergence="mixed"` remains available for controlled experiments. Fast retains its prior mixed request when relaxed convergence is admitted.

For example C4 still gains another 13.0% by selecting Fast after the shared Standard optimization; C6 gains another 16.9% in its single-call pair. The four cells distinguish this remaining convergence benefit from reusable arithmetic scheduling.

## Work and accuracy

| Case / controls | CCSD iterations | CCSD FP32 evaluations | Lambda FP32 + FP64 actions | Lambda seconds |
| --- | ---: | ---: | ---: | ---: |
| C4 / standard + fp64 | 20 | 0 | 0 + 23 | 40.478 |
| C4 / standard + auto | 21 | 11 | 12 + 13 | 32.392 |
| C4 / fast + fp64 | 12 | 0 | 0 + 16 | 31.558 |
| C4 / fast + auto | 13 | 6 | 12 + 7 | 24.924 |
| C6 / standard + fp64 | 21 | 0 | 0 + 23 | 322.465 |
| C6 / standard + auto | 22 | 11 | 12 + 14 | 257.357 |
| C6 / fast + fp64 | 12 | 0 | 0 + 17 | 259.911 |
| C6 / fast + auto | 13 | 6 | 12 + 7 | 185.964 |

Standard CCSD energy/residual limits stay at 1e-12/1e-10. Lambda and orbital-response controls, FP64 independent residual and transpose checks, triples and analytic derivatives are unchanged. The extra strict CC iteration can be offset by cheaper auxiliary contractions, particularly in Lambda. Q32 remains the admitted Lambda ceiling. Counts exclude the separate independent Lambda audit.

- The offline verifier recomputes 646 component/composite observable gates and 20 timing populations, including unchanged strict limits, actual arithmetic, final residuals, full-vector controls and all failures.
- Fresh independent C2 PySCF energies and two directional derivatives gate every component and the composite at 1e-8 Eh / 3e-6 Eh/bohr. Across both convergence policies, maximum errors are 3.11e-9 Eh and 1.07e-8 Eh/bohr. C4/C6 full-force comparisons are FP64 controls, not independent errors.
- New Standard versus strict FP64 has maximum composite force differences below 6.2e-13 Eh/bohr in this cohort. Separate water tests compare every Cartesian component at original/moved geometries, both HF reference models, replays and energy-only calls. The compact STO-3G/def2-SVP water pair checks algebra, not a physical CBS limit.
- 47 integration tests and five existing native CCSD/Lambda refusal/exhaustion contracts pass. Combined native/oracle tests use a 24 GiB allocation and PYSCF_MAX_MEMORY=12000, avoiding PySCF's default 4 GB process-RSS ceiling without relaxing convergence.
- C2 complete calls succeed and retain strict parity at 1/2 GiB numeric budgets. All 16–512 MiB failures remain recorded; they are not successful fallback timings. Actual optional-provider refusal in the native probes retains identical FP64 outputs.
- New automatic Python calls against the clean master library retain the legacy strict route with an explicit capability fallback. Its C2 energy equals the candidate FP64 control; forces differ by less than 2e-14 Eh/bohr.

Numeric capacities and owned-device counts are retained; they are not measured peak VRAM. Formal #138 performance promotion (including complete iteration traces and peak-memory measurement) was not performed. The publication accepts the scoped numerical checks and retains descriptive endpoint timings.

## Negative results and provenance

The first exploratory C1 FP64 call included process startup (4.88 s versus subsequent 3.27 s); this cannot substantiate a mixed speedup. Subsequent explicit mixed calls gave no useful C1 benefit, motivating automatic FP64 selection. The entire pilot and its source patch remain retained.

An initial candidate build missed changed files because copied mtimes preceded clean-baseline objects. A missing capability rejected that library before accepted timing; current-mtime extraction, rebuilding and a capability smoke test corrected it. Native builds used verified sccache 0.16.0; generated launcher commands were checked and cache statistics are retained. Full build logs and binaries remain in ignored `.artifacts/cbs-standard/` and `n1:/data/jzzeng/cbs-standard-20261025/`.

## Reproduction

Offline numerical/work/timing verification (no GPU):

```bash
PYTHONPATH=python:. python benchmarks/results/cbs-standard-preconvergence-20261025/verify.py \
  --output .artifacts/cbs-standard-replay.json
```

The retained build and execution scripts record the measured node/toolchain paths. To rerun native endpoints, reconstruct master plus the relevant decompressed patch in an isolated checkout, build with the recorded CUDA architecture and verified cache launcher, then use finite `srun` allocations and `experiment.py`. Preserve Slurm device visibility. `samples.json.xz` includes every geometry/protocol; the original QM7 input is checksum-pinned in `benchmarks.readme_cbs`. `reference.json.xz` retains the independent numerical inputs and results.

Source patches include the exact measured Python/native changes. Later documentation and evidence edits do not change those runtime sources. The archived benchmark driver and scripts are provenance/reproduction assets, not a new production backend.
