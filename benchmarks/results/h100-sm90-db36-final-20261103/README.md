# Frozen DB36 H100 fixed30 final result

The original fixed30 contract passes: **30/30 scientific inputs, 180/180 qualified fresh processes, 720 complete energy/force endpoints, 21/30 wins**, and complete-set geometric mean **1.314955687500442**. Each win requires median of the three paired GPU4PySCF/native warm-median ratios >1.05 and all three ratios >1. No screening/profile votes or discarded final failures. DF13/14 and Direct8/16 are descriptive splits of the same thirty cases.

Measured source `db3657f08d983791c45b7d949cb9028fe0055e3f`, tree `681cefe8bd066a3fb973ab38e0ef856280625ed1`, core SHA256 `472bd0f9bceaa7917f7137c2cb543f0ebfc6c0b728ed024aeacc81b92c8c83a4`. H10080GB HBM3, MIGoff, UUID `GPU-4c4f204e-197c-ef5e-c109-fd4cac76b59c`; driver595.58.03, recorded700W limit, no locked-clock claim. Each endpoint uses one cold and three own-cold-seeded warm calls. The two originally excluded WB97M-V DF cases remain excluded by the unchanged original protocol SHA256 `2391b71fb49660f682f1b60056b2892d5516f01871d7479319f0be2cf925fdaa`.

## CPU reproduction

Use a full Git clone with the pinned measurement/helper commits available locally, unoptimized Python3.12+ and NumPy2.2.6. No CUDA/device, compiler, solver, profiler or network execution:

```sh
uv run --offline --no-env-file --no-project --python 3.12 --with numpy==2.2.6 python -m benchmarks.h100_db36_final_cohort
```

The wrapper composes the existing DB36 safety/profile and explicitly labelled source7013 DF/screen baseline readers, verifies all nine recomputed pre-dispatch proof hashes against the actual integration audit, and then reconstructs the unchanged final archive from its two sub-MiB parts. The original final unpacker and reader run without source or numerical edits. All1055 original members, all180 plans/raw/log/qualification records, frozen inputs, independent original oracle, runtime images, executed producers, consumed admissions and actual pre-dispatch host completion are retained. Existing recomputed proof bytes must equal the archived dispatch inputs before final replay.

Expected final proof SHA256 `dcc1bb5ab96f5c97e0a17dbb2878e16560b8d5c964165a82f82c29404bfd50e2`; original archive SHA256 `edd023ea51c70fc2c8b314af2524e205a76586796003a212c78234181f78c093` (1225848bytes). `cohort.json` binds every stored member and losslessly decoded manifest.

## All nine nonwins

| Input | Median of three paired ratios | All three ratios |
| --- | ---: | --- |
| hf-water-96-direct | 0.875605885833726 | 0.875605885833726, 0.869819716493947, 0.880612733067775 |
| hf-water-96-df | 0.813935476854574 | 0.813935476854574, 0.825498209242909, 0.811030142090473 |
| pbe-water-96-direct | 1.03389013702929 | 1.00384643669861, 1.03389013702929, 1.03672906624894 |
| pbe0-water-96-direct | 0.951589912032494 | 0.951589912032494, 0.971844775252796, 0.92847986396871 |
| b3lyp-water-96-direct | 0.930654562004316 | 0.930654562004316, 0.937430617969685, 0.929885388889689 |
| r2scan-3c-water-48-direct | 0.941141515299227 | 0.927444751020792, 0.941141515299227, 1.02163425935099 |
| r2scan-3c-water-96-direct | 0.619385263553276 | 0.619385263553276, 0.615433854788743, 0.621989401501086 |
| wb97m-v-water-48-direct | 0.825968691118648 | 0.825968691118648, 0.824194727699449, 0.836379998621819 |
| wb97m-v-water-96-direct | 0.88989161881942 | 0.88989161881942, 0.891891494786855, 0.884030697875135 |

These include both slower cases and PBE96Direct, which misses the >1.05 threshold. Every unfavorable sample remains in the denominator. The bounded descriptive diagnostics in `actual-result-audit.tar.xz` read the existing54 processes for these9cases, require no new GPU calls, and do not claim component-level causal attribution.

## Actual result review and limits

The lossless audit metadata archive preserves the actual completed report (SHA256 `ccc263820117665fc7e288c3255a9d99bb073bb1ed52e25edde4f70efd12a99e`), report seal, initial harness failure/correction, original task/response traces, input seal, host completion and all9 slow-item diagnostics. The reviewer independently recomputed every original energy/force comparison and all90 ratios. Requested reviewer gpt-6.1-sol/ultra; backend-reported identity unavailable; **same-family/provisional**, not independently verified external review. Metadata scripts are historical records and are not executed by the public reader. Its historical publication-pending fields and the pre-dispatch acceptance:false record remain unchanged. The host receipt authenticates the report, input seal and actual response.

The original numerical gates are energy/atom1e-5Hartree, maximum force1e-4Hartree/Bohr and RMS force3e-5Hartree/Bohr. Against the original oracle, the review reports maximum energy/atom8.204134796111854e-6, maximum force5.688554111138444e-5, and RMS2.5819321785223374e-5.

Acceptance is limited to this frozen source, water48/96 matrix, original fit/basis/grids, policy and measured H100. It does not qualify newer main commits, other GPUs, general molecules or UKS, and does not establish H100-versus-RTX5090 rankings. Native33-artifact build/admission hashes and post-owner mappings do not prove that every AOT companion was mapped during every timed call. The intrusive Nsight capture and posthoc CPU observer are not clean timing votes or a before-importer independent witness.

Nine owned old CD2B guard packets move to exact recovery from already merged Git, under unchanged storage limits; all original negative outcomes and the old fourteen proof-file chain/final20/30 remain reproducible. No other-owner science is retired, no release/external backup is created, and no GPU validation is repeated for publication.
