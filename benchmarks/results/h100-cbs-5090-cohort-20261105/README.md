# H100 CBS qualification using the published RTX 5090 validation method

Refs #3003. The original **13 physical-basis populations and all seven C1–C7 complete endpoints PASS**. The H100 routine-cohort reader replays the original scientific gates and passes 2348 observable checks. An additional offline comparison passes all 180 carried independent alkane energy/directional gates. No numerical threshold was relaxed and no new independent-oracle solve was performed.

Measured H100 source: `6707989cce434b0ca4ef43f9920d5f91525f30c1`, tree `f9cb8c86cf88eda9ce7f7e858d0d7108c3baffdf`, core SHA-256 `3bce06074b2cdff13d12a6231fe9b9b4948740527d406b925c931ddb982d3691`. The original source-bound SM90 build is reused from [the reference closeout](../cbs-reference-closeout-20261104/README.md), pinned at `9afbe5aba7cdb946e3148a40ca1c1b8b8877a6c1`. This publication adds measurements and a CPU-only reader; it changes no production implementation.

## Exact scientific inputs and independent verification

The [5090 routine publication](../cbs-routine-standard-20261025/README.md) supplies the exact geometries, charges, orbital/JKFIT basis identities (cc-pVDZ/cc-pVTZ, with aug-cc-pVDZ/aug-cc-pVTZ and augmented JKFIT for diffuse hydroxide), all-electron extrapolation and correction model, coefficients, independent PySCF raw reference records, and declared gates. Its records are pinned at `4370d6c4b1bf4033818f31a9fcc622d3045a33bf`, SHA-256 `03b2722d016c13daaf6025a021de18e0ac333ebbb6efe94943c57748a57f5d3a`. Every prepared basis identity equals that input before native execution. Eleven small populations use full DF/RI-JK and two use exact RHF with correlation-only DF.

Each of the 13 populations has two fresh-owner default E+analytic-force calls, two tight calls, and one energy-only call. The frozen reader reconstructs independent full Cartesian finite differences from both signs at `1e-4` and `3e-5` Bohr, checks all five constituents and the composite, displaced-energy locality, two-step stability, relative energies, energy-only parity, component reconstruction, translation and physical final CC residuals. Original gates remain energy `1e-7 Eh`, force maximum `3e-6 Eh/Bohr`, composite force RMS and FD stability `1e-6 Eh/Bohr`, local constituent energy distance `0.01 Eh`; energy-only parity is `1e-8 Eh`. Returned convergence budgets and QM7 constituent identities are checked explicitly. Full-vector oracle coverage is limited to these 13 populations.

Maximum default composite errors over the 13 populations and both complete repeats are **1.455960330e-08 Eh**, **4.852432929e-08 Eh/Bohr maximum force**, and **3.373877551e-08 Eh/Bohr force RMS**.

| Population | Energy error (Eh) | Force maximum (Eh/Bohr) | Force RMS (Eh/Bohr) |
| --- | ---: | ---: | ---: |
| full-df--water | 7.082e-09 | 9.996e-09 | 5.021e-09 |
| full-df--water-stretched | 1.456e-08 | 2.266e-08 | 1.011e-08 |
| full-df--ammonia | 6.032e-09 | 1.151e-08 | 6.401e-09 |
| full-df--ammonium | 1.007e-10 | 1.729e-08 | 9.176e-09 |
| full-df--hydroxide | 1.198e-09 | 3.800e-09 | 1.866e-09 |
| full-df--formaldehyde | 6.131e-10 | 3.718e-08 | 1.483e-08 |
| full-df--nitrogen | 1.649e-09 | 1.394e-09 | 8.049e-10 |
| full-df--water-dimer | 1.140e-09 | 2.507e-08 | 9.004e-09 |
| full-df--hydroxide-diffuse | 2.846e-09 | 1.898e-08 | 8.928e-09 |
| full-df--hydrogen-stretched | 1.155e-10 | 4.852e-08 | 3.374e-08 |
| correlation-df--water-stretched | 1.454e-08 | 1.455e-08 | 8.227e-09 |
| correlation-df--ammonium | 1.004e-10 | 3.352e-08 | 1.662e-08 |
| full-df--nitrogen-stretched | 5.581e-09 | 1.190e-08 | 6.730e-09 |

The original C2 independent energy/directional summary is also checked at both steps. Its historical individual displaced energies remain unavailable; this delivery does not recover them. The separately preserved [six-center independent energies](../cbs-independent-energy-20261023/README.md) and [C4/C7 two-step directional reference](../cbs-independent-forces-20261024/README.md) are additionally compared with all three H100 samples per alkane, using their unchanged `5e-8 Eh` and `5e-6 Eh/Bohr` gates. The additional reader verifies raw CPU receipts, outcomes, component reconstruction, basis identities, source and log hashes and exact shifted geometries. It uses the exact pinned measured direction, avoiding host-BLAS rounding in regenerating its normalization. These extra comparisons are postprocessing and provide no independent full-vector alkane reference.

## Complete default H100 endpoints

Every timed call constructs a new outer CBS calculator and returns host energy plus analytic forces. Outer construction and inner preparation are included. Prepared basis tables, imports, process/CUDA startup and compilation are outside the balanced default samples. Because each scientific input runs in a separate process here, one complete startup call per process is retained separately. The default profile permits bounded mixed auxiliary preconvergence before fresh FP64 physical solves and checks; tight controls are separate numerical controls.

| QM7 case | Atoms | Default median (s) | Two-call range (s) | CCSD iterations, first default | Final CC residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 5 | 1.142 | 1.126–1.158 | 12 | 3.991e-10 |
| C2 | 8 | 3.190 | 3.178–3.202 | 13 | 1.461e-09 |
| C3 | 11 | 8.957 | 8.936–8.979 | 13 | 3.758e-09 |
| C4 | 14 | 23.207 | 23.187–23.227 | 13 | 7.949e-09 |
| C5 | 17 | 55.419 | 55.364–55.474 | 13 | 1.074e-08 |
| C6 | 20 | 125.484 | 125.390–125.579 | 13 | 1.343e-08 |
| C7 | 23 | 269.211 | 268.101–270.321 | 13 | 1.517e-08 |

Two calls provide descriptive medians/ranges, not confidence intervals or statistical promotion. Raw receipts retain all constituents, execution groups, capacities, CC/Lambda work, convergence and startup observations. Numeric capacities are not measured peak VRAM.

Actual device: `NVIDIA H100 80GB HBM3, GPU-77a0d6b8-162e-6299-38ee-49757a359174, 595.58.03, 9.0`. Image: `generativeqc-science-py311-cuda129:20261015`; the submitted command is cross-checked against the platform command receipt. One qz H100 Job allocated **1 GPU, 10 CPU, 200 GiB, 8 GiB shared memory**, with one math thread and a two-hour ceiling. Requested priority 4 was capped to recorded NORMAL/20. The unchanged numeric budgets are 32 GiB for QM7 and 8 GiB for the 13-case cohort. `CUBLAS_WORKSPACE_CONFIG=:16:8` is explicitly enforced before imports; it is part of this measured configuration. Each population passed its numerical gate before the next population was dispatched; the full aggregate verifier then ran before terminal completion. There were no automatic retries. The terminal platform receipt confirms successful container release.

Total retained work is **73 primary complete E+F calls plus 20 separately recorded startup E+F calls and 13 energy-only calls**. The public reader requires finite physical-shape forces for every E+F composite and constituent before counting it, including the second tight repeat, and requires all force fields absent for energy-only calls. A missing-force corruption control found a hole in the original prepared verifier; this additional acceptance guard closes it without altering the executed GPU producers or observations. Actual preflight/producer/input/core identities are bound to authenticated platform logs and both running and terminal allocation records.

## Scope and reproduction

The original 5090 measurements used source `98531471e79d5ce31119ec8224adf5c1e5e989e2` plus the recorded candidate5 patch. These H100 and 5090 cohorts use different source revisions and architectures. Their wall times do not isolate a hardware effect and are not a paired speedup experiment. The independent CBS comparator here is CPU PySCF for accuracy; there is no matched GPU4PySCF CBS timing denominator in either delivery. Existing HF/DFT GPU4PySCF comparisons retain their separate scope.

Retained scientific limitations carry over: stretched N2 compares the same internally unstable RHF branch, with independently solved reference displacements and locality checks. Agreement does not establish a ground state, dissociation accuracy, automatic root selection, model/basis accuracy, vibrational accuracy, force continuity or long-time energy conservation. Same-family fresh-context review is provisional, not external peer review.

Use unoptimized Python 3.11+ with NumPy and a full Git clone containing all pinned source and reference commits. The command executes retained numerical readers only, with no native solver, CUDA, compiler or network work:

```sh
python benchmarks/results/h100-cbs-5090-cohort-20261105/replay.py
# Optional exact recovery into a new, nonexistent directory:
python benchmarks/results/h100-cbs-5090-cohort-20261105/replay.py --recover /new/empty/path
```

The raw observations, thresholds, verdicts and nondirectional report fields remain exact. Across host BLAS implementations, only recomputed C2 directional error scalars and C4/C7 directional dot/error scalars may differ within the explicit FP64 reduction envelope `2*gamma(2*n)*sum(abs(F*d))`, where `gamma(k)=k*eps/(1-k*eps)`. This affects comparison of recomputed reports only; every scientific gate is evaluated unchanged before that comparison. The initial cross-host replay retained a maximum C2 report difference of `1.4094628242311558e-18`; its original Linux report remains unmodified.

The reader verifies every publication/capsule member, fully replays the prior source-bound CBS capsule, binds all new observations to the same core/source and original scientific inputs, reproduces original and additional gates, and checks declared call counts. Raw producers and prepared inputs are recoverable. New real-device execution requires fresh task/job/output names and adapted absolute paths; historical receipts and source directories remain immutable.
