# Default CCSD(T)/CBS: convergence qualification and complete endpoint times

Measured source: `98531471e79d5ce31119ec8224adf5c1e5e989e2` plus `candidate5.patch.xz`. This campaign accepts numerical convergence on the declared cohort; it does not claim formal statistical performance promotion.

The default uses the former routine composite allowance, with two retained safeguards: SCF density at most `1e-11` and CCSD physical residual at most `1e-7`. The CCSD energy-change ceiling is `1e-8 Eh`. Bounded mixed auxiliary warmup ends before fresh FP64 CCSD/Lambda solves and independent checks; triples and force response remain FP64. The [user guide](../../../docs/user/ccsdt_cbs.md) owns the public profiles and controls.

## Complete default endpoints

Full DF / RI-JK, all-electron cc-pVDZ/cc-pVTZ MP2 extrapolation plus a cc-pVDZ DF-CCSD(T)−MP2 correction, with matching JKFIT auxiliaries. QM7 representatives use the pinned dataset and exact basis identities in the records. Every call constructs a fresh owner and returns host energy plus analytic forces. Basis snapshots are prepared outside timing; process/CUDA startup is separately retained. No reference density or oracle amplitudes enter production.

One Slurm RTX 5090 allocation, one math thread, 32 GiB numeric budget. Medians and ranges describe two calls per case, not a confidence interval or a speedup comparison. Numeric capacities do not measure peak VRAM. Only the default is displayed; internal stricter controls remain in raw validation records.

| QM7 case | Complete E+F median (s) | Two-call range (s) |
| --- | ---: | ---: |
| C1 | 3.202 | 3.200–3.204 |
| C2 | 8.452 | 8.444–8.460 |
| C4 | 66.647 | 66.597–66.697 |
| C6 | 465.540 | 465.394–465.686 |

Semantic work below is from the first repeat; both complete receipts, all group times, capacities and native counters are retained. Lambda counts exclude the separate independent transpose audit.

| Case | CCSD iterations | CCSD FP32 evaluations | Lambda FP32 + FP64 actions | Final CC residual |
| --- | ---: | ---: | ---: | ---: |
| C1 | 12 | 6 | 10 + 6 | 3.992e-10 |
| C2 | 13 | 6 | 11 + 6 | 1.461e-09 |
| C4 | 13 | 6 | 12 + 7 | 7.949e-09 |
| C6 | 13 | 6 | 12 + 7 | 1.343e-08 |

## Independent numerical qualification

Thirteen physical-basis populations include neutral, ionic, diffuse, noncovalent and stretched geometries. Eleven use full DF / RI-JK; stretched water and ammonium additionally use exact RHF with correlation-only DF. The diffuse hydroxide uses aug-cc-pVDZ/aug-cc-pVTZ and matching augmented JKFIT. Charges, coordinates, basis identities, coefficients and all native component observables are retained.

Independent PySCF references reconverge every Cartesian displacement at `1e-4` and `3e-5 Bohr`. Gates declared before measurement are component/composite energy `1e-7 Eh`, component/composite force maximum `3e-6 Eh/Bohr`, composite force RMS `1e-6 Eh/Bohr`, and two-step force stability `1e-6 Eh/Bohr`. Every displaced component must also stay within `0.01 Eh` of its central branch. Relative energies cover water/N2 stretch, dimer binding and the diffuse-basis shift; their gates sum the participating absolute energy allowances.

| Population | Default energy error (Eh) | Maximum force error (Eh/Bohr) |
| --- | ---: | ---: |
| full-df/water | 7.082e-09 | 9.995e-09 |
| full-df/water-stretched | 1.456e-08 | 2.267e-08 |
| full-df/ammonia | 6.034e-09 | 1.151e-08 |
| full-df/ammonium | 1.009e-10 | 1.729e-08 |
| full-df/hydroxide | 1.198e-09 | 3.801e-09 |
| full-df/hydroxide-diffuse | 2.845e-09 | 1.898e-08 |
| full-df/hydrogen-stretched | 4.691e-10 | 4.852e-08 |
| full-df/formaldehyde | 6.135e-10 | 3.718e-08 |
| full-df/nitrogen | 1.649e-09 | 1.397e-09 |
| full-df/nitrogen-stretched | 1.324e-08 | 1.441e-07 |
| full-df/water-dimer | 1.140e-09 | 2.507e-08 |
| correlation-df/water-stretched | 1.456e-08 | 1.458e-08 |
| correlation-df/ammonium | 1.002e-10 | 3.352e-08 |

The offline verifier passes 1549 observable gates, including energy-only parity, all component energies/forces, native strict controls, relative energies, oracle locality and stability, and final physical residual replay. In the 13-case independent cohort, maximum default composite energy/force errors are 1.46e-8 Eh / 1.45e-7 Eh/Bohr.

QM7 C2 additionally reuses the earlier campaign’s independent PySCF component energies and two-step directional forces after verifying identical geometry, basis and Hamiltonian inputs. All QM7 sizes are compared numerically with the retained strict FP64 endpoints. Those cross-revision controls are not independent full-force errors or timing baselines; no speed ratio is inferred from them.

Final integration: 71 passed, four opt-in large conventional CPU cases skipped; 11 native CCSD/Lambda symmetry, resource-refusal, early-switch and finite-exhaustion contracts passed. Host policy/renderer tests and the full source/documentation checks are recorded in the PR and provenance. The original strict resource-recovery oracle tests now explicitly select the strict profile and retain their original accuracy gates.

## Retained failed candidates and scientific scope

- **Mixed T2 handoff:** long-bond H2 initially failed Lambda pair-symmetry admission. The existing compiler-owned FP32-noise projection incorrectly depended on the optional spectator-pair contraction path. It now runs for admitted exactly symmetric initial states regardless of that optional schedule, before the fresh FP64 epoch. Asymmetric supplied states and final FP64 iterates are not projected; the Lambda gate is unchanged.
- **Reference density:** the same H2 geometry can fail physical HF export with the former relaxed density ceiling, for either FP64 or mixed CC execution. An energy-only tightening does not fix it; density `1e-11` does. Both routine names retain that ceiling. Initial failure receipts and isolated controls remain in `negative_trials`.
- **Stretched N2 roots:** native Hcore and PySCF MINAO reach different DZ RHF stationary branches, causing about 0.255 Eh / 0.187 Eh/Bohr composite mismatch. Both branches fail internal RHF stability. PySCF independently reproduces the native branch with its own Hcore guess; no native seeds are transferred. On that branch DIIS CCSD stalls, so a bounded SciPy Krylov solve of PySCF amplitude-update equations reaches update 2-norm below `1e-10`. Every displaced point is independently solved and checked for branch locality. This establishes same-branch numerical agreement, not the ground state or reliable dissociation physics.
- **Residual ceiling:** after retaining strict density, the initial routine residual ceiling gave stretched-N2 energy error `2.16e-7 Eh`, failing the declared `1e-7` gate. Tightening energy change alone did not help. The final `1e-7` physical-residual cap reduces the error to `1.32e-8 Eh`. The failed candidate remains retained; the gate was not relaxed.

These are static energy/force checks. They do not establish model/basis accuracy, automatic RHF stability/root selection, vibrational-frequency accuracy, force continuity or long-time energy conservation. Full per-iteration traces and measured peak memory were not collected, so formal #138 performance promotion is not asserted.

## Reproduction and source provenance

```bash
PYTHONPATH=python:. python benchmarks/results/cbs-routine-standard-20261025/verify.py \
  --output .artifacts/cbs-routine-replay.json
python benchmarks/results/cbs-routine-standard-20261025/render.py \
  --output .artifacts/cbs-defaults.svg
```

Candidate patches reconstruct dirty source on the named base. Candidate3 built the final native library with verified sccache 0.16.0, CUDA 12.9/SM120; candidates4/5 change Python convergence controls without changing any native-source hash. The final candidate5 patch contains the accepted benchmark driver. The earlier matched-reference run used candidate4 plus the exact compressed `oracle-driver.py.xz` overlay. Reference-only diagnostic scripts and all negative trials are retained. Later README/rendering/evidence changes do not change production sources.

Use the retained build/execute scripts as recorded environment examples, adapting checkout/toolchain paths. Run all real-device work through finite `srun` allocations, preserving Slurm visibility. `benchmarks.cbs_convergence` collects the 13-case populations; `experiment.py` collects fresh default QM7 calls. The N2 independent branch uses `--reference-guess 1e --reference-cc-solver krylov`. Cache statistics, binary/source hashes, allocation and test receipts are in `provenance.json.xz`; full local logs/binaries remain in ignored artifacts and the scheduled worker checkout.
