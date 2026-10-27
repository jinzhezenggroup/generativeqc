# Comparison conditions

Geometry and basis differ from the water README cohort; nothing else in the
endpoint protocol does. Every fresh process measures one cold and three warm
complete energy-and-force calls, each warm call starting from that engine's own
frozen cold density.

| Condition | GenerativeQC | GPU4PySCF 1.8.1 |
| --- | --- | --- |
| Geometry | `remdesivir.xyz`, PubChem CID 121304016 3D conformer, converted once to Bohr with PySCF's `BOHR` | Same coordinates, charge 0, spin 0 |
| Orbital basis | Spherical def2-SVP (def2-TZVP cohort); r²SCAN-3c uses its compiled def2-mTZVPP basis | Same |
| r²SCAN-3c corrections | Native D4 + gCP, included in energy, forces and timing | Independent D4 + gCP (dftd4 4.3.0 / gcp), included |
| DF auxiliary | Frozen once from this geometry's elements (H, C, N, O, P) by `prepare_inputs` and reused by every DF cell | Stock `density_fit()` auxiliary |
| XC/VV10 grids | `#2786` default level-3 Treutler/Lebedev, NWChem pruning; no `grid_overrides` | Installed defaults |
| Moving-grid forces | Full coordinate and partition response | Full response, `grid_response=True` |
| Convergence | Framework defaults (energy change 1e-9 Eh, density RMS 1e-5; ωB97M-V 1e-10 / 1e-8) | Default energy change 1e-9 Eh |
| Iteration cap / DIIS history | 100 / 8 | Defaults 50 / 8 |
| Cold call | Constructor + prepare + SCF + forces + host export | Same boundary, `MINAO` initial guess |
| Warm call | Same owner; frozen engine-local cold density; 3 repeats | Same; 3 repeats |
| Threads | `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`, `--cpus-per-task=2` | Same |
| Point timeout | 5,400 s per endpoint (`timeout --kill-after=30`) | Same |
| Accelerator | NVIDIA GeForce RTX 5090, 33,668,988,928 bytes | Same device, sequential cells |

## Differences from the water README cohort

- **System.** 77 atoms / 767 AOs (def2-SVP) and 1,518 AOs (def2-TZVP) versus the
  3/12/48/96-atom water clusters. The molecule contains phosphorus and `spd`
  shells, so shell composition rather than atom count drives the Direct cost.
- **No Fast arm.** The public Fast precision policy is excluded by request; only
  Standard and GPU4PySCF are measured here.
- **Two bases.** def2-SVP covers all eight methods; def2-TZVP covers seven
  (r²SCAN-3c is measured only in its own def2-mTZVPP basis).
- **Capacity-limited cells.** Unlike the water cohort, several endpoints have no
  GenerativeQC measurement; the definitional failures are retained in
  `capacity-evidence.json` instead of being replaced by a smoke or profiled run.

## What this cohort does not establish

- No independent accuracy gate. The reported differences are cross-engine
  differences, the same qualification the water cohort uses for its own
  published deltas.
- One fresh process per population, as in the water cohort. The four Direct cells
  repeated in `repeat.json.xz` agree within 0.3 % cold and to the last digit
  warm, but they are a reproducibility check rather than confidence intervals.
- No claim that the Direct ordering generalises. On water-96 Direct favours
  GenerativeQC; on this geometry it does not, and the retained data support only
  the system-specific statement.
