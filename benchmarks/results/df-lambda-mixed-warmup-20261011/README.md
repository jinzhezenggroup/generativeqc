# DF Lambda mixed warm start: scoped numerical qualification and exploratory timing

Measured candidate: clean `04cdb116bcd377788ace975e65fd08f02becf4c0`, tree
`8b383570dc8dfca271831c678a8fb10ff73cbe85`. Baseline is the frozen prior
CCSD-mixed/Lambda-FP64 PR implementation at `f0d361308e1340fd9361286de2c40118526ac068`;
native binaries have the same executable sources as `4b1b9487`.

Slurm job **7560**, node1/main, one RTX 5090, four CPUs, finite `01:30:00`.
All calls are fresh independent processes, compiled executable reused, no
electronic state supplied. Same ethane input: 230 AO, 488 auxiliary functions,
9 occupied and 221 virtual orbitals; conventional exact RHF reference and
correlation-only DF-CCSD(T). (T) and final derivatives remain FP64.

## One completed cold triplet, not five-pair qualification

| Observable | Prior PR, FP64 Q32 | FP64 Q16 ablation | Mixed warm start, strict Q16 |
| --- | ---: | ---: | ---: |
| Complete native E+F, seconds | 293.210665477 | 305.645875821 | 289.226577167 |
| Full process wall, seconds | 293.59 | 306.04 | 289.60 |
| Lambda, seconds | 78.710987019 | 91.193064252 | 74.766767397 |
| Lambda owned device bytes | 10,324,999,872 | 6,212,893,888 | 6,212,893,888 |
| Sampled total-device peak, MiB | 27,547 | 27,547 | 27,547 |
| Lambda iterations/actions | 21/22 | 21/22 | 22/25 |

This single triplet observes about 1.36% less complete time and 5.01% less
Lambda time versus the prior PR. **It does not establish a statistically
qualified or general speedup.** Do not pool these calls with historical master
measurements. The planned repetition stops after this triplet to integrate
remote master; the user then explicitly requests no repeated validation.

The 4,112,105,984-byte component storage reduction comes from the smaller Q
tile: the strict-only ablation has identical ownership but slower execution.
FP32 warmup offsets that tiling cost; it does not itself explain the storage
reduction. Whole-endpoint sampled peak is unchanged, and sampled occupancy is
**not** an exact allocator peak. No extra owned device allocation holds warmup
inputs: they borrow an epoch-limited tail of existing strict scratch.

Mixed warmup uses 11 iterations/12 actions, then a new strict FP64 solve uses
11 iterations/13 actions. It executes 10,044 FP32 GEMMs, 12,436,723,043,784 FP32
GEMM summands and converts 219,266,918 elements. Total Lambda contraction terms
increase 37,901,303,184,171 → 41,404,915,812,540; packing output bytes change
1,862,138,223,520 → 1,788,792,470,416. This is not reduced total semantic work.

## Numerical acceptance and version boundary

On the frozen measured candidate, 64 targeted tests pass (three unrelated
(T)-mixed tests deselected); 167 CPU tests pass with 27 expected GPU skips.
H2/water/LiH independent oracle energies and two-step directional force finite
differences pass. Water complete-E+F memcheck reports zero errors and zero
leaked bytes. Paired ethane energy is unchanged; maximum mixed-versus-baseline
force-component error is `5.437428285404167e-12` Eh/Bohr. Original strict CCSD,
Lambda, Z and stationarity certificates remain mandatory; no discarded JK
attempt occurs.

Remote master integration and the unused-Lambda-warmup-option admission guard
follow this measured source. **The final publishing code is not remeasured or
requalified by these records**, per the user's instruction. Added admission
regression cases are left to CI; earlier receipts are not presented as results
for those new cases. The option remains experimental and default-off.

## Retention and reproduction

- `evidence.json.gz`: measured source/hardware/binary identities, oracle values,
  exact error gates, receipts, work/storage values and explicit limitations.
- `samples.json.gz`: every original endpoint JSON value, losslessly factored
  into per-mode defaults and per-call overrides; original raw JSON hashes and
  complete GNU-time receipts. Reconstruct with `defaults[mode] | overrides[name]`.
- `occupancy.json.gz`: every original timestamp and memory-used observation;
  first timestamp plus exact adjacent millisecond deltas, without resampling.
  Unused clock/power telemetry is omitted, original CSV hashes retained.
- `measure.sh.gz`: exact historical runner. It plans five alternating triplets;
  only the first completes. The original water fixture path names job 7554;
  recreate that input from `load_fixture("water")` and `write_input`, or use the
  same newly generated water input in the runner's pytest directory.

Shared ethane input: `../df-lambda-gemm-20261004/ethane230.input`, SHA-256
`9428f2b1d1db38ffa374387705099e8d57fde98e0e068faed2861b04604a1c6e`.
Build frozen baseline and candidate with verified ccache, CUDA 12.9.86, GCC
11.4.0, Release and sm120. Set up the runner's baseline/candidate layout with
per-mode libraries. Future reproduction uses a finite Slurm allocation on
node1 via SSH alias n1, preserving scheduler-assigned visibility; it is not
permission to bypass Slurm or use maintenance node n3.

The publication accepts the **recorded numerical scope**, not a performance
promotion or a claim about the later integrated source. See the
[decision note](../../../.agents/notes/implemented/performance/2026-10-11-df-lambda-mixed-warmup.md).
