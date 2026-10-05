# Canonical CC denominator storage qualification

Same binary at `57a6e01cb`, explicit versus derived d2, on n2 through finite
Slurm allocations. Inputs/settings match the adjacent #1900/#1903 records:
ethane230 o=9,v=221,q=488; water7 o=5,v=2,q=7; DIIS8, Q tile8, 64 GiB budget.
No supplied reference orbitals or amplitudes enter either complete endpoint.
All times below are observations, not calibrated work-model estimates.

## Energy: job2298, two alternating pairs

GPU `GPU-54595246-dbdc-a633-dc38-7bd8eea3831a`, RTX PRO 6000, driver595.91.07.

| Median seconds | Ethane explicit | Derived | Water explicit | Derived |
| --- | ---: | ---: | ---: | ---: |
| Complete native | 294.623813 | 300.623919 | 1.189267 | 0.932187 |
| RHF | 138.765315 | 145.106371 | 0.871710 | 0.610399 |
| Source | 3.659558 | 3.630496 | 0.246749 | 0.250778 |
| CCSD | 147.255772 | 146.944156 | 0.063439 | 0.064157 |
| (T) energy | 4.943073 | 4.942800 | 0.007297 | 0.006783 |

**No endpoint speedup is established.** Ethane complete time increases6.000 s;
its unchanged RHF phase increases6.341 s. CCSD changes by only0.312 s (0.21%).
Water CCSD is0.000719 s slower in the median, while complete time decreases
because RHF varies. Retain the explicit selector for that observed small
latency-losing domain; these two pairs do not justify a universal size cutoff.
The derived path's demonstrated benefit is lower capacity and setup transfer.

| Ethane metric, bytes | Explicit | Derived |
| --- | ---: | ---: |
| Complete energy/CCSD numeric capacity | 4,476,290,544 | 4,412,996,440 |
| CCSD device allocation | 4,083,602,688 | 4,051,955,712 |
| Setup host-to-device logical bytes | 358,465,632 | 326,818,504 |

Device allocation decreases31,646,976 bytes and total numeric capacity decreases
63,294,104 bytes. The source-derived payload saving per host/device copy is
`8*(o²*v²-o-v)=31,647,128` bytes; alignment and actual vector capacities explain
the differences. This is not a measured memory-bandwidth figure.

All large observations retain20 iterations/38 evaluations. Energy spread is
9.948e-13 Eh and independent physical replay residual is at most5.317e-13.
Water energies are identical; its replay maximum is7.397e-12.

## Complete forces: job2299, one pair

Separate GPU `GPU-4b4be14f-ec84-6736-a7d8-968d62900c72`. Do not pool these times
with job2298 or infer force overhead by subtracting another GPU's energy time.

| Ethane phase, seconds | Explicit | Derived |
| --- | ---: | ---: |
| Complete native | 1380.676039 | 1399.116075 |
| RHF | 175.242224 | 193.902348 |
| Source | 3.624752 | 3.631972 |
| CCSD | 146.101215 | 145.990471 |
| (T) pullback and Fock response | 110.068092 | 110.111414 |
| Lambda | 272.905912 | 272.641639 |
| Source response | 3.713669 | 3.713243 |
| Orbital/nuclear response | 668.986004 | 669.094906 |

Again there is no complete-time win: RHF varies18.660 s while total varies
18.440 s. Complete force numeric capacity falls7,170,696,275→7,107,400,123 bytes.
The force-path triples timer includes energy, pullback and Fock response.
Within the derived force run, RHF/source/CCSD sum343.525 s; the other phases
include both (T) energy and response and cannot be labeled pure force overhead.

Maximum all-component force difference is4.370e-9 Eh/Bohr (water1.777e-15),
below the unchanged atol=rtol=3e-7 gate. Derived Lambda residual6.115e-13,
Z residual1.360e-13, stationarity1.088e-11 and translation7.126e-12 pass.
Lambda remains21 iterations and exact J/K remains28 actions.

Job2300 passes small-water independent PySCF all-coordinate central differences
at1e-4/3e-5 Bohr and energy/force/failure publication. Existing job2288 central
energies, for identical geometry/method, independently re-audit the large
derived forces at C0-z/H1-x and both steps. Maximum error3.074e-8 Eh/Bohr;
maximum gate ratio0.1023. Only accuracy is reused, never timing calibration.
This is not an all-coordinate independent large-force audit or benzene264
qualification, and it does not certify global reference stability.

## Work, correctness and reproduction

The compiler preserves the original FP64 grouping
`(ei-ea)+(ej-eb)-2*shift`; adding shifted singles is excluded. Admission uses
O(ov) single gaps plus the most negative doubles endpoint, with monotone FP64
addition establishing the complete threshold/overflow bound. The derived
representation replaces an O(o²v²) tensor with O(o+v) provenance. Each consumed
Jacobi denominator adds five scalar operations; its iteration-only count is
150,332,598 on ethane and is not a total endpoint FLOP count. CC contraction
summands remain44,063,877,939,352. Supplied/noncanonical and native CPU inputs
retain explicit d2. Unrequested phases are null in `summary.json`.

Build2292 passes. Host2295 passes four native admission/owner/Lambda tests and
18 CPU solver cases (five CUDA-only skips), including bitwise arithmetic,
near-threshold/overflow rejection, mixed/stale provenance, independent
determinant residuals, trajectory prefixes and exact/one-byte-short admission.
GPU2296 passes23 solver cases, the independent noninteracting Lambda and
capacity suite, and18 native Lambda/factor cases. Compiler/metadata/ownership
checks pass. CUDA scientific LOC is unchanged; runtime ownership delta is+13.

Build with the #1900 CUDA12.9.1/sm120 settings and ccache. In a finite n2
`main --gres=gpu:pro6000:1` allocation, preserve Slurm visibility and run:

```sh
./probe INPUT explicit.json 1 1 FORCES 1 8 8 8 0
./probe INPUT derived.json 1 1 FORCES 1 8 8 8 1
```

Raw records, binaries, hashes and logs remain at
`n2:/data/jzzeng/cc-1904-20261005/{endpoint-0-2298,endpoint-1-2299}/`;
ignored local copies are in `.artifacts/1904/`. Normalized observations,
medians and gate comparisons are retained in `summary.json`.
