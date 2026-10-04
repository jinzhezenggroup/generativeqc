# Restricted DIIS history packing qualification

These frozen observations use production source `3c28ca85c` and one binary,
with full versus packed histories selected explicitly. They predate the
parent's prepared-contraction integration; they are not timings of that later
tree. Current integration qualification is retained separately when complete.

All compilation, tests and postprocessing run on n2 in finite Slurm allocations,
with ccache. Inputs are native molecular water7 (o=5,v=2,q=7) and ethane230
(o=9,v=221,q=488), DIIS8, canonical derived denominators and a64 GiB ordinary
budget. No supplied orbitals/amplitudes or production reference oracle is used.
`summary.json` retains binary/input/source hashes, every observation and gates.

## Complete energy: job2327, two alternating pairs

GPU `GPU-4b4be14f-ec84-6736-a7d8-968d62900c72`, RTX PRO 6000, driver595.91.07.

| Median seconds | Ethane full | Packed | Water full | Packed |
| --- | ---: | ---: | ---: | ---: |
| Complete native | 282.446577 | 288.996095 | 1.335240 | 0.916127 |
| RHF | 128.064087 | 134.538964 | 1.019683 | 0.587731 |
| Source | 3.601241 | 3.603353 | 0.245080 | 0.257507 |
| CCSD | 145.884701 | 145.958838 | 0.063675 | 0.064003 |
| (T) energy | 4.896455 | 4.894845 | 0.006730 | 0.006820 |
| DIIS, inside CCSD | 0.061125 | 0.050456 | 0.001042 | 0.001190 |

There is no complete endpoint speedup. Unchanged RHF accounts for most of the
total variation. Large DIIS saves about11 ms, but full CCSD does not improve;
the initial full-tensor symmetry scan and packing audits are included. Tiny
water DIIS and CCSD regress slightly. Packing therefore stays opt-in, with full
history retained as the ordinary latency choice.

| Ethane capacity, bytes | Full | Packed |
| --- | ---: | ---: |
| Complete energy/CCSD numeric capacity | 4,412,996,440 | 4,161,912,920 |
| CCSD device allocation | 4,051,955,712 | 3,800,872,192 |
| Amplitude/error history arena | 506,638,336 | 253,573,632 |

The histories share one separate allocation; the last row reports its complete
amplitude/error payload. Metric weights and scalar audits remain charged in the
ordinary arena. Net device/complete-energy saving is251,083,520 bytes, including
alignment. The source-derived nominal payload saving is251,083,404 bytes; the
small difference is alignment, not uncounted conversion storage.

All large runs retain20 iterations and38 evaluations. Packing remains active
without refusal. All-repeat energy spread is7.816e-13 Eh, maximum independent
physical replay residual5.317e-13. Water energies are identical and replay is
at most7.397e-12.

## Complete force: job2328, one pair

GPU `GPU-cacd0aaf-c80f-41eb-d7a2-4c3a5970f282`. This is a separate allocation;
do not subtract its force time from another GPU's energy time.

| Ethane phase, seconds | Full | Packed |
| --- | ---: | ---: |
| Complete native | 1325.971806 | 1329.459464 |
| RHF | 123.954689 | 130.872747 |
| Source | 3.561308 | 3.602388 |
| CCSD | 145.288653 | 145.009108 |
| (T) pullback and Fock response | 109.762143 | 109.262149 |
| Lambda | 272.176998 | 270.900901 |
| Source response | 3.720943 | 3.695852 |
| Orbital/nuclear response | 667.479641 | 666.088658 |

CCSD phase capacity decreases by the same251,083,520 bytes. **Complete force
peak remains7,107,400,123 bytes**, because later response phases dominate it.
The triples force timer includes energy and response; it is not pure force
overhead. No complete-time force improvement is established.

Maximum all-component force difference is2.348e-9 Eh/Bohr (water2.221e-15),
within unchanged atol=rtol=3e-7 gates. Translation is at most9.424e-12. Lambda,
Z residual and stationarity pass. Job2326 independently qualifies small-water
all nuclear coordinates at1e-4 and3e-5 Bohr, plus failure publication, while
requiring packing to remain active. Existing independent central energies from
job2288 re-audit C0-z/H1-x at both steps: maximum error2.998e-8 Eh/Bohr and gate
ratio0.09976. Only accuracy is reused. This is not an all-coordinate independent
large-force audit or global reference-stability certificate.

## Complete constrained-budget admission: job2329

GPU `GPU-1979d573-6626-6718-5e52-11741b0a9575`. Both paths use scalar/one-Q CCSD,
so an optional matrix/tile fallback cannot disguise the admission difference.
With the measured packed complete budget of2,735,193,944 bytes, full history is
rejected with `RCCSD CUDA resident state exceeds correlation memory budget`.
Packed history completes the same native energy endpoint in452.408 s with the
original convergence/replay gates. The ample-budget packed precursor is464.140 s.
This demonstrates capacity admission, not a speedup against a failed run.

The rejected process spends136.72 s before refusal. Its unpublished phase work,
native endpoint time and energy are unavailable and remain null, not zero.

## Work and validation

For m=ov, F=m+m² and C=m+m(m+1)/2, history payload changes
`16*h*F -> 16*h*C+C` bytes before alignment. The compiler owns the simultaneous
pair involution and one/two orbit metric. Input symmetry is bitwise; iteration
projection only admits the documented FP64 rounding bound. Excessive asymmetry
restarts full history, or bounded Jacobi if the replacement cannot fit.

With L the sum of inserted live row counts, Gram summands change
`L*F -> L*C`, with `L*C` additional weight multiplications. On ethane these are
490,805,640→245,649,456 dot summands, plus245,649,456 weight multiplies. Full
output extrapolation stays486,847,530 summands. CC residual contraction work is
unchanged. Packing/conversion reports2,412,824,076 logical bytes; these are not
measured bus bytes. Task/kernel counts and summands are not total FLOPs.

Host2323 passes10 cases (one explicit GPU skip). Build2325 and GPU2326 pass29
solver/generated-consumer cases, two Gram/ring cases and the two independent
complete-force FD/publication cases. Tests cover orbit edges, actual weighted
CUDA history kernels, independent determinant replay, trajectory prefixes,
supplied asymmetry, runtime refusal and exact/one-byte-short capacity. Resource
tests inject every packed/full owner setup failure. Host2332 repairs stale
interface-extraction fixtures without weakening their no-work assertions.

Reproduce the frozen binary with the #1900 CUDA12.9.1/sm120 Release settings and
ccache, in n2 `main --gres=gpu:pro6000:1` with finite time and Slurm visibility:

```sh
./probe INPUT full.json   1 1 FORCES 1 8 8 8 1 0
./probe INPUT packed.json 1 1 FORCES 1 8 8 8 1 1
# Capacity comparison uses matrix=0 and CCSD tile=1 in both paths.
./probe INPUT packed-scalar.json 1 0 0 1 8 8 1 1 1
```

Raw records remain at
`n2:/data/jzzeng/cc-1902-20261005/{endpoint-0-2327,endpoint-1-2328,budget-2329}/`.
The representation/rounding/fallback rationale is in
[the Agent Note](../../../.agents/notes/implemented/performance/2026-10-05-rccsd-packed-diis.md).

## Prepared-provider integration

Production `1b63282e1` is independently qualified by build 2334, host 2339,
real-device solver/small all-coordinate FD 2341, shared Lambda/factor 2348 and
complete energy/force/budget jobs 2342–2344. Report 2346 passes. Records are in
`prepared-provider-summary.json`; its source-manifest checksum binds this version.
It predates the subsequent retained-RHF ownership integration.

Ethane energy CCSD medians are 145.659721 / 145.552978 s (full / packed), and
complete medians 303.653926 / 331.683467 s. RHF medians are 149.477555 /
177.628133 s. On the separate force allocation, complete time is 1332.151061 /
1314.441975 s, with RHF 144.472181 / 126.654633 s. The apparent force difference
is explained by RHF variation; there is still no demonstrated packing endpoint
speedup. Each comparison stays within its own GPU UUID and source version.

CCSD device bytes remain 4,051,955,712 / 3,800,872,192. Complete energy capacity
is 4,413,124,536 / 4,162,041,016 bytes, including new prepared descriptors.
Complete force capacity remains identical at 7,107,772,763 bytes. At the same
2,735,193,944-byte scalar-schedule budget, full history is refused and packed
history completes (447.392 s); unpublished rejected-run work remains null.
Large maximum force difference is 5.309e-9 Eh/Bohr and energy spread at most
3.269e-13 Eh. All residual/stationarity and existing independent two-coordinate,
two-step FD re-audits pass. This adds no all-coordinate independent large audit.

Latest parent `c7486bf2e` is composed separately as `d3548f1fd`. It passes the
expanded 77-case host admission/lifetime suite in 2351; latest GPU and complete
endpoints have separate manifests and gates below. Earlier measurements are not
relabeled as that source.

## Retained-RHF integration

Production `d3548f1fdcaa7e4137490f32bf7f4abfdfecbafa` passes build 2350,
host 2351, real-device solver and independent small all-coordinate FD 2353,
shared Lambda/factor 2354, complete energy/force/budget 2356–2358 and report 2360.
`retained-reference-summary.json` and its source-manifest checksum preserve this version,
including verified source/compiler/policy files and per-allocation GPU UUIDs.

| Ethane230 | Full | Packed |
| --- | ---: | ---: |
| Energy CCSD median s | 145.958333 | 145.959855 |
| Complete energy median s | 354.075731 | 298.333061 |
| Energy RHF median s | 199.615963 | 143.875171 |
| Complete force s (separate allocation) | 1336.988423 | 1351.807057 |
| Force RHF s | 161.158397 | 180.603671 |
| CCSD device bytes | 4,051,955,712 | 3,800,872,192 |
| Complete energy capacity bytes | 4,413,131,064 | 4,162,047,544 |
| Complete force capacity bytes | 7,107,791,849 | 7,107,791,849 |

The apparent energy difference is explained by the unmodified RHF phase;
CCSD medians are essentially identical. The separate force pair does not show
a complete speedup. Keep packing opt-in for capacity, not ordinary latency.
At the unchanged 2,735,193,944-byte scalar-schedule budget, full history refuses
while packed history converges in 439.607 s. Rejected precursor work remains
unavailable, not zero. Device/energy capacity still saves 251,083,520 bytes;
the complete-force peak is unchanged because later response dominates it.

Maximum force difference is 3.365e-9 Eh/Bohr; energy spread is at most
3.837e-13 Eh, translation residual 1.620e-12 and independent CC replay
5.317e-13. Original Lambda/Z/stationarity gates and existing independent large
two-coordinate/two-step FD re-audits pass. The all-coordinate independent force
audit remains small water only. Both large histories retain 20 iterations and
38 evaluations. Semantic dot, metric-weight, conversion and full extrapolation
work counts are unchanged from the frozen comparison; no total FLOP or
cross-GPU timing claim is introduced.

The complete `current-sources.sha256` and `v2-sources.sha256` lists remain at
`n2:/data/jzzeng/cc-1902-20261005/` and in ignored local qualification artifacts.
The tracked summaries retain their checksums, checked file counts, exact Git
revision, pathspecs and deterministic reconstruction recipe. Existing Git
history reconstructs the lists without duplicating repository-wide hashes in
the PR. All numerical observations and binary/probe/input hashes remain tracked.
