# Exact density-keyed XC geometry response qualification

This default-off experiment retains successful density-only nuclear and XC
geometry response within one immutable prepared owner. Integral/Pulay response
is still evaluated from the current authenticated state. Both arms enable the
separately qualified exact two-electron response reuse; only
`GENERATIVEQC_KS_GEOMETRY_RESPONSE_REUSE` differs (0 versus 1).

All 72 complete E/F calls pass independent 1e-8 Eh / 1e-7 Eh/Bohr gates under
the original FP64, 48×16×32 grid, 100-SCF-iteration, 1800-second point protocol.
The default README curves and the unresolved 96-atom acceptance are unchanged.

| Atoms | Off cold | On cold | Off warm | On warm | Off moved | On moved | Off moved-warm | On moved-warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 7.40662 | 7.59645 | 0.32705 | 0.19861 | 5.52434 | 5.68562 | 0.32701 | 0.19701 |
| 12 | 31.20749 | 31.00331 | 1.21108 | 0.69052 | 19.73509 | 19.61703 | 1.21140 | 0.69203 |
| 48 | 254.51284 | 254.67130 | 7.42569 | 4.36243 | 213.91562 | 214.06719 | 7.56981 | 4.39369 |

Seconds cover complete endpoints; warm values are five-sample medians. On warm
hits, actual partition/AO/geometry visits and pair sweeps are zero. Capacity
estimates remain labelled as capacity; producer timings are not replayed.

Measured and built source: `8683bd03f735dc5ee52e66b58dce9a7d6650cfb9`.
Native identity: `d200091023ce32df93713f466c3a39e665b2255d5110c16b079bee964f62de21`.
Library SHA256: `1ae4d70019e3a4f018bb55df74c71a87dc3eaf7347980a7c9edd3991e1fa2a8b`.
The source inventory and all consumer hashes are retained in the receipts.

Finite n1 RTX 5090 Slurm qualification:

- Job 8447: 9 passed, including six independent molecular E/F and reconverged
  displaced-energy cases, actual RKS/UKS snapshot hits, moved geometry, budget
  rejection and stale/failure recovery.
- Jobs 8448 / 8452: memcheck / synccheck, 3 passed each, zero errors.
- Complete same-library populations use jobs 8449 / 8450. All samples and full
  force vectors are retained. Sanitizer durations are not timing observations.
- The first deployment's source-identity mismatch remains recorded; qualification
  used the subsequently rebuilt matching library.

Run `python benchmarks/results/exact-geometry-response-20261014/verify.py`
(with `defusedxml` installed) to recompute every E/F gate, population and median
and inspect the retained JUnit outcomes. No tolerance-based key, host reference
calculation or altered stopping rule is introduced.

Two explicit expanded-range 96-atom attempts are retained separately. Integral
reuse alone (n4 job 915) returns cold 944.105 s and five warm calls with median
38.376 s; integral plus geometry reuse (n1 job 8451) returns cold 943.962 s and
five warm calls with median 15.525 s. Both reach the unchanged 1800-second
whole-point limit during moved/0. All 12 returned calls pass independent E/F
gates, but neither is a complete population or a qualified moved timing. These
experiments do not close the outstanding 96-atom acceptance in #2426. Their
distinct source/library identities, flags, full vectors, logs and outcomes are
preserved and verified alongside the six complete smaller populations.
