# Current-source occupied-XC cold endpoint qualification

Measured source: `66a8a98bf5e81d17a71225f9fcd091022d64e4bf`.
This integrates the existing compiler/native occupied-XC proposal from PR #2577
(`b1851abc43856f777a0f99e8150bc87b2d8e6b66`) with the current cold SCF owner.
Historical PR timings are not reused or relabeled. The explicit selector
`GENERATIVEQC_CUDA_KS_OCCUPIED_XC=1` enables the candidate; unset stays off.

Only a token-bound canonical frame that actually formed accepted D can supply
four strict-FP64 meta-GGA orbital jets. Tiles with occupied rank below their
actual AO count reuse dead density panels; potential assembly keeps its AO
jets. Pointer/generation/shape/occupation checks guard the lease. Every completion
retires it, and only an accepted proposal that did not consume frame scratch can
republish it. Held confirmations, rejected probes, mixed precision, nonlocal,
response and unsupported domains keep the existing matrix route. PBE's one-jet
route is unchanged. No basis, quadrature, screening or convergence change.

Five alternating fresh-process selector-0/selector-1 pairs per mode measure
water96 r2SCAN-3c complete energy and analytic forces on a Slurm RTX 5090.
Both arms use 32 DF charge partitions where applicable. Cold includes construction,
preparation, SCF, forces and host export. Three same-owner warm calls contribute
one median per process.

| Mode / phase | Off median / MAD (s) | On median / MAD (s) | Improvement | Qualified gain |
| --- | ---: | ---: | ---: | --- |
| DF cold | 43.990828 / 0.315277 | 41.718511 / 0.045070 | 5.165% | yes |
| Direct cold | 49.427546 / 0.030732 | 47.710711 / 0.100847 | 3.473% | yes |
| DF warm | 12.184105 / 0.059636 | 12.143195 / 0.011401 | 0.336% | no |
| Direct warm | 15.915172 / 0.021604 | 15.907422 / 0.004412 | 0.049% | no |

Both cold gains exceed the unchanged maximum of 2% and twice summed relative
MADs (2% for all four populations). Warm's held-density path correctly declines
the occupied lease; no warm speedup is claimed. These incremental gains do not
establish parity with GPU4PySCF. Earlier single-process Standard/Fast screens for
PBE, r2SCAN, r2SCAN-3c and M06-2X remain separate; every observation is retained.

All **168 native calls** independently qualify: 64 pilot, 80 paired, eight
intrusive diagnostic and 16 cold/warm/moved/moved-warm calls. Static qualification
uses the audited dense-grid and CPU/GPU references. Moved trajectories compare
against GPU4 at the exact displaced geometry (atom one z +0.001 Bohr), including
a newly measured Direct control. This establishes sampled schedule parity, not
independent displaced-grid convergence. Maximum moved force error is 1.316e-5
Eh/Bohr (Fast Direct); Standard Direct is 7.108e-7 and Standard DF is 1.179e-7.

Separate trace receipts confirm eight occupied frames in r2SCAN and nine in 3c;
first cold and held warm calls retain matrix contraction. Their total XC device
intervals change 6734.194 → 6292.336 ms and 13669.070 → 11751.823 ms respectively.
These diagnostics do not enter clean endpoint statistics. Selected exact XC
trace records and full-trace hashes are retained alongside the endpoint data.

Validation: 1260 host cases; compiler structure check; 18 independent GPU feature
and owner cases; all eight owner cases under memcheck, synccheck and racecheck,
with zero errors/hazards. Current executable tests cover lease rejection and
recovery across held/rejected SCF decisions. The initial integration's two failed
obsolete source-spelling assertions are retained in the first host log; existing
executable full/delta handoff checks remain, and no scientific coverage is removed.

The immutable sm120 package and five stationary manifests were built with
verified sccache 0.16.0 after authenticating 6004 inputs. `receipts.json.xz`
retains all numerical arrays, work/settings, source/package identities, original
runners, upstream import patch, build/cache/test logs and JUnit receipts. The
input inventory is an authenticated delta against the charge-partition inventory.

```bash
PYTHONPATH=python:. python benchmarks/results/occupied-xc-cold-20261023/reproduce.py \
  > .artifacts/occupied-xc-cold-reproduced.json
```

The reproducer rechecks every native call and all process-level timing statistics.
Nine historical DF receipts, including rejected schedules, are losslessly
compacted under `../issue377-379-df/` with exact restoration. Storage caps are
unchanged; no observations are dropped and no external archive is created.
