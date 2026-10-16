# Occupied-frame XC: qualified opt-in cold optimization

Reuse the accepted density's exact AO frame for four orbital jets and generated
rho/gradient/tau algebra when occupied rank is below tile AO count. Keep AO
jets/Vxc, exact Direct J/K, FP64 and bounded fallbacks; no new allocation/sync.
Enable with `GENERATIVEQC_CUDA_KS_OCCUPIED_XC=1`; **default off**.

## Sources and protocol

Clean pairs freeze master `b8102246862f68603adae0394dd41f2205d87425`, without
unmerged #2558. Baseline/C2 libraries and source hashes are in the packet.
After #2574/#2573, integration at `a67846c22d415c889e63d9530e15b6da9e4cbfe3`
passes157 host tests and job9379's eight native cases,24 sanitizer runs and24
complete3/12-atom calls. Do not relabel frozen timings or homepage #2572.

Finite RTX5090 Slurm jobs1099/1695 preserve assigned visibility, eight CPU
threads, cold + five warm + moved + five moved-warm calls, and the unchanged
1800s population cap. r²SCAN-3c uses canonical spherical def2-mTZVPP, strict
FP64 and the moving unpruned48×16×32 Becke grid per atom. PBE0/HF retain matched
def2-SVP controls. Cold includes preparation/returned host forces; caches remain.

## Complete endpoints

Seconds, baseline→C2; warm columns are medians of all five untrimmed calls.

| Direct case | Cold | Warm | Moved | Moved-warm |
|---|---:|---:|---:|---:|
|r²SCAN-3c96 /1088AO|296.461741→275.539082|82.482484→82.523721|205.431849→185.186882|82.496878→82.430095|
|r²SCAN-3c48 /544AO|73.381158→65.852037|21.315912→21.247030|55.105953→50.175575|21.279209→21.286518|
|PBE0/48|23.887615→23.948525|4.606315→4.603921|22.702510→22.590984|4.602439→4.591913|
|HF/48|11.170658→11.172475|0.745783→0.746391|7.220619→7.223221|0.744577→0.745791|

Primary96 cold improves7.06%, with identical24/24 iterations/Focks; moved
improves9.85% but14/14→13/13 work. At48, cold/moved improve10.26%/8.95%,
with identical19/19 and13/13 work. PBE0 cold is+0.255%, HF+0.016%; their
cold/moved work is17/18,12/12 and21/22,16/17 respectively. Every warm call
is1/1; no work normalization is used.

The opposite-order96 repeat gives cold306.917294→276.879606s(-9.79%), with
25/25→24/24 work; moved198.229687→186.560340s, both13/13. Warm/moved-warm
are82.974035→83.024312s/83.006735→83.128368s. All120 clean C2-cohort calls
pass independent1e-8Eh/1e-7Eh/Bohr gates; maximum errors are
2.50112e-11Eh/6.65939e-11Eh/Bohr. Small integration calls also pass separately.

## Separate C2 profiles

Job1695 cold profiles have identical24/24 work and complete wall
299.707898→277.686424s. Kernel **device** seconds, not wall-time summands:

| Cold device scope | Baseline | C2 |
|---|---:|---:|
|SCF Direct J|102.103332|102.421902|
|XC/AO/grid|95.872462|73.503203|
|Order6 J′|21.924526|21.922539|
|Order7 J′|28.255722|28.257807|
|Order8 J′|5.816746|5.810276|

Density-product/feature kernels fall45.518254→23.136080s with the same73,920
launches;23 accepted-frame leases are consumed, none warm. Logical products
change `4*p*a*a`→`4*p*a*r`, not just candidate/page counts. Hardware FLOP
counters are unavailable (`ERR_NVGPUCTRPERM`). Warm order6/7/8 device times
are21.948622/28.253005/5.812709→22.063104/28.254248/5.817379s. Their executed
primitive-pair products remain2,400,504,912/1,250,544/95,976; missing recurrence
counters remain null. The packet separately retains exclusive host/device-window
ledgers, other force/Becke work, validation, preparation and synchronization.

## Evidence and limits

The packet losslessly replays48 primary calls and12 profile calls;264 other
calls retain independent collection-time force receipts, not offline force
replay. Integrated3/12 receipts are separate. All populations complete; earlier
observations and C1 negatives remain. The storage cap is unchanged.

C1's FP32 resource coupling is rejected; C2 specializes both sources and gates
batch/standalone resources. Static resource bounds are not measured occupancy.

Final combined host/evidence suite:188 passed; integrated-library suite:157 passed.
Native cases and full-leak memcheck/synccheck/racecheck pass with zero errors,
hazards or leaks. Independent replay and corruption/error-path tests run with:

```bash
PYTHONPATH=.:python python benchmarks/results/occupied-xc-20261017/summarize.py
PYTHONPATH=.:python python -m pytest -q tests/python/test_occupied_xc_evidence.py
```

**The≥10% warm target is not achieved.** Warm remains82–83s, roughly3.1× the
historical GPU4PySCF26.59s (a different source cohort, not a new matched test).
Next masses are SCF Direct J and stationary order6/7 recurrence/contraction work.
See `.agents/notes/implemented/performance/2026-10-17-occupied-xc-frame.md` for
ownership invariants, rejected trials and source/cache/integration rationale.
