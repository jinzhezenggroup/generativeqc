# Decision: close executable replay and nonlocal AUTO qualification separately from promotion

Status: implemented
Date: 2026-10-10

## Scope and source

This acceptance closes #1856's executable-reference lifecycle and #1854's
component-wise AUTO qualification. It does not close the frozen DFT-MP-v1
matrix, correlated large-system performance, or a default-promotion tracker.
No production equation, precision default, screening or convergence gate changes.

The clean production source is merged commit
`15bc697009d191a88120607e0e50a15561d35d61`, tree
`7b32f9d8c2f547d78ed27ddae9c534c3c863c30b`. The native library SHA-256 is
`7fc478aa78b15a1ee29fb8659c6121c8feb01f4f61aa917e59207d55af7c37c0`.
The source archive and build-record hashes are respectively
`0d44384be0a5434e010b0f867a81a7e6bec7f7f096b818cc555c17e1e17769e1` and
`753ca8a55c3db35a4fa6cfadaf161609a6a6f464b34b29b0621c555ac36b3339`.
Subsequent authored changes are tests, benchmark capture and this rationale,
not a relabeling of a later master binary. The build uses Release/sm_120 and
verified ccache 4.5.1. Its optional shell/stationary AOT settings are off, so it
is not the installed no-runtime-compilation build required by #1186/#1190.

## Executable-reference acceptance

- 47 host capacity/retirement/source-lifetime/publication tests pass without
  skips. They retain exact and one-byte-short admission, allocation unwind,
  restored downstream budgets and non-memory-error propagation.
- 25 real-GPU tests pass without skips in finite Slurm main/node1/5090 job
  7044, assigned visibility 1 preserved, completed exit 0:0 in 24 seconds.
- The new six-case public MP2/RCCSD/RCCSD(T) regression exercises actual bond
  displacement, stationary/moved replay, failed input and recovery, with
  scientific density reuse both enabled and disabled. Both geometries use
  independent PySCF 2.14.0 analytic E/F references; triples use corrected Lambda.
- The existing water and d/f native-reference/resource gates and independent
  nonzero-triples water-force test remain included. The small timing fixture's
  zero triples are not substituted for nonzero-triples scientific acceptance.
- 54 complete prepared energy/E+F calls cover three repeats of cold,
  stationary and changed geometry for each method with density reuse disabled.
  Maximum independently checked errors are `2.4425e-13 Eh` and
  `4.1045e-13 Eh/bohr`. Retained-plan bytes and complete numeric capacity remain
  distinct; opaque CUDA context/stack allocations are not advertised as measured
  numeric-buffer peaks. No large-system or plan-retention-only speedup is claimed.

## Nonlocal AUTO acceptance and negative promotion decision

Three host census tests and both opt-in CUDA RKS/UKS lifecycle tests pass.
Executed mixed work is restricted to Direct Coulomb J/recurrence; SR/LR exchange,
XC/tau/VV10 and final physical-state audit remain FP64. The existing census,
strict-refinement and returned-state assertions are unchanged.

Five alternating FP64/AUTO pairs for each spin cover cold, warm, moved and
moved-warm complete E+F: 80 endpoints. Cold includes prepared-owner construction;
replays include execution through exported force results. Calculator/CUDA
initialization and teardown are excluded. Independent PySCF 2.14.0 uses the same
12x4x8 moving quadrature, original Becke partition without radius adjustment,
STO-3G and WB97M-V definition. All Cartesian force components are checked against
two-step independently reconverged energy differences (`3e-4`, `1e-4 bohr`).
Maximum E/F errors are `2.2465e-9 Eh` and `1.5952e-8 Eh/bohr`, within the unchanged
`2e-8 Eh` / `2e-7 Eh/bohr` comparison gates.

| Spin | Phase | FP64 median (s) | AUTO median (s) |
| --- | --- | ---: | ---: |
| RKS | Cold | 0.980956 | 0.968640 |
| RKS | Warm | 0.028596 | 0.043452 |
| RKS | Moved | 0.954606 | 0.976424 |
| RKS | Moved-warm | 0.028472 | 0.053648 |
| UKS | Cold | 1.018993 | 2.230424 |
| UKS | Warm | 0.040609 | 1.302798 |
| UKS | Moved | 1.003974 | 2.225825 |
| UKS | Moved-warm | 0.040231 | 1.302007 |

The small RKS cold difference does not overcome replay/moved or UKS regressions.
Retain AUTO as explicit opt-in and FP64 as default. This is negative promotion
evidence for the explored tiny RKS/UKS domain, not a universal mixed-precision
conclusion or a DFT-MP-v1 speedup receipt.

## Harness mistakes that must not recur

`validate_and_normalize` consumes raw basis coefficients. Calling it again on a
normalized geometry copy changes the scientific basis, so rejecting executable
reuse is correct. Coordinate-only native tests must preserve normalized shells.
All energy/frame and exact/one-byte-short numerical gates remain intact.

Prepared execution with omitted coordinates uses the original prepared input,
not the preceding displaced input. Moved-warm and recovery tests must explicitly
resubmit moved coordinates. The original-return timing population is retained
as a failed harness population and is not relabeled as moved-warm evidence.

Capacity rejection can throw before submission or return an OOM item after
submission. Check the exact one-byte-short boundary and non-publication for both,
not an assumed exception representation. Independent native subprocess probes
run before the parent executes the larger correlated CUDA kernels: the reverse
order produced explicit OOMs, retained as failures, while isolated probes and the
final ordered population pass. This is a harness isolation observation, not a
proved CUDA leak or permission to exclude numeric allocations.

The initial water MP2-force pilot spends substantial host work in the existing
RawSource/CPHF path; its finite diagnostic/pilot failures remain retained. This
acceptance neither claims CUDA-resident MP2 response nor promotes that workload.

## Retention and reproduction

Raw source/build/cache records, XML/logs, original failed populations, full
observations, independent references, validators and finite-job receipts remain
in ignored `.artifacts/issue1856-acceptance-20261010/` locally and under
`/data/jzzeng/qc-issue-1856-acceptance-20261010/` on n1. The checked
`acceptance-summary.json` reports PASS for these two scopes only.

Reproduce with a source-matched native library, pinned PySCF 2.14.0 and the verified
compiler cache. Run the 47 host tests from `host.sh`, and the original component
census tests plus `test_wb97mv_auto_matches_fp64_cold_warm_and_moved` for #1854.
For #1856, use the retained `gpu.sh` with the native residency probe first and
all explicit CUDA opt-ins, followed by both commands:

```sh
python -m benchmarks.correlated_reference_plan_reuse --case h2 --repeats 3 --output .artifacts/replay-energy.json
python -m benchmarks.correlated_reference_plan_reuse --case h2 --repeats 3 --forces --output .artifacts/replay-forces.json
```

Every GPU command must remain inside a finite compatible `srun`; preserve its
assigned visibility. `precision-1854.py`, `oracle-1854.py` and `finalize.py` retain
the exact independent/matched acceptance protocol, including failed initial
variants. No Release, external archive, merge or auto-merge is needed or authorized.

## Remaining issues

#1186 still has 44 frozen semilocal FP64 rows with zero validated receipts;
#1187's frozen hybrid rows likewise need their own matching campaign. The
repository's #1190 runner accepts an external installed-production adapter;
the adapter is not delivered by the runner itself. Neither small acceptance
here nor the older different-grid campaigns may be imported as frozen-row PASS.
#1853/#1873 retain their explicitly recorded cross-functional/provider acceptance
and ownership gaps. #1883 still has the serial orthogonalizer reconstruction;
#1877's standalone rank-k PR does not deliver two production consumers. These
issues must not be closed merely because the two scopes above pass.
