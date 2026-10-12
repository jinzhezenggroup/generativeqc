# Decision: share the exact RHF CUDA source throughout MP2 E+F

Status: implemented
Date: 2026-10-12
Base: `ad97d29b9ae684059479cbdc51536616e4d7090f` (`origin/master`)

## Problem

Conventional CUDA MP2 kept the reusable RHF executable but constructed a new
CPU `RawSource` for correlation and another compatibility force consumer.
The force chain repeatedly transformed the same AO source for the energy
adjoint, canonical Fock replay, orbital rotations, every response action and
relaxed stationarity replay. CUDA tensor kernels alone could not eliminate
CPU integral production or the many shell-sized source/transform launches.

CC already used a qualified RHF interaction-source handoff, a bounded generic
device-read provider and resource-selected cross-shell source tiles. Duplicating
those scientific producers or introducing a PySCF production path was unnecessary.

## Decision

- Request the existing resident-or-compact exact CUDA RHF source, with current
  geometry/basis/device identity, and borrow it through both energy and forces.
- Generalize the conventional force boundary to `ElectronInteractionSource`;
  the RI force boundary remains its explicit CPU `RawSource` contract.
- Charge the complete source owner through the existing provider plans. Clear
  the duplicate handoff reference, then reclaim a unique resident executable
  only after complete-endpoint peak accounting and drained consumers.
- Retry once without optional source residency only on memory refusal. Keep
  numerical, stationarity, backend and derivative failures as failures.
- Use the existing compiler-owned source-read minimizer for device-backed
  energy and force tiles. Force candidates include every streamed shape and
  simultaneous output lifetime, response payload and the legacy derivative
  allowance up to the bounded derivative scratch ceiling. Retain shell-sized
  execution when no broader candidate is admitted or controls disable it.
- Preserve the CPU endpoint, generated MP2 equations, full-force residual and
  stationarity gates, transactional publication and explicit unsupported
  CUDA RI-MP2 force status.

## Controls and observability

`GENERATIVEQC_MP2_REFERENCE_SOURCE=0` restores the old source route.
`GENERATIVEQC_MP2_ENERGY_SOURCE_TILING=0` and
`GENERATIVEQC_MP2_FORCE_SOURCE_TILING=0` isolate source handoff from tiling.
The opt-in progress journal records per-call completed source reads, device
reads, contraction summands and transfers. Deltas prevent accumulated caller
ledgers from being counted repeatedly. A failed call does not publish completed
work counters; full endpoint timing still includes discarded attempts.

The conventional path still stages MO results and executes host orchestration,
adjoint and Krylov work. This is not a fully resident MP2 force implementation,
nor CUDA RI-MP2/CBS force enablement. No total-device residency or unmeasured
allocation peak is inferred from the planned byte budget.

## Validation

- Native force parity through a method-neutral forwarding source; exact
  one-byte-short complete-force admission rejects before any source reads.
- Existing native gradient, public force, resource, transaction and warm-batch
  tests retain their numerical/lifecycle gates.
- The obsolete Python source-spelling/order audit moves to the real native
  admission check. Compiled workspace/allocation tests and RI resource gates
  remain; no numerical or allocator coverage is removed.
- `benchmarks/mp2_cuda_ef_endpoint.py` measures complete cold and prepared-warm
  E+F, with pinned independent energy inputs and an optional PySCF gradient.
  Oracle gates are 1e-9 Hartree and 2e-6 Hartree/Bohr; CUDA source-route parity
  remains tighter at 2e-8 Hartree/Bohr. Oracle construction and traced replay
  are outside the endpoint samples.
- Real-device runs use finite-time `srun` on an allowed node. The CUDA libraries
  were built on `n4`; qualification ran on `n1` (`node1`), the `main` partition
  and `gpu:5090:1` when `n4` was occupied. The maintained `n3` was never used.
  Slurm device visibility was preserved. Raw source/library-matched measurements
  live in ignored local artifacts.
- Local Release/OpenBLAS CPU regression: both MP2 native contract/gradient
  executables passed; the focused Python public, batch, workspace, documentation
  and compiler-schedule suite returned 56 passed / 21 skipped (GPU not allocated).
  Independent PySCF 2.14.0 full-force checks for H2, water, LiH and the f-shell
  HeH+ fixture have maximum
  energy error 1.68e-12 Hartree and force error 1.54e-7 Hartree/Bohr. The frozen
  H2/cc-pVDZ spherical case separately agrees within 6.44e-15 Hartree and
  5.54e-14 Hartree/Bohr. These are CPU acceptance evidence, not GPU speed claims.
  The benchmark's `--case h2-ccpvdz --oracle` uses the checked retained basis
  record and validates every measured cold/warm result against its gradient.

### Real-device endpoint evidence

Unmodified frozen master and candidate used the same Release `portable_cuda`
build, SM120, CUDA 12.9.1 and compiler cache. Runtime was an RTX 5090 (32 GiB),
driver 580.95.05, on `node1`; OpenMP/OpenBLAS used one thread. Separate Slurm
jobs used one GPU each, without locked clocks or exclusive-node allocation.
The master library was frozen before uploading native changes. An incremental
rebuild initially saw older archived source mtimes; touching the uploaded files
forced all changed C++ objects to rebuild, and the generic force symbol and
library hashes were checked before candidate qualification.

Both benchmarks run the full RHF + MP2 + adjoint + response + relaxed weights +
analytic derivatives. Prepared warm means executable reuse, not replaying a
cached energy/force or oracle density. Three warm samples give the median below;
oracle construction and work-journal replay are outside those samples.

| Case | Public AO / occupied | Master cold (s) | Candidate cold (s) | Master prepared warm (s) | Candidate prepared warm (s) | Warm ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H2/STO-3G | 2 / 1 | 0.868705 | 0.827493 | 0.123276 | 0.122879 | 1.00x |
| Water/STO-3G | 7 / 5 | 118.563790 | 6.773490 | 121.741795 | 6.894938 | 17.66x |
| LiH/STO-3G | 6 / 2 | 33.103885 | 3.178292 | 33.185221 | 3.153629 | 10.52x |
| H2/cc-pVDZ spherical | 10 / 1 | 16.074574 | 1.079305 | 16.107773 | 1.065855 | 15.11x |

All measured cold/warm samples pass the independent PySCF E+F gates. Across
these cases the candidate maximum energy error is 1.701e-12 Hartree and maximum
force error 1.542e-7 Hartree/Bohr. The master/candidate saved total energies are
identical; maximum saved force difference is 2.263e-15 Hartree/Bohr. A separate
candidate f-shell endpoint/gradient qualification passes too (prepared warm
9.187422 s); this is not an f-shell speedup measurement against master.

Completed-work census uses one untimed full endpoint per case. The legacy
ledger is the candidate with `GENERATIVEQC_MP2_REFERENCE_SOURCE=0`, not an
instrumented master library; public source-route parity tests qualify that
unchanged producer/transform route. These are actual MO-source/transform work
counts, not whole-endpoint FLOPs or device instruction counts.

| Case | Legacy / new source reads | Legacy / new transform summands | Legacy / new H2D bytes | D2H bytes (both) |
| --- | ---: | ---: | ---: | ---: |
| H2 | 448 / 28 | 7,328 / 2,324 | 6,656 / 3,072 | 1,624 |
| Water | 147,281 / 576 | 32,663,582 / 4,636,002 | 11,435,760 / 371,952 | 275,216 |
| LiH | 21,805 / 270 | 6,998,160 / 1,514,640 | 2,945,856 / 146,496 | 139,392 |
| H2/cc-pVDZ | 144,631 / 232 | 297,502,190 / 22,927,990 | 18,977,920 / 417,920 | 974,120 |

Source scans and source values remain identical for each matched case; all new
source reads use the device interface, while legacy device reads are zero.
Broader tiles remove redundant staged transforms/launches, not scientific
operator applications. Unchanged D2H makes the remaining host staging explicit.
H2 has no meaningful whole-endpoint speedup despite lower transform work.

CUDA-built native MP2 contract and gradient executables pass under `srun`.
Python GPU-enabled public/batch/workspace regression gives 54 passed, 12 skipped,
5 deselected; f-shell legacy-route checks were excluded from that run because
CPU integral replay is expensive. The independent f-shell candidate benchmark
provides its separate scientific gate instead. Warm executable reuse, changed
geometry, transactionality, backend selection and source route parity pass.

An additional persistent opt-in GPU test searches the actual legacy E+F
admission boundary, then executes both source routes twice with a retained warm
density just above that boundary. It passes separately (1 passed, 44 deselected).
The byte-exact diagnostic sweep found minimum cold legacy admission at
113,386,440 bytes; at 113,387,464 bytes both warm endpoints return valid E+F,
and the optional source reports `mp2_reference_source_fallback=memory` before
using the compatibility source. Budgets 104/108 MiB reject both routes, while
109–128 MiB accept both, remain within their endpoint budgets, and preserve the
warm executable when its storage fits. This is a bounded fallback/lifetime
qualification, not a simulated allocation failure or a measured allocator peak.

A further candidate-only H2/cc-pVTZ spherical qualification (28 public AOs,
one occupied orbital; exact PySCF 2.14.0 basis inputs frozen in the receipt)
returns complete E+F in 72.309226 s prepared cold and 71.139585 s prepared warm.
Energy error is 9.993e-15 Hartree, maximum force error 4.078e-10 Hartree/Bohr and
response absolute residual 8.677e-18. The 256 MiB endpoint admission selects
force AO tile 28, with 1,150 source scans, 1,165 device reads, 706,854,400 source
values, 9,928,871,316 transform summands and 59,216,696 D2H bytes. This case has
no master timing, so no TZ speed ratio is claimed. Its much greater logical
work and host staging identify further response/replay work for CBS-sized
systems; successful DZ/TZ gradients alone do not establish a CBS workflow.

Raw local receipts: `.artifacts/mp2-cuda-ef-20261012/{master-ef,candidate-ef,
candidate-f-ef,legacy-work}.json`, corresponding logs, and `tests-candidate.log`.
Additional receipts are `low-memory.json`, `h2-ccpvtz-ef.json`,
`tests-candidate-resource.log` and the retained diagnostic scripts in that
ignored directory.
Remote isolated build receipts remain at
`/data/jzzeng/qc-mp2-gpu-20261012-ad97d29` on `n4`, and device receipts at the
same path on `n1`. Library SHA256:

- Master: `b830bf7189fd93ee6345e776b47bc6d7b18a418faff87abc23f823441939b490`.
- Candidate: `95f5e5531174813261f7830b5c1073c677fe086c45ffb66f812a9f5fd2be6d6b`.

Reproduction (through `srun --partition=main --gres=gpu:5090:1`, with an explicit
finite `--time`, the frozen library in `GENERATIVEQC_LIBRARY` and
`PYTHONPATH=python:.`):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/mp2_cuda_ef_endpoint.py \
  --case h2 --case water --case lih --case h2-ccpvdz --oracle --repeats 3 \
  --output .artifacts/mp2-ef.json
```

Add `--trace` only with candidate instrumentation. Large-basis CBS extrapolation
and CUDA RI-MP2 analytic forces are not established by these measurements.

## Rejected alternatives

CPU RI energy-only GEMM optimization does not answer GPU E+F and is not part of
this patch. Retaining a dense response Hessian, changing the MP2 equations or
adding a second AO producer would expand ownership, lifetime and scientific
qualification unnecessarily. A kernel-only speedup or an energy-only RI
benchmark cannot establish complete-force performance.

## Revisit when

Complete endpoints expose remaining host MO transfers, per-row response blocks,
or orbital-rotation replay as the dominant cost. Reuse the established resident
response/adjoint machinery then, with independent gradients and complete
endpoint accounting. Large correlation-consistent basis and RI E+F capability
must be qualified separately before making CCSD(T)/CBS workflow claims.
