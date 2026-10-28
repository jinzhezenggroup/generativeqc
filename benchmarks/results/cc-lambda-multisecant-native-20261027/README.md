# Native CCSD-history Lambda preconditioning: no default promotion

The rank-4 and rank-6 diagnostic candidates did **not** accelerate complete native
CBS energy + analytic force endpoints. Production source and defaults remain
unchanged. This follows the small host screen in PR #2822; it does not establish
that every future native multisecant implementation will be unprofitable.

## Measured task

All comparisons use base `cd9ad801b09225d923ae5e2995c70274f76a50cf`, pinned QM7
C2/C4/C6 geometries, cc-pVDZ/cc-pVTZ and matching JKFIT bases, DF HF reference,
`policy="standard"`, `preconvergence="auto"`, and a 32 GiB numeric budget.
`complete_seconds` includes creation of a fresh CBS calculator and its complete
energy + force call, including history export, fit and preconditioner callbacks.
Basis metadata preparation is outside this timer. A C1 call warms process/CUDA
startup separately and is retained in each record. No converged scientific state
is reused between endpoints; BLAS/OpenMP use one thread.

Both arms use the **current budget-scaled Standard controls**. The receipts name
this selection `relaxed` with reason `budget_scaled_convergence`; this experiment
does not relax any gate or change the existing FP32 preconvergence policy.
In particular, response solve tolerance is `2.3827302298009143e-9` and the
independent Lambda tolerance is `2.3827302298009143e-8`. The earlier host screen
used a `1e-12` solve and is not a production timing baseline. Historical archive
names containing `strict` mean placement in the **final FP64 stage only**, not a
different convergence policy.

Every GPU endpoint ran through finite Slurm allocations on the same n4 RTX 5090:
jobs 1352 (pilot), 1353 (FP64 placement), 1354 (warmup + FP64), and 1355 (C6).
Raw records include GPU UUID, driver, device visibility, scientific input
identities, complete observables, phase timings, operator counts and controls.
C2/C4 have two repetitions per arm with reversed order on the second repetition;
C6 has **one observation per arm**. All 29 comparison endpoints are retained,
including the two initial C2 pilot observations.

## Results

Times below are complete E+F seconds. Parentheses give both repetitions; the
leading number is their median. Action pairs are FP32 / FP64 Lambda actions.
Every measured arm retained Q32.

Final FP64 stage only (`cohort.json.xz`):

| Case | Diagonal | Rank 4 | Rank 6 | Actions: diagonal / rank 4 / rank 6 |
| --- | ---: | ---: | ---: | --- |
| C2 | 8.348892 (8.353536, 8.344247) | 8.377243 (8.371316, 8.383170) | 8.383417 (8.382451, 8.384383) | 11/6, 11/6, 11/6 |
| C4 | 66.551320 (66.451098, 66.651541) | 67.138124 (67.141891, 67.134356) | 67.311832 (67.333703, 67.289962) | 12/7, 12/7, 12/7 |

FP32 warmup and final FP64 stage (`warmup-cohort.json.xz`):

| Case | Diagonal | Rank 4 | Rank 6 | Actions: diagonal / rank 4 / rank 6 |
| --- | ---: | ---: | ---: | --- |
| C2 | 8.361449 (8.376704, 8.346194) | 8.442197 (8.440532, 8.443863) | 8.457852 (8.460215, 8.455489) | 11/6, 10/7, 10/7 |
| C4 | 66.605554 (66.550544, 66.660564) | 66.809611 (66.829870, 66.789352) | 67.029445 (67.031684, 67.027205) | 12/7, 11/7, 11/7 |

C6 warmup + FP64 pilot (`large-pilot.json.xz`), one observation per arm:

| Rank | Complete E+F seconds | Lambda seconds | FP32 / FP64 actions |
| --- | ---: | ---: | --- |
| Diagonal | 464.611116 | 186.049200 | 12 / 7 |
| 4 | 467.602193 | 186.916756 | 12 / 7 |
| 6 | 469.320716 | 187.093123 | 12 / 7 |

The initial C2 pilot (`pilot.json.xz`) gave diagonal 8.347945 s and rank 6
8.381660 s, both with 11/6 actions. No population shows a median endpoint win.
FP64-only placement saves no actions. Warmup placement saves one FP32 action on
C4, but its complete cost exceeds that saving; C2 also adds an FP64 action.
C6 saves no actions. Its rank-4/rank-6 callback totals are 2.795/4.523 s, of which
fit costs are 1.006/2.255 s. Retained factors occupy 374,770,800/541,335,600 bytes.
Construction estimates are approximately 2.75/3.91 GB: these are conservative
numeric estimates, **not measured RSS or VRAM peaks**.

All component and composite energies equal the corresponding same-record
diagonal control; maximum full-vector force difference is `3.66918e-11` Ha/Bohr.
The largest independently audited candidate Lambda residual is `2.03217e-9`,
within unchanged native limits. The pinned independent C2 PySCF reference from
`cbs-standard-preconvergence-20261025` is reused only after matching geometry,
orbital/auxiliary identities, HF reference and dataset. Maximum component or
composite energy error is `6.28959e-10` Ha and maximum directional force error
against its two finite-difference steps is `1.06285e-8` Ha/Bohr. Gates remain
`1e-8` for baseline energy/full-force parity, `1e-8` for independent energy, and
`3e-6` for independent directional force. Historical reference timings are not
used. C4/C6 have full-vector diagonal parity, not separate PySCF oracles.

## Diagnostic implementation and limits

The archived patch exports the already-computed final FP64 CC ring plus final
accepted point before its owner releases storage. It does not reevaluate the CC
residual. This probe accepts the unweighted native ring only, refuses the packed
ring, and explicitly packs into Lambda's sqrt-weighted representative ordering.
Denominators are packed without amplitude weights. Existing `SecantHistory`
recovers the incoming Jacobi amplitudes and the existing scaled thin-SVD fit
builds the low-rank inverse. The synchronous host callback applies its transpose
inside actual native GMRES. Generated CUDA actions and independent residual
audits remain authoritative.

Detached host factors are charged against subsequent endpoint phase budgets.
The complete timing includes Python/NumPy capture/fit/apply overhead. This is a
measurement prototype, not a production callback, ownership API or GPU-resident
implementation. Rank zero disables the callback in the same candidate binary.
No native source changes from this probe are retained in production.

No C7, tight-budget native fallback, packed-ring integration, or peak-memory
qualification was performed. Two reversed repetitions do not establish a broad
statistical ranking, and the C6 pilot is especially limited. Revisit promotion
only after a cheaper implementation or better history selection demonstrates
repeatable complete endpoint gains, preserves Q-batch and passes physical gates.

## Evidence replay and reproduction

Offline replay needs Python + NumPy, no GPU. Historical source reconstruction
also needs Git history containing the measured base and the `patch` executable:

```bash
python benchmarks/results/cc-lambda-multisecant-native-20261027/verify.py \
  --reconstruct-source --output .artifacts/multisecant-verification.json
```

The verifier checks all 29 observations, frozen member/driver/reference hashes,
source and Slurm bindings, unchanged controls, actual candidate use, complete
observables and independent numerical gates. It emits every timing population.
The optional reconstruction checks only the measured historical base plus
patches; it does not pin current production source or add a required CI test.

`qualification.json` binds the evidence. `source-identities.json` binds both
source variants; `compiled-provenance.json` records the second variant's source,
binary and actual Ninja compiler commands. `build-receipts.json.xz` retains
compiler-cache snapshots, Slurm receipts and runtime-library hashes. Builds on
n1 used CUDA 12.9.1, SM120 and verified sccache 0.16.0. Cache statistics are shared
server snapshots, not isolated per-job hit attribution.

For a fresh native rerun, use a separate checkout of the measured base. Decompress
and apply `strict-candidate.patch.xz` with `patch -p1`; for the second variant,
also apply `warmup-addition.patch.xz`. Configure/build as in `build.sh`, adapting
host dependency paths and verifying the compiler cache before any compilation.
Preserve a separate binary for each variant. Decompress the exact measured
drivers and host tests using this mapping:

| Archive | Name used by retained runner |
| --- | --- |
| `strict-experiment.py.xz` | `experiment.py` |
| `experiment.py.xz` | `experiment-warmup.py` |
| `test_probe.py.xz` | `test_probe.py` |
| `test_probe_warmup.py.xz` | `test_probe_warmup.py` |

`run-n4.sh` and `run-warmup-n4.sh` are historical machine-specific receipts,
including the original `/data/jzzeng/cc-multisecant-20261027` layout. They first
run two host callback tests and then the endpoint driver. They use an isolated
copied n1 loader/runtime because the n1 binary required a newer glibc than n4;
system files were unchanged. A local compatible build need not transport that
runtime. Adapt paths explicitly; do not change Slurm's `CUDA_VISIBLE_DEVICES`.
The pinned dataset SHA256 is
`fd6c26c1086887310518e6a6da778253bc5d3ceb8166007a679a740fff9ee6ea`.
With those historical paths provisioned on n4, the comparison commands are:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:15:00 \
  bash /data/jzzeng/cc-multisecant-20261027/run-n4.sh \
  --carbons 2 4 --ranks 0 4 6 --samples 2 --budget-mib 32768 \
  --policy standard --preconvergence auto --output /tmp/multisecant-fp64.json
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:15:00 \
  bash /data/jzzeng/cc-multisecant-20261027/run-warmup-n4.sh \
  --carbons 2 4 --ranks 0 4 6 --samples 2 --budget-mib 32768 \
  --policy standard --preconvergence auto --precondition-warmup \
  --output /tmp/multisecant-warmup.json
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:40:00 \
  bash /data/jzzeng/cc-multisecant-20261027/run-warmup-n4.sh \
  --carbons 6 --ranks 0 4 6 --samples 1 --budget-mib 32768 \
  --policy standard --preconvergence auto --precondition-warmup \
  --output /tmp/multisecant-c6.json
```

Run these on a permitted node with compatible resources; n3 is excluded.
Exact measured patches/drivers/tests are compressed to preserve their bytes.
Their archived hashes describe historical provenance, not a stable public ABI.
