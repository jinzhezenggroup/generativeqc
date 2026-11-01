# Decision: capability-selected bounded FP64 XC and occupied Gram products

Status: implemented
Date: 2026-11-01

## Problem and evidence

The frozen three-device DF96 integration still has five A100 cold endpoints
behind GPU4PySCF despite all seven warm endpoints leading. Generated triangular
occupied Gram and compact XC scheduling were primarily qualified on Blackwell;
those choices need not minimize complete A100 endpoints.

A100 224212 compares generated Gram, SYRK, and full typed FP64 GEMM with public
SCF tile1024 and force tile1024. Conditioning precedes two complete cold/three-warm
populations per route in generated/SYRK/GEMM/GEMM/SYRK/generated order. GEMM cold
changes are -4.3225% B3LYP, -3.7417% PBE0 and -3.9355% M06-2X, with unchanged
physical iterations and Fock counts; all populations pass the independent energy,
full-force, grid-stability and CPU/GPU spot gates. M06-2X also enables the earlier
XC prototype in every Gram arm. These are diagnostic combinations, not a default
production ranking. Failed script 224210 is retained alongside the repaired run.

A100 224205 separately measures XC GEMM at tile256: B3LYP regresses 1.5243% cold
and 0.5347% warm, whereas r2SCAN-3c improves 10.0960% cold and 3.7070% warm at
unchanged SCF work. Do not globally promote the GGA library path. Its independent
WB97M-V provisional-tolerance experiment reduces cold time 4.5226% and 28 to 25
iterations; that solver change is not part of this implementation.

## Candidate implementation and invariants

The TensorIR symmetric cross product has a fourth, separately identified indexed
GEMM recipe. It retains the exact existing XC panel algebra, materializes one
FP64 cross product, and publishes X + transpose(X) through the shared indexed
scatter. The diagonal contributes twice. Dense and indexed SYR2K recipes retain
their original identities and behavior. A generated callback remains available
for unsupported, empty, unavailable, or resource-constrained domains.

A preparation-owned handle executes only typed FP64 Dgemm; its default math mode
cannot introduce TF32 or FP32 input arithmetic. No global handle changes or casts
are allowed. Cache allocation, opaque-provider allowance, stream/device lifetime,
integer extents, alias rejection, graph replay, finite publication and sticky
errors remain checked. Preparation must precede capture; a runtime failure is not
silently retried as a different scientific calculation.

The candidate KS policy only considers large local strict-FP64 panels (at least
512 AOs, 2^19 points and four work jets), full FP64 throughput (positive FP32/FP64
ratio at most two), compute capability major at least eight, and a bounded
128 MiB optional allowance. The architecture condition prevents pre-Ampere
full-rate scalar FP64 devices from being mistaken for FP64 matrix hardware.
Public numeric
ledgers and explicit reference-memory budgets retain their incumbent. Final-source endpoint qualification is retained on Blackwell. A100 endpoint
measurements below retain the earlier source identity; final-image A100
qualification remains pending the unavailable SSH connection. User point tiles
and independent force tiles are preserved exactly.

The Gram candidate retains the original generated shape/capacity admission and
borrows the already charged output. Within that domain, full-throughput FP64 can
select the complete library Gram only with the same matrix-capability predicate;
older and low/unknown capabilities keep generated splits.
Capability acquisition is independent of explicit projection controls. The final
U response lease is untouched, and FLOP counters report the full second triangle.

## Validation so far and remaining work

The four-recipe binding passes independent long-double scalar, spin/tail/map,
CPU XC, overwrite/accumulate, capture replay, unavailable-provider, insufficient
allowance and numeric-ledger exhaustion cases on RTX 5090 (Slurm 1414) and PRO6000
(Slurm 3845). RTX 5090 memcheck reports zero errors and zero leaks. These checks
precede the method-level automatic policy and are not endpoint qualification.
Compiler structure/type checks and 28 targeted host policy/algebra cases pass.

The final-source 30-case independent packed-DF suite passes on RTX 5090 and
PRO6000. Complete default-grid Blackwell endpoint evidence is linked below.
Still required: final-source A100 endpoints and independent force validation. No claim of
universal GPU performance follows from the throughput ratio alone. Keep all
pilot, failed, slower and successful populations with their original identities.

The initial A100 combined image `a0de46b3f44a7d046e2dc256066217030bfb65642f8188462aecf5f90ddcc9e1`
passes the independent scalar/map/capture/resource suites and nine PySCF
cold/warm/moved/force cases in allocation 224220. It predates the architecture
restriction; do not relabel its native identity as the revised implementation.
Allocations 224221 and 224222 retain complete ABBA default-tile populations.
Cold differences against GPU4PySCF are -4.6131% PBE, +16.9650% PBE0,
+12.9591% B3LYP, -14.8467% r2SCAN, +15.5304% r2SCAN-3c and +3.5474% M06-2X.
All six warm comparisons lead and all populations pass the independent gates.
These results leave four measured cold deficits; they do not complete the
requested seven-functional performance target. The original failed startup and
old-Bash empty-array runs remain in the ignored qualification artifacts.


## Final-source Blackwell controls and rejected batching change

The [complete campaign replay](../../../../benchmarks/results/capability-fp64-20261101/README.md)
retains every original failed, slower, instrumented and successful population.
The final native identity is
`0b5701fcc3509f9b9df0b85d45dca76a9d5fab4052fa23d51cad25f797a1099a`.
No GPU product-name whitelist is introduced. Low/unknown FP64 capability and
pre-Ampere hardware retain their incumbent schedules. This is bounded selection,
not a guarantee that all unmeasured GPUs improve.

Slurm 1425 (5090) and 3855 (PRO6000) compare internal XC batches 64/128/256
without changing the 1-GiB ceiling or public SCF/force tiles 256/1024. Each arm
has conditioning followed by forward/reverse cold plus three-warm populations.
Separate instrumented calls record actual admitted work. Requests 128 and 256
both admit 128 tiles and 745,828,768 bytes, compared with 64 tiles and
400,867,296 bytes. Each XC evaluation still visits 275,531,328 point/AO entries
and 74,247,183,680 point/AO-square entries.

At 128 requested tiles, cold/warm changes against 64 are +0.0344%/-0.0068%
on 5090 and +0.1081%/+0.2089% on PRO6000. Request 256 likewise brings no gain.
All complete endpoints pass independent energy/full-force gates, and paired
maximum force differences remain below 2.55e-12. Wider batching retains more
memory without reducing this measured endpoint, so the default remains 64.
Do not infer an A100 batching optimum from these Blackwell observations.
