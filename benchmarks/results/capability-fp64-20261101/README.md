# Capability-selected FP64 XC and occupied Gram evidence

The performance target remains incomplete. The combined A100 experiment still
has four measured cold deficits, and its WB97M-V ranking has not been recovered.
Final-source Blackwell controls and the earlier A100 combined image have separate
identities below. No result establishes universal GPU or geometry performance.

## Complete default-grid DF96 comparisons

Each cell is **cold / warm** `100 * (median(native) / median(GPU4PySCF) - 1)`;
positive means native is slower. Warm uses all six observations. Cold includes
construction, preparation, SCF, analytic forces and host export. Compilation and
process caches are conditioned consistently before the measured populations.

| Method | A100 (earlier image) | RTX 5090 (final source) | PRO6000 (final source) |
| --- | --- | --- | --- |
| PBE | -4.61% / -5.42% | -8.43% / -12.88% | -18.27% / -9.05% |
| PBE0 | +16.97% / -9.79% | -7.88% / -43.20% | -13.37% / -41.79% |
| B3LYP | +12.96% / -7.26% | -12.07% / -42.60% | -16.92% / -41.02% |
| r2SCAN | -14.85% / -22.07% | -15.90% / -32.04% | -22.18% / -29.53% |
| r2SCAN-3c | +15.53% / -6.76% | -0.04% / -32.22% | -4.23% / -29.83% |
| M06-2X | +3.55% / -18.62% | -12.57% / -48.98% | -16.30% / -47.59% |
| WB97M-V | not recovered | -41.37% / -54.06% | -36.69% / -63.18% |

The 5090 r2SCAN-3c cold difference of -0.04% is effectively a tie. Native cold
observations are 23.151176/23.149546 s and GPU4PySCF 23.150878/23.168482 s; this
margin does not establish a reliable win. A100 entries are from the earlier
combined source, and missing WB97M-V observations are not filled with an older
revision's ranking. The prior snapshot remains in
[portable integration](../portable-integration-20261101/README.md).

Every method has native/reference conditioning, then native/reference/reference/
native fresh owners with cold plus three frozen-owner warm calls. Only complete
method cohorts enter the table. Slurm 1422/3852 later failed when the deployed
GPU4PySCF 3c correction helper was missing; their first four methods completed
full cohorts. Slurm 1424/3854 separately measure 3c and M06-2X after deploying
the authenticated helper. No ratio joins samples across allocations.

The endpoint keeps stock semilocal/VV10 quadratures, explicit native SCF/force
tiles 256/1024, spherical basis functions, single-threaded math, each engine's
production solver and method corrections. The independent energy/full-force,
grid-stability and CPU/GPU spot gates are unchanged. References retain their
original producer and package metadata through the explicit reference-reuse
contract. This endpoint qualification covers the sampled static geometries;
separate component and moved-geometry tests have narrower, named scopes.

## Native identities and executable checks

Final native source:
`0b5701fcc3509f9b9df0b85d45dca76a9d5fab4052fa23d51cad25f797a1099a`.
Blackwell library:
`806ac8213bd4b676f085505b58b404136350b0a8be3534e9c6033d8e3b9bcbf1`.
The reconstruction uses base `e5faf350a91f56c68f48d9f0d9baec0c85be3d31`, the
retained patch and the added `cuda_fp64_matrix_policy.hpp` text. Compiler/provider
and stationary-extension identities are retained separately. Blackwell rankings
are Slurm 1422/1424 on n4, 1950 on n5 and 3852/3853/3854 on n2.

The earlier A100 combined source is
`a0de46b3f44a7d046e2dc256066217030bfb65642f8188462aecf5f90ddcc9e1`,
library `a6e136da86d2ecf03f47a137cc8f9cd97cb222fed8790d50c79dee6d53c8b078`.
It lacks the final architecture restriction and independent Gram capability
acquisition. Original overlay hashes and the retained reconstruction patch bind its 2,036
native source inputs to base `5adef0250199a6efbe0bc41156266c6c4b702cd7`; a one-time
Git-object reconstruction reproduced that measured identity exactly. Its six
comparisons are LSF 224221/224222 on gpu22. They are not final-image A100
qualification.

Independent scalar/long-double, CPU local-map, spin/tail, overwrite/accumulate,
capture and resource-failure tests cover all four XC recipes on 5090/PRO6000.
5090 memcheck reports zero errors and leaks. The final 30-case packed-DF CUDA
suite passes on n4/n5/n2. A100 allocation 224220 passes the component suites
and nine PySCF cold/warm/moved/force cases on the earlier combined source.
Host capability tests preserve pre-Ampere, unknown and low-FP64 routes. The
repository Libxc qualification snapshot remains explicitly stale; these seven
methods do not requalify the entire catalogue.

## Pilots and rejected wider batching

A100 224205 uses the experimental indexed GEMM prototype: B3LYP cold/warm
regresses 1.52%/0.53%, while r2SCAN-3c improves 10.10%/3.71%. This rejects an
unconditional GGA promotion. Its independent WB provisional-tolerance control
changes solver work and is not attributed to the XC kernel. Pilot metadata is
not relabelled as the new four-recipe provider identity.

A100 224212 compares generated/SYRK/GEMM occupied Gram at explicit public SCF
and force tiles 1024/1024. GEMM improves cold B3LYP/PBE0/M06-2X by
4.32%/3.74%/3.94%, with the same iterations and Fock work. The M06-2X arms also
all enable the XC prototype. These are paired scheduling diagnostics; their
percentages cannot be multiplied into the default-grid ranking.

Slurm 1425/3855 compare internal XC requests 64/128/256 on 5090/PRO6000 at
unchanged public tiles 256/1024 and unchanged 1-GiB capacity ceiling. Each arm
has conditioning, two cold/six warm calls in forward/reverse order, and a
separate instrumented call. Actual admitted tiles are 64/128/128, with retained
storage 400,867,296/745,828,768/745,828,768 bytes. Every XC evaluation retains
275,531,328 point/AO visits and 74,247,183,680 point/AO-square work.

At request 128, cold/warm changes versus 64 are +0.0344%/-0.0068% (5090) and
+0.1081%/+0.2089% (PRO6000). Request 256 brings no improvement. All numerical
gates pass; paired maximum force differences are below 2.55e-12. The production
default remains 64 because the larger allocation provides no endpoint benefit.
This does not establish the best batch for A100 or another unmeasured device.

## Retention, limitations and replay

The lossless v3 streams preserve every original endpoint, incomplete receipt,
slower control, source/driver/build/cache log and component ledger collected for
this campaign, including recovery snapshots. Duplicate snapshots keep their
paths and checksums; the report counts distinct complete receipt bytes separately
so copying a recovery result does not create another measured observation.
Deployment failures (library soname, runtime assets and correction helper), the
A100 loader-path failure and old-Bash empty-array failures remain unranked.

A100 LSF 224223 (WB ranking), 224225 (single-control full-target initialization)
and 224229 (B3LYP/3c profiles) were submitted but their final state and original
archives remain unavailable after VPN SSH handshake resets. No new measurements
or successful exits are inferred. Their submitted scripts are retained.

The complete profiled A100 224224 endpoint and component ledger are retained.
Its optional 18.85-MB progress timestamps and 5.93-MB py-spy sample stream remain
in ignored local artifacts with exact byte counts and hashes in this packet.
They were not deleted and are not presented as Git-retained raw traces or inputs
to the independent ranking gates. Parent/child and host/GPU intervals overlap
and must not be summed as exclusive endpoint components.

```bash
PYTHONPATH=python:. python benchmarks/results/capability-fp64-20261101/report.py
```

The replay authenticates the original text/FP64 words, source/binary and actual
scheduler identity, all scientific gates, full Fock timelines and complete
comparison populations. It prints every timing observation, median/range/MAD,
semantic work, incomplete receipts and admitted XC batching counters. All checks
remain active under `python -O`. No external archive or Release is required to
replay the retained endpoint evidence. The 80-MiB repository cap is unchanged.

This snapshot authenticates 272 complete receipt paths representing 219 distinct
complete receipts and 714 complete energy-and-force calls.
