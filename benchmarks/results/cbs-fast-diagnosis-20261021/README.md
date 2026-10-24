# CCSD(T)/CBS Fast bottleneck diagnosis

This follow-up to PR #2648 investigates why the published Fast mode has limited
complete energy-and-force speedup. It does **not** promote a new default or
replace the homepage measurements. The runtime changes were experimental;
the production source was restored after the screen.

## Matched clean endpoints

All calls below use the same n4 RTX 5090, finite Slurm job 1283, one numerical
CPU thread, a 32 GiB numerical-capacity limit, and the original QM7 geometries,
DZ/TZ orbital bases and RI-JK model from the [homepage campaign](../readme-cbs-fast-20261020/README.md).
Every observation starts a fresh calculator in a fresh process. Timers include
both basis groups and complete returned energy and analytic forces. Python/DSO
startup and basis metadata preparation are outside the timer. No reference
state enters production. There is one cold observation per configuration/case.

| Case | Standard (s) | Current Fast (s) | Lambda warmup, Q32 (s) |
| --- | ---: | ---: | ---: |
| C4 | 99.386156 | 86.979990 | 80.338942 |
| C2 | Not measured here | 11.343354 | 11.042549 |

The candidate saves another 7.64% on C4 and 2.65% on C2 relative to current
Fast. C4's Standard/candidate ratio is 1.237x, versus 1.143x for current Fast.
These are descriptive single observations, not statistically qualified speedups.
Execution order was C4 Standard, candidate, current Fast, then C2 current Fast,
candidate. The four diagnostic calls below are excluded from this table.

## Disjoint Lambda phases and work

Separate jobs 1279–1282 record the existing host progress journal. These host
phase intervals include their existing stream drains and trace I/O; they are
not kernel profiles. Immediate children are disjoint within the Lambda parent.
Do not add nested GMRES/action time to its parent or add provider time to setup.

| C4 diagnostic | Lambda (s) | GMRES, nested (s) | Audit, nested (s) | Solver actions |
| --- | ---: | ---: | ---: | ---: |
| Current Fast | 31.697239 | 20.245509 | 4.827615 | 16 FP64 |
| Existing Lambda warmup, Q16 | 28.988467 | 15.730195 | 6.325417 | 12 warm + 7 FP64 |
| Lambda warmup, Q32 | 25.090154 | 13.626973 | 4.841178 | 12 warm + 7 FP64 |
| Q32, tighter warmup handoff | 26.657870 | 15.185385 | 4.843311 | 19 warm + 6 FP64 |

Current Fast's full diagnostic endpoint is 86.900599 s: DZ 58.241411 s and
TZ 28.657739 s. Its DZ CCSD solver is 8.243206 s, including 4.570605 s of
iteration graphs and 3.565764 s of independent replay; triples take 12.299521 s.
Lambda therefore matters more than further CCSD-only warmup tuning in this case.
The TZ group is also a substantial fixed cost for these Lambda-only candidates.

The native mixed Lambda option halves its requested Q cap. Merely enabling it
changes Q32 to Q16 and slows the final FP64 audit and derivative contractions.
The second candidate requests cap 64, preserving actual Q32 here. Per physical
action, the observed auxiliary-batch count returns from 32 to 16 for 510 Q rows;
strict GEMMs return from 951 to 519. Warm actions report 87 **FP64** GEMMs plus
the auxiliary mixed graph; the journal does not export its FP32 GEMM counter.
Action counts include initial and exact-residual actions, not just iterations.
Only a finite warm solution crosses into a new FP64 solve and independent audit.

The final candidate tightens the warmup handoff from `1e-6` to
`max(1e-8, selected_lambda_residual)`, here `2.3827302298e-8`. It exhausts the
existing 16-iteration bound, adds exact-residual checks and saves only one FP64
action. It is slower than the Q32 candidate. Extending FP32 work is not by
itself an optimization.

## Accuracy and limits

Every C4 Fast observation passes all constituent and composite full-vector
controls against the newly measured Standard. The clean candidate's composite
energy difference is `7.93e-10 Eh`, maximum force difference `3.03e-10 Eh/Bohr`.
Compared with current Fast, its energy is identical at recorded precision and
maximum force difference is `1.24e-11 Eh/Bohr`. Standard is not an independent oracle.

Both C2 clean calls pass the unchanged independent PySCF constituent/composite
energies and normalized force projection at both `1e-4` and `3e-5 Bohr` steps.
The candidate's maximum constituent energy error is `6.29e-10 Eh`, directional
error `5.42e-9 Eh/Bohr`. The original oracle and exact protocol are retained in
`samples.json.xz`. This is directional coverage, not full-vector independent
qualification. No larger shape, changed geometry, constrained-memory cohort or
broader device qualification was measured for the candidate. Promotion remains
inconclusive. Existing physical gates and public Fast defaults are unchanged.

## Provenance and replay

The native base is `6339327feda0e28d961f45f3c6f5944730f1024d`; exact experimental
patches and library hashes are retained. Python uses the prior frozen homepage
deployment (its sole difference from the base is the already documented public
docstring correction). Builds reuse verified ccache 4.5.1, GCC 11.4.0 and CUDA
12.9.1 Release/sm120. Cache statistics are shared-cache observations, not isolated
campaign hit counts. GPU UUID, driver, power limit, controls and Slurm IDs are
in each complete receipt.

`profiles.json.xz` retains every original record in the Lambda subtree, plus
the full ignored journal's checksum/size. Other phase journals remain in
`.artifacts/cbs-fast-diagnosis/`; they are not needed to replay the Lambda
conclusion. All nine completed receipts and three candidate patches are kept,
including the slower handoff experiment. The existing README curve is unchanged.

```bash
PYTHONPATH=.:python python benchmarks/results/cbs-fast-diagnosis-20261021/verify.py
```

`run-candidate.sh` and `clean-cohort.sh` retain the exact deployment commands;
adapt paths after rebuilding each candidate from the retained patch. Always
run through finite `srun`, preserving assigned GPU visibility. The diagnostic
commands use the same runner with argument 3 set to `1`; the initial baseline
used the same native command with the original `python-preload.sh`.
