# Decision: do not promote the exact HF page-tail screen

Status: rejected default promotion; experimental production changes removed
Date: 2026-11-08

## Problem and candidate

HF Direct96 remains the largest A100 README cold gap. Its bounded owner already
sorts current-geometry pair-class/system segments by descending Schwarz bound.
The experiment used a GPU binary search to skip the exact-class compactor's
provably rejected ket suffix, preserving multiplication order, strict comparison,
the force `min(screening, 1e-14)` ceiling and every survivor's exact predicate.
No threshold, convergence rule, precision, oracle dependency or allocation changed.

## Decision

Do not add a new opt-in or promote this implementation. Both measured cards show
only about 1.4% lower HF96 cold latency, below the existing 2% promotion floor,
and the A100 remains far behind GPU4PySCF. More repetitions cannot turn this
observed effect size into a sufficiently large improvement. Retain the negative
evidence and pursue the expensive strict K applications, not this small screen.
The original eight changed/added prototype paths are restored; the discarded
code and an unmeasured safety refinement remain in ignored local artifacts.

## Evidence

Frozen parent `7f99fa282c5c400a274d4063c539f4e28d0ce824`, native source identity
`b4a20a0db5c6c4c23ccbae59ab5fe0ded6d577537491e580dc6a0c1d5ed12a71`,
normal Release SM80-real/SM120-real ELF SHA256
`025aedf664038733cccd831f77864c9a872fe759c8c64ebe9f6ea6b8a125a1da`.
Master provider/convergence integration is actually compiled, not an old image
given a new revision label. The borrowed build source/runtime is restored.

| Card / HF atoms | Cold off / on / GPU4PySCF (s) | Cold reduction |
| --- | --- | --- |
| A100 / 48 | 6.238705 / 6.210157 / 4.927945 | 0.458% |
| A100 / 96 | 16.118055 / 15.900522 / 5.661240 | 1.350% |
| RTX5090 / 48 | 6.261144 / 6.261647 / not measured | -0.008% |
| RTX5090 / 96 | 16.848177 / 16.609672 / not measured | 1.416% |

Every endpoint includes construction/preparation, SCF, analytic forces and host
export. Three warm calls replay each engine's own frozen cold density. Native
HF48 has seven preliminary plus ten target Focks; HF96 has seven plus fifteen.
Both arms retain those counts and one target Fock on every warm call. Reference
HF48/96 each takes ten SCF iterations. Individual candidate/rollback warm ratios
are all below 1.03; no warm outlier was dropped.

Slurm CPU build16762 succeeds. Initial CPU stage16764 has 20 passes/four setup
errors because the runner omitted its available compiler cache. Recovery16769
sets `CCACHE` to the installed sccache, retains the failed log and passes all24
host policy tests without changing any test or native image. Slurm16765 runs
the four complete native populations through finite srun. LSF CPU setup228573
and exclusive A100 screen228574 finish successfully; device visibility is kept.
Independent real-kernel task censuses on5090/16766 and A100/228574 each report
96 comparisons,169804 admitted tasks,zero differences. This is an s-shell queue
fixture, not complete molecular independent force qualification.

All16 native off/on cross-comparisons per size/card have energy discrepancies
at most3.638e-12 Eh and force discrepancies at most5.017e-13 Eh/Bohr. Default
GPU4PySCF forces differ from native by up to2.426e-7 Eh/Bohr: these samples
pass the existing shared comparison, not the tighter independent1e-7 diagnostic.
No new tighter oracle qualification or default promotion is asserted.

## Safety and retained failures

Before promotion, ordinary `>` sorting would need a strict-weak nonfinite-prefix
comparator, and the binary search would need nonnegative finite bra/density
guards. A local follow-up implements these and passes its host comparator plus
resource-identity tests (9 passes), but was never GPU-measured. It is not part
of the frozen image, cannot inherit its timings and is removed with the loser.
An overly broad local host selection additionally has11 missing-native-library
failures; they are retained, not reported as a successful full suite.

The first result-list helper exits141 from `tar | head` after receiving a valid
DONE/exit0 callback. A later packaging command exits2 because compute-private
LSF stdout files are not visible on login. The valid archive contains the actual
worker driver, original callbacks and submission receipts; a separate check
authenticates its complete member list and bytes. Neither helper failure causes
a replacement benchmark or a duplicate submission. SCP returns numeric exit0.

Local campaign: `.artifacts/page-schwarz-tail-v1-20261108/`. Original A100 callback
SHA256 `4ad03c8e2fe3dcbd196c884024c74d875a5aea9e94ca9b8b837fd9ef6e95475e`;
complete receipt archive
`19dddc8b72ce8d40b9f25fc8d8729303cef75b8ba082605a96d11beb8cacf75d`;
RTX5090 archive
`51ccc3e050952365dc35a1e7c0bc964b85de1f4a42d4b9124d608aacae57f430`.
Issue2976 retains the decision. The original64-pair baseline is not rewritten.

## Revisit when

A materially different queue algorithm removes substantially more complete
endpoint work, or a different independently qualified workload establishes a
gain exceeding the existing promotion rule with all scientific/card guards.
Do not rerun this exact binary-search proposal merely with a new timing order.
