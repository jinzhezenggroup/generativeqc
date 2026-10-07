# Decision: guarded CPU/CUDA MINAO is the Calculator cold-start default

Status: implemented
Date: 2026-10-07

## Problem

The repaired GPU provider has profitable complete cold E+force endpoints,
not just reduced iteration counts. The user requests default promotion and
CPU inclusion: MINAO's scientific construction is not specific to CUDA.
Global promotion of an explicit MINAO policy would instead reject existing
UKS, DF, ECP, mixed-precision and unsupported-element calls.

## Decision

The Python Calculator defaults to `initial_guess="auto"`, resolving MINAO for
FP64 restricted exact all-electron H-Ar CPU HF/KS and CUDA KS. Explicit `None`
is the stable Hcore rollback. Explicit HF/LDA/MINAO policies retain their
existing fail-closed domains and numeric controls; no new native ABI kind is
needed, and the low-level C API remains explicit-policy driven.

Use one element/ECP admission function for both the native descriptor and
resource inventory. A native batch shares its method descriptor, so any
unsupported item keeps the entire batch on Hcore. Do not change target
capabilities merely because automatic MINAO is a candidate. Existing explicit,
imported and retained warm densities take precedence; preparation failure/cap
exhaustion uses Hcore, and seeded target failure has one fresh Hcore retry.

## Rejected alternatives

- An unconditional MINAO default would break unsupported domains rather than
  preserving their existing behavior.
- Introducing a native auto kind would expand the ABI unnecessarily; the
  Calculator already owns backend/model and batch resource admission.
- Mixing per-item native policies in one batch would need new ownership and
  inventory contracts. A conservative whole-batch fallback is sufficient.
- Promoting HF/LDA preliminary SCF adds a second solve and is not authorized
  by the zero-Fock MINAO evidence.

## Evidence and limits

[The GPU owner repair](2026-10-07-cuda-minao-eigen-owner.md) records the frozen
qualification: strict FP64 exact PBE0/def2-SVP, three interleaved fresh-owner
samples per arm, independently gated E/forces, complete cold wall reductions
of 12.32%, 14.37% and 10.44% on water-48, water-96 and a peroxide holdout.
Those timings precede this default switch and are not measurements of the
post-switch source or the current master exact-K default.
The [qualification bundle](../../../../benchmarks/results/minao-default-20261007/README.md)
retains raw endpoint/reference records and the frozen source patch.

CPU uses the independently admitted occupied-ANO construction and reference
decompositions. Sharing scientific semantics does not establish equal wall
cost: CPU default promotion is requested policy, not a transfer of CUDA
speedup claims. Regression tests cover selection, Hcore rollback, unsupported
domains, batch/resource parity, endpoint equivalence and warm precedence.
The existing linear H3+ / STO-3G compatibility case is a retained negative
trajectory example: CPU automatic MINAO takes nine iterations versus six for
Hcore, while final energy/force gates agree. This is not a CPU wall-time
campaign and must not be presented as uniform iteration or runtime benefit.
The CPU/CUDA HF iteration-parity test now explicitly requests Hcore on both
backends, since CUDA HF deliberately remains outside automatic MINAO.

Default-path follow-up validation on parent
`7c07fa309cf7f9123cde697e460169e481bfca01` plus this change passes 236 focused
Python tests, 122 public Calculator/checkpoint/Hessian tests (two skips), and
native CPU-preliminary/real-GPU MINAO probes. Slurm job 6427 preserves device 1
on node1 with a 15-minute limit. Omitted-keyword water-48 and peroxide cold
E+force smoke pass independent GPU4PySCF gates, and water-48 warm replay skips
preparation. The bundle retains their JSON and unchanged start/end library
hash; these single samples do not replace the three-repeat historical campaign.

## Revisit when

Independent final-state/root gates regress, representative complete CPU or
CUDA endpoints show systematic slowdown, or broader domains become qualified.
Keep the explicit rollback available and retain all negative timing samples.
