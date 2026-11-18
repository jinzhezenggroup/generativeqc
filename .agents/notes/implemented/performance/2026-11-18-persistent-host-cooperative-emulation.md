# Decision: use one sleeping 32-rank team for host cooperative-kernel emulation

Status: implemented
Date: 2026-11-18

## Problem

The host-emulation kernel in PR #3083 intermittently exceeds its existing
60-second subprocess cap on hosted CI. Run `44134150903` fails the
double/96-atoms/1088-AO case; after reusing a warp across blocks,
`44218958302` passes but `44224991871` fails double/128-atoms/1024-AO.
An isolated local pass does not establish that the scheduling problem is fixed.

The previous harness still constructs a new 32-thread team for every replay in
the complete parameter/failure matrix, and the standard library barrier may
spin while other ranks wait for a CPU time slice. This is CPU emulation of a
GPU warp, not a GPU execution or a performance qualification.

## Decision

Use a single RAII-owned 32-thread team for the entire generated executable.
Dispatch each synchronous replay to that team, retaining the completion barrier
between simulated blocks. A generation-counted condition-variable barrier puts
waiting ranks to sleep rather than spinning under oversubscription. Dispatch
barriers protect the callback and shutdown flag, and the destructor joins every
rank, including when the numerical test returns an early failure code.

Keep all 32 ranks, every original atom/AO/precision parameter, tail and ownership
case, independent serial comparison, sticky-error injection, buffer sentinel,
translation check and the original 60-second cap. Do not shrink the scientific
matrix, reduce the simulated warp size, or count a timeout as a pass. The
separate adversarial vote-ordering test retains its delayed reader and real
concurrent participants.

## Evidence

- All 21 cases in the complete host-emulation module pass in 70.52 s locally.
- Four concurrent copies of the formerly failing double/128/1024 executable,
  pinned together to only two CPU cores, all exit successfully under the same
  60-second cap: 8.47, 8.44, 8.45 and 8.46 s. This exercises 128 live emulated
  ranks on two CPUs without changing the kernel work or rank count.
- Receipts are under `/data/jzzeng/pr3083-ci-20261118/remaining/`, especially
  `persistent-cooperative.log` and `persistent-stress-*.time`. Fresh hosted
  qualification is still required; these receipts do not establish GPU parity.

## Rejected alternatives

Merely increasing the timeout retains avoidable OS-thread churn and spin-wait
contention. Merely reusing a team across blocks still leaves repeated creation
across all error replays, and has already failed on hosted CI.

## Revisit when

A shared host CUDA-emulation scheduler can preserve the same synchronization
and failure coverage without introducing a second scientific implementation.
