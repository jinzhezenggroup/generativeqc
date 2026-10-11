# Decision: compose resident Lambda vectors with the finished mixed warmup

Status: implemented
Date: 2026-10-11

## Scope

PR #2291 at `f85ef272d7b6adee7366b5976142805fd4a0c170` is integrated
with master `b30fa2da4ede0dabf27645ec93623186451b5fe4`, including the
already-landed #2278 mixed-warmup and scratch composition. Both native schedules
remain opt-in. No equation, convergence threshold, public default or precision
qualification changes.

## Epoch and storage boundary

Run the existing host-vector auxiliary mixed warmup before admitting the optional
resident-vector owner. Finish its single-use epoch, reject its nonfinite statuses,
and retain only its finite solution proposal. The original strict response plan
then consumes that proposal through either the resident or host controller, each
of which recomputes the physical FP64 residual. No approximate Krylov basis,
residual or mixed scratch contents are reused to certify convergence. The
independent transpose/RHS audit and terminal parameter/factor publication remain
strict. A finite exhausted warmup remains only a proposal; numerical and stream
failures still propagate.

The DF owner already charges the extra proposal vector and mixed descriptors.
Resident admission adds its vector arena and distinct vector-provider allowance
to the existing complete host/device and device-only ceilings. The two scratch
selectors stay independent; the resident owner is allocated after the mixed epoch
has ended and keeps the terminal owner's stream. Allocation/budget refusal retains
the same strict host solve and diagonal policy. The checked OOM/drain handler is
unchanged.

Move the independent-audit mixed-epoch guard from the host transfer wrapper to
the common transpose dispatcher. This preserves the existing prohibition for
both host and borrowed-device entry points; it is not a new scientific policy.

## Compatibility

Keep master's positional native-force parameters: reference experiment, serial
scratch, CCSD mixed warmup, Lambda mixed warmup; append default-off Lambda
residency after them. Retry paths forward all four independent selectors.
Existing benchmark command-line positions remain unchanged.

The test probe keeps master bits 4096/8192/16384/32768 for scratch and mixed
controls. Residency uses 65536. Its first 41 diagnostic entries retain master's
layout; append the 11 resident diagnostics and allocate 52 entries in the Python
adapter. Update the branch's resident assertions to those appended offsets.
Historical receipts keep their frozen source identities and bytes.

## Validation and limits

The combined tests retain all 78 Hamiltonian and 94 Lambda lifetime cases. The
probe packer/decoder exercises all 32 selector combinations. A production-extracted
controller probe links both actual GMRES implementations against deterministic
host storage, checks strict physical solutions after finite mixed exhaustion,
and covers mixed/nonmixed, resident/host, right preconditioning, budget/device
refusal, allocation refusal, nonfinite warmup and stream failure combinations.
Existing scratch admission, pointer-lifetime, policy and retry tests remain.

This is bounded host composition coverage, not new real-device qualification or
performance evidence. No GPU campaign, full build or default promotion is claimed.
Normal repository checks and the merge queue gate the actual merge.
