# Decision: Bind CPU KS resource admission to compiled MethodIR identity

Status: implemented
Date: 2026-11-23

## Decision

Select the existing CLI CPU KS capacity layout by exact compiled MethodIR
identity. A table row's address changes across equivalent aliases; a carrier
enum can also be shared by unrelated graphs. Neither may decide the capacity
or authorize a derivative. Keep the native Hessian owner's existing linked
binding and context gates, and preserve the caller's original selector in the
full receipt hash. Publish the program's digest separately in
`method_ir_identity`, matching the execution receipt.

## Evidence

Before this change, PBEH, PBE1PBE, B3LYPG, and the generated LDA_XC_PW spelling
could select a linked CPU Hessian but could not size its primal/derivative
reservation together. Each now matches the canonical resource plan apart
from the display spelling and its full receipt hash. At the exact modeled
host peak the raw Hessian matches the canonical execution; one byte less
refuses before SCF. Named PBE-D3(0) still cannot size an unlinked derivative.
On the installed HeH⁺ endpoint at its exact 201,508,784-byte cap, ten paired
post-warm-up sample medians were 0.178 s before (PBE0) and 0.178 s after
(PBEH), with identical raw matrices and matching work: eight SCF Fock builds,
six directions, 24 induced Fock actions, and 18,432 response XC point visits.
Evidence lives under `/data/jzzeng/gqc-2227-sdk-JsefEq0x/native-cli-resource-alias`.
This does not change the CPU DF, CUDA, or process-RSS qualification boundaries.
