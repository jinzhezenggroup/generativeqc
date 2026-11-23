# Decision: Use the native CPU Hessian byte floor in the CLI

Status: implemented
Date: 2026-11-22

The dry-run-only host policy below is superseded by
[CLI CPU Hessian host admission](2026-11-22-cli-cpu-hessian-host-admission.md).

## Decision

`hessian` and `hvp` ask the existing SDK query for mandatory bytes on the
same normalized system that will enter the named KS batch. Insufficient
derivative capacity fails before batch preparation. The CLI neither expands
the basis twice nor counts primitive/quartet work during the query.

`resources --property hessian|hvp` uses the same option decoder, total program
resolution and resource query. It composes the byte floor with the declared
primal host peak, including any supported D3 owner and HVP direction staging.
No retained storage is subtracted on the assumption that SCF releases it.
No derivative equations or allocation formulas are shared by copying them
into the CLI. The dry-run only accepts the intersection with its existing
SCF/correction inventory. B3LYP remains executable where linked, but it is
not a supported derivative dry-run without a matching primal inventory.
That B3LYP inventory was subsequently added; see
[the B3LYP resource decision](2026-11-22-cli-b3lyp-host-inventory.md).

The dry-run reports modeled mandatory host capacity; it does not reserve
that capacity, admit optional response panels, or promise that solver and
work limits will fit. Its `--host-bytes` is not a CLI execution cap.
`--numeric-budget-bytes` caps the derivative itself during execution.

## Evidence

The CLI contract checks the exact host limit with a host reserve, the
one-byte-short host and derivative limits, and dry-run/runtime agreement for
HVP and dense PBE and PBE0-D3(BJ) RKS. An admitted one-step SCF fails to
converge; the one-byte-short run refuses earlier and emits no result.
Unsupported CUDA derivative inventory is refused without a device query.
The resource query and the installed CLI run without Python or a compiler
on `PATH`. This advances #2227 R2/R5; neither slice is complete.

The installed CLI passed all 17 Hessian contract checks; eight selected CPU
CTests passed, including the SDK and cross-frontend numerical parity. On an
AMD EPYC 7K62 with one BLAS/OpenMP thread, five interleaved process pairs per
case exercised both distorted STO-3G water geometries. Dense LDA, PBE and
PBE0-D3(BJ) changed by -0.01%, +0.07% and +2.22% in median endpoint time.
PBE0-D3(BJ) and B3LYP-D3(BJ) HVPs changed by -0.10% and -0.02%.
Raw derivatives, energies and reported SCF/response/ERI/XC work counts were
identical in every before/after pair. Executable/library hashes and complete
receipts are in `/data/jzzeng/gqc-2227-sdk-JsefEq0x/cli-reservation-bench`.
