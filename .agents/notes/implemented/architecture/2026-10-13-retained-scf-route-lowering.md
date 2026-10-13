# Decision: share retained density and DIIS coefficient lowering

Status: implemented
Date: 2026-10-13

## Problem

The ordinary HF/KS structured bindings already exist, but DF-SCF still carried
an independent occupied-density contraction. CUDA DIIS also reconstructed its
Fock history in native code even though CPU already consumed the canonical
`diis_extrapolation_program`. These are ownership gaps in #1875 and #1874,
independent of any profitable vendor schedule.

## Decision

DF binds a runtime-scalar variant of the existing generated occupied-density
kernel. The common weighted-Gram recognizer admits that physical candidate
against the same scientific equation. Its dimensions remain `size_t`; there is
no narrowing through the ordinary SCF launch ABI. A scalar template variant
preserves multiplication by arbitrary weights, including zero and negative
weights, without duplicating the orbital loop.

Move CPU DIIS coefficient emission into `tensor.history_combination` and add
the CUDA indexed-element schedule under that same canonical request. Native
ring views map chronological indices after retirement; the compiler owns the
ordered reduction. CPU retains its history-major output loop, while CUDA
retains the existing per-element local accumulator and enclosing FMA policy.
The emitter does not own solver success, history mutation or publication.

## Rejected alternatives

- Replacing DF density with a new method-local GEMM/SYRK selector would retain
  the ownership problem and require separate numerical/default qualification.
- Calling the ordinary `int32_t` launch adapter would narrow DF's dimensions
  and lose its scalar-weight and clipped-launch contract.
- Moving the complete DIIS solver into TensorIR would mix history retirement
  and singularity policy into a pure contraction and broaden this change.
- A new ring-specific scientific equation would duplicate the existing
  chronological history contraction merely to represent a physical layout.

## Invariants

Preserve FP64 product/reduction order, occupied-prefix reads, publication masks,
singular fallback, ring retirement/wrap and all resource lifetimes. DF mirrors
only complete-domain launches at weights 1 or 2; partial launches must neither
skip their original writes nor write outside the launch domain. Poisoned
virtual orbitals and retired history slots must not enter the contraction.

## Evidence

`test_scf_retained_lowering.py` compiles actual emitted code with the cached C++
fixture. The executable compares DF density with the original scalar order and
an independent long-double oracle, and compares CPU/CUDA history arithmetic for
capacities through 64, every live-window start/count, spin-sized vectors, empty
windows and poisoned neighboring/retired slots. Candidate tests verify shared
science, zero workspace and fail-closed rejection of modified equations.

Host execution does not qualify real CUDA scheduling, Graph replay or complete
SCF endpoints. This ownership change makes no performance claim or library
promotion; the broader issue acceptance remains separate.

## Revisit when

A shared provider has independent scientific and endpoint evidence supporting a
new schedule. Keep the retained ordering available wherever it remains required.

## References

- #1874, #1875, #762, #926
- [Current provider contracts](../../../../docs/developer/lowering_providers.md)
