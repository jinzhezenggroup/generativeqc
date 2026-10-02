# Decision: charge the CUDA force Hamiltonian provider before execution

Status: implemented
Date: 2026-10-02

## Problem

The prepared-source force cutover selected the CUDA MO transform while its
complete force admission still used the CPU provider plan. That omitted the
existing generated CUDA provider's device buffers and library allowance. Passing
the complete endpoint budget again to the nested provider also failed to express
the CC, triples, Lambda and parameter outputs still live beside that provider.

## Decision

Use the existing generated `numeric_block_plan` with the executing transform
backend. Add its host and device costs to the raw-Hamiltonian phase while keeping
the previously retained outputs charged. Give the nested provider the exact
allowance admitted for its own buffers and borrowed reference/source, rather than
the complete endpoint budget. Public CPU planning remains CPU-only.

This extends the phase composition documented in
`2026-09-23-rccsdt-force-phase-capacity.md`; it does not change CC, response,
integral or force equations. Source identity validation precedes the complete
admission, which precedes numerical force work.

## Evidence

`test_cc_force_cuda_provider_admission.py` extracts the live complete planner and
uses the real generated provider resource contract. With other response scratch
isolated, it covers every occupied/virtual split for 2 through 12 AOs, with and
without triples, and verifies backend cost, nested allowance, exact admission
and one-byte-short rejection. Existing native CPU allocation and independent
gradient tests remain the numerical gates. This host test does not qualify GPU
numerics or performance.

Agent: dot
