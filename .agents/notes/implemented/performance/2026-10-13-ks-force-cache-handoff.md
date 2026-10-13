# Decision: retire optional SCF ERIs only when force memory needs them

Status: implemented
Date: 2026-10-13

## Problem

The master-based README campaign converged PBE, PBE0, B3LYP and r2SCAN at
96 atoms / def2-SVP, then failed allocating analytic forces. SCF's optional
resident Direct ERI source reserves roughly 256 MiB of runtime headroom, while
the public ordinary stationary consumer has an additional 512 MiB envelope.
An admitted value cache can therefore leave insufficient room for the next
phase even though the mandatory calculation fits on a 32 GiB RTX 5090.

Admission depends on current device headroom. A diagnostic master replay on
another allocation passed with about 14 GiB free at force entry; that does not
invalidate the earlier failures. Fixed default runs with admitted ERIs entered
forces with only 234–252 MiB free and reclaimed 15,120,367,225 device bytes.

## Decision

Keep SCF cache admission and successful energy-only retention unchanged. The
current idle `CudaKsPlan` owns an authenticated, advisory force-memory handoff.
Before constructing force arenas, compare the additional consumer envelope
with `cudaMemGetInfo` on the bound device. If it exceeds free memory, drain the
SCF producer, invalidate graph borrowers and retire only the optional phase
source using its existing ownership path. Debit both its device and host
inventory. Successful SCF has already audited the source.

Native stationary, Python stationary and composite consumers use the same
native owner operation. Replays exclude retained force arenas from their
additional headroom. The private snapshot bridge takes the exact final-state
token under the context lock and reports requested/free-before/free-after/
reclaimed bytes. Python work and benchmark telemetry retain that receipt.
Older libraries and custom providers without the optional bridge keep their
existing allocation behavior.

## Invariants

- Active, failed or stale final states cannot reclaim a source. Capture cannot
  perform this handoff. No graph may retain a pointer to a retired source.
- Preserve resident D/W, final orbitals/XC, the exact token, and the last-good
  warm density and energy baseline. Successful SCF already rebases the warm
  energy onto the incumbent J/K route; retirement adds no hidden SCF solve.
- Only immutable ordinary ERIs are discarded. The derivative provider remains
  available and still computes the required derivative integrals.
- The handoff neither reserves global free memory nor changes an explicit
  budget. A conservative bound may exceed free memory even when an actual
  allocation fits; the consumer remains authoritative for allocation failure.
- Do not evict other calculations or silently retry scientific failures. There
  is no CPU reference fallback and no change to numerical acceptance gates.

## Rejected alternatives

Globally shrinking/disabling the SCF value cache would sacrifice the intended
SCF reuse. Unconditional post-SCF retirement would penalize energy-only and
fixed-density value consumers. Benchmark-only environment changes would leave
the public default failure intact. Arbitrary larger force allowances would not
create physical memory. The chosen boundary lets each phase use available
storage without confusing ERI reuse with derivative reuse.

## Evidence

The report's [force-memory archive](../../../../benchmarks/results/readme-dft-20261013/force-memory.json.xz)
binds source inventory `1de96258bf4ffdbff20b21854d055eac1b4fbc082ceb885b8478008f0da6bd16`
and the actual Release library. Evidence is retained locally under
`.artifacts/force-memory-20261013/`; original master evidence remains under
`.artifacts/readme-20261013/` and the unmodified report under the fix directory's
`master-report/`. GPU executions used finite Slurm allocations on node1,
preserving assigned device visibility. C++ builds used verified ccache 4.5.1;
runtime JIT used verified sccache 0.16.0 and retained existing caches.

- Default 12- and 96-atom PBE/PBE0/B3LYP/r2SCAN: 96 complete independently
  checked endpoints, including cold, frozen warm, moved and moved warm.
- A diagnostic CUDA allocation after SCF left 64 MiB free. Master returned OOM;
  the fixed PBE/96 consumer retired the optional cache and passed all 12 checks.
  Pressure-injected timings are excluded from the README.
- M06-2X and composite WB97M-V three-atom controls: 24 additional checked calls.
  Across these 132 fixed-build endpoints, maximum errors were 1.046e-11 Eh and
  9.159e-11 Eh/Bohr, within 1e-8/1e-7 gates.
- The real-device hybrid source probe checks roomy retention, stale-token
  refusal, pressure retirement, inventory debits, unchanged density/D/W leases
  and warm seed, idempotence and subsequent warm SCF. It retains independent
  PySCF J/K and energy assertions.
- Ten native CLI RKS/UKS LDA/PBE/PBE0/r2SCAN/B3LYP endpoints pass complete forces,
  translation invariance and fully reconverged finite differences.
- 140 host tests cover owner forwarding, typed failure, ctypes widths, budget
  admission/replay, composite resource planning and publication gates.

## Revisit when

Add cross-item arbitration only with explicit batch ownership and stream
lifetimes. Revisit the envelope if a new force owner introduces allocations
outside its current resource plan. This policy cannot make arbitrary mandatory
storage or concurrent external allocations fit; those failures remain explicit.
