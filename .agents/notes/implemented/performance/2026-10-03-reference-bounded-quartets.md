# Decision: use bounded exact quartets for uncached CUDA RHF references

Status: implemented
Date: 2026-10-03

## Problem

Physical-reference export disabled ordinary quartet dispatch. The optional
s/p ERI cache helped smaller correlation calculations, but hundreds-AO d/f
references still recomputed individual public-AO integrals inside every
matrix-direct Fock element. Independent 230/264-AO molecular jobs exhausted
two-hour allocations without publishing a converged state.

A separate child-launched CUDA debugger sample of the 230-AO input identified
`build_fock_direct_packed_kernel` inside `contracted_eri_cartesian`, with a
52,900-block grid. Earlier 40-second Nsight traces contained only completed
startup kernels; they did not measure the outstanding first Fock kernel and
must not be interpreted as its timing breakdown. A host sample in cuSOLVER's
device-to-host copy was waiting on preceding stream work, not evidence of an
eigensolver arithmetic bottleneck.

## Decision

Use the existing bounded exact quartet owner when a physical reference has
d/f shells, or s/p topology larger than the optional 256 MiB ERI cache. A pure
policy shared by host packing and device setup makes that decision. The host
must provide the public-to-Cartesian transform even for small spherical d/f
systems: merely removing the reference guard in device dispatch fails the
frame-consistency gate. The bucket may include this policy header, but still
cannot include the physical-reference export implementation.

Bounded quartet scheduling avoids a full quartet descriptor array. It reuses
the existing integral tiles across Fock outputs and performs density/Fock
frame transforms at matrix level. No new integral formula, eigensolver,
reference solver or CPU numerical fallback is introduced. This remains a
four-center conventional RHF Hamiltonian; changing it to DF-RHF would change
the correlation-only DF method.

## Invariants and compatibility

- Unscreened strict FP64 and generated/native shell-class coverage remain
  unchanged. The existing physical P/F/C/epsilon stationarity gates still
  control publication.
- Cache-eligible s/p references preserve their optional ERI cache and original
  matrix-direct fallback under budget/allocation pressure. Angular domains
  outside quartet coverage retain the original evaluator.
- The selected quartet arena, topology, matrices, actual eigensolver workspaces
  and detached reference state participate in the complete reference bound.
  Exact budget succeeds; one byte less rejects without a result.
- Stream/resource ownership remains with the existing RHF plan. No extra
  streams or cross-owner borrowing is introduced.
- Completed-reference diagnostics include the selected route, shell-pair count
  and candidate shell quartets per physical Fock build. These counts are not
  a claim that screened/skipped candidates performed a primitive contraction.

## Evidence

An early diagnostic DSO on n2 completed 230-AO ethane RHF in 128.7267 seconds
(20 iterations, 21 physical Fock builds), without resident ERIs. Its energy
differed from the independent PySCF 2.14.0 oracle by 3.4e-13 Eh. Density and
orbital-energy maximum differences were 2.67e-10 and 5.88e-10; its native
commutator residual was 9.33e-13.

The same diagnostic then completed geometry/basis-only native RHF, source and
CCSD in 431.1802 seconds (external process 431.59 seconds): 165.5820 seconds
RHF, 0.9673 source and 264.6309 CCSD. Twenty CC iterations needed 38 primary
evaluations and one expanded replay. Source work was 25,815,200 values and
30,346,393,704 transform/block summands; CC work was 44,105,252,811,046
summands and 19,032 auxiliary visits. The complete numeric bound was
2,922,595,504 bytes. Correlation-energy error was 1.12e-12 Eh; expanded
singles/doubles residuals were 4.06e-12 / 1.58e-12. This run did not include
triples or forces. The earlier timeouts are not matched complete-endpoint
baseline timings, so no speedup ratio is inferred.

The diagnostic DSO SHA256 was
`bf81aa9f04e22bd466e60af17e9d3880cf556d6f6db441efc46ec60f090cd6d3`,
linked against frozen #1785 library
`67c502d412db6c73e5cf4068c6efd6233a4817ffa4764185c535d6e79bacf734`.
The final complete rebuilt CUDA library SHA256 is
`9df4ef8bf305a05ef1b2d7ab34c586c90db1e4f23d517cbbecff42f4ae43f3ce`.
It passes 108 host policy/SCF-structure tests and six focused GPU cases: both
representations retain their s/p cache/fallback/resource gates, admit d/f
quartet references exactly at their reported budget and reject one byte less.
Four d/f HeH+ force cases compare complete native CCSD(T) energy and forces to
independent PySCF FCI energies and their nuclear central differences. Two
electrons make CCSD exact in this gate. Every case covers a standalone call,
prepared cold/warm calls and a changed geometry. Energy and force gates are
3e-10 Eh and 3e-7 Eh/bohr; net force must stay below 2e-9 Eh/bohr.

All six initial GPU cases passed memcheck with zero errors. After extending
the four force cases to warm/changed geometry, all four passed again under
memcheck with zero errors. Four existing public RCCSD(T) CUDA cases (live
PySCF water, exact/near-degenerate methane, and the 14-AO water-cluster
directional derivative) also passed. Hooks and focused type checks passed.
All real-device checks used finite Slurm allocations and preserved visibility.
The source, copied inputs, hashes, diagnostics and raw results remain ignored
in `.artifacts/reference-quartets/` in the isolated worktree. Final native
molecular CCSD(T) runs are separate from the force qualification above.

The final library then completed cold geometry/basis-only ethane 230-AO
RHF/source/CCSD/(T) in n2 job 2176: 418.4062 seconds internally and 418.81
seconds including process startup and state serialization. RHF/source/CCSD/(T)
times were 147.2301 / 0.9684 / 265.2403 / 4.9673 seconds. Total energy was
-79.71851664319378 Eh, within 1.1e-12 Eh of the independent same-Hamiltonian
oracle; triples energy was -0.014391214097531201 Eh (9.1e-14 Eh error).
The solve retained 20 CC iterations, 38 primary evaluations, one expanded
replay and the source/CC work counts above. The 165 occupied triples tiles
used 152 panel GEMMs and 1,980 moment GEMMs, totaling 3,258,407,583,236
contraction summands. Charging the retained endpoint bound alongside triples
reserved 4,068,012,720 numeric bytes. No oracle arrays entered that run.

Probe source SHA256:
`2a2a0170fb7968b557eafe4f45b5256411a04eee90b2e43e2d75ec9650483623`;
geometry/basis input SHA256:
`9428f2b1d1db38ffa374387705099e8d57fde98e0e068faed2861b04604a1c6e`.
This complete energy endpoint excludes nuclear forces. No large DF force
qualification or public capability expansion follows from it.

## Revisit when

Complete RHF/source/CCSD(T)/force measurements show a remaining source or
contraction bottleneck. Preserve explicit reference identity, work counts,
independent force/energy gates and bounded owner lifetimes when changing
precision, screening or buffering. This route does not by itself complete
the DF nuclear response chain.

References: #1763, #1764, #1765; the preceding
[optional reference-cache decision](2026-10-03-cuda-rhf-reference-residency.md).
