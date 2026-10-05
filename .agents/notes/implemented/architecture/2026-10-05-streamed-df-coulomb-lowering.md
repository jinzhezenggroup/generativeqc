# Decision: canonical streamed DF Coulomb metric operations

Status: implemented
Date: 2026-10-05

## Problem

After the resident Coulomb migration, the actual streamed DF-KS owner still
submitted four metric GEMVs directly. One updates existing charge, and another
reads a column panel with padding; treating either as a contiguous overwrite
contraction would change the mathematical operation or physical view.

## Decision

Express the four operations with existing TensorIR einsum/add nodes. The generic
update adapter recognizes only `seed + einsum(A,B)` with unit coefficients,
matching shape/dtype/arithmetic, and an input seed whose exclusive last use is
the root. The seed and result have an explicit donated alias group. Scientific
identity remains the original immutable SSA graph, including the add node;
provider selection cannot hide beta in a binary-contraction identity.

Project all operations through the same canonical request/candidate selector
and shared `CudaVectorContraction`. Keep the existing FP64 GEMV incumbent, with
the executable strided-GEMM alternative unpromoted. Padding is a physical view
property. Full and tail panel bindings are prepared once; at most six bindings
exist regardless of panel count. Replay borrows the existing handle, stream and
numeric scratch, and binding objects die before their borrowed resources.

Plan diagnostics charge the actual binding objects. The generic candidate
diagnostics retain request/precision/layout identities and provider versions.
Explicit extra numeric workspace is zero; opaque borrowed cuBLAS allocations
remain unknown. No speedup is claimed.

## Invariants and rejected alternatives

- Preserve metric rank/cutoff, eigenvalue scaling, raw source passes and kernels.
- Preserve the staged host-tensor compatibility path and all bounded scratch.
- Seed donation must reject exported/shared seeds and product-input aliases.
- Full/tail preparation must remain outside panel loops and graph capture.
- Replacing accumulation with overwrite, flattening padded metric panels, or
  changing the incumbent algorithm without endpoint evidence is not admissible.

`df_coulomb.cpp` now has no direct vendor LA calls, so its vendor-boundary debt
entry is removed. Raw generated tensor kernels and other DF modules remain
separate migration work; this does not claim completion of #1886/#1890.

## Evidence

n1, Slurm main/5090, CUDA 12.9: 35 compiler/selection tests pass (one skipped),
focused type checking passes, all 42 native vector cases pass, and complete
native DF and KS suites pass. The metric cases exercise both candidate recipes,
four operations, sizes 1/7/13/32, padded NaNs, repeated seed accumulation and
changed-input graph replay against long-double scalar references. Memcheck and
initcheck report zero errors.

The production DF gate independently reconstructs four-center RI values for
full and truncated metric rank, auxiliary width 3 and pair widths 3/17, with
two nonsymmetric densities and J-only execution. Its tolerance is `3e-11`.

Complete streamed DFT qualification uses four separated water molecules,
STO-3G orbital / def2-SVP auxiliary Cartesian bases (28/100 functions), an
8.5 MiB DF budget and width-8 auxiliary panels. Initial 64 KiB–2 MiB budgets
were rejected; the rejected admission probes remain as evidence. The independent
PySCF energy gate is `1e-8` Hartree with residual `<1e-9`.

Reproduction scripts, source/binary hashes, verified ccache commands/stats and
logs are retained under `.artifacts/1890-streamed-coulomb/` in
`/data/jzzeng/qc-dft-streamed-coulomb-lowering-1890` on the editing host and n1.
The qualified library SHA-256 is
`27a9e2bfbf0a84c2e8219dfc17672fd221ebb5b41f3928e55898dd0e72020dc7`.
One binding occupies 1,864 host bytes in this build: full/tail streaming retains
11,184 bytes (7,456 when no auxiliary tail exists).

Both public streamed PBE/PBE0 tests pass, including cold, warm and changed
geometry. Separate complete endpoint measurements retain configuration,
preparation, two warm replays and geometry rebuilds; maximum independent PySCF
energy error is `2.11e-12` Hartree. Every metric diagnostic remains streamed,
rank 100, auxiliary width 8. The 294,912-point host oracle explicitly raises
its export bound to 400,000; its initial default-limit rejection is retained.

| Endpoint | Prepare (s) | Cold (s) | Warm (s) | Geometry (s) | Cold/warm/geometry Fock builds |
| --- | ---: | ---: | ---: | ---: | ---: |
| PBE-RKS | 0.5262 | 2.9416 | 0.2950 | 2.9090 | 10 / 1 / 9 |
| PBE0-RKS | 0.2765 | 4.9353 | 0.5194 | 4.3178 | 11 / 1 / 9 |

Configuration takes 0.0023–0.0027 s. These are current complete energy timings,
not a before/after performance claim. The physical metric path remains two
raw source passes and two full-rank metric vector contractions per J build;
the independent truncated-rank native gate exercises accumulation/padded panels.

## References and revisit conditions

Extends [resident Coulomb ownership](2026-10-05-resident-df-coulomb-lowering.md)
for #1886/#1890. Revisit provider selection only with complete endpoint evidence;
broaden the generic update recognizer only with explicit new SSA/effect rules.
