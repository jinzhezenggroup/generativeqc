# Comparison: borrow restricted XC specialization, not dense force assembly

Status: algorithm comparison; default-off PBE0 integration with qualified snapshot endpoint gain
Date: 2026-10-10

## Scope and source

Compare the algorithms in the installed GPU4PySCF 1.8.1 reference to the retained
GenerativeQC #2185 execution snapshot plus the current restricted-point delta.
This is not a comparison to an unspecified current GPU4PySCF release or to
current master. Source excerpts are retained under ignored
`.artifacts/xc-restricted-stage/gpu4pyscf-1.8.1/`. The moving-grid comparator uses
full grid response; do not substitute its cheaper frozen-grid force algorithm.

## Restricted density and XC evaluation

GPU4PySCF `dft/numint.py:140` constructs GGA total density and three gradients
from one `dm.dot(ao[0])` for a Hermitian density. `eval_xc_eff:1784` selects
unpolarized Libxc and supplies total rho and one sigma. Restricted-spin
information is therefore used before evaluating XC, not recovered after doing
the full polarized algebra.

Our grid producer already projects the owned restricted density once and copies
the ordered panel for its two-spin consumers (`src/dft/cuda_grid.cu:671`). Another
"remove the second GEMM" change would duplicate an existing optimization. The
remaining avoidable work is the force point's eight-direction full-spin
differential. The bound PBE0 lowering reuses the shared scalar algebra with four
directions, one exchange evaluation and zero-polarization correlation, returning
the same two-spin consumer ABI. This is a structural work reduction, not a new
functional, smaller grid, density cutoff or mixed-precision approximation.

On the restricted manifold, n=2*rho_s and grad(n)=2*grad(rho_s). Differentiating
the shared equal-spin variable varies both spin channels; symmetry permits
recovering each physical first partial as half the restricted derivative. The
existing two-spin adjoint consumer therefore keeps its factor conventions.
This does not recover the spin-antisymmetric Hessian: equality at the evaluation
point alone does not constrain response perturbations. Retain eight directions
for general/response consumers rather than extending the first-derivative
specialization to HVP by analogy.

Producer ownership and generation, not an RKS label or equal floating values,
prove the restriction. Separate template instantiations let the compiler erase
unused polarized work; a guarded general-plus-restricted body previously grew
code without improving registers. Preserve the bounded general fallback when
the witness, mathematical composition or scratch/storage preconditions fail.

## Analytic force contraction

GPU4PySCF `grad/rks.py:417` full response constructs atom-sorted grid blocks,
rho/XC weights and AO derivatives. `_gga_grad_sum_:384` builds derivative AO
matrices with dense dot products; full response accumulates a
`(3, nao, nao)` orbital-response matrix, contracts it with density, and adds
grid-density and grid-weight response terms. This is a valid BLAS-oriented
algorithm with an explicit quadratic AO intermediate.

Our stationary geometry consumer pulls the scalar XC adjoint through AO
translation directly into center/atom force seeds and separately executes
phased Becke response. It does not need that dense three-direction AO response
matrix. Copying GPU4PySCF's entire force assembly would abandon this useful
data-movement difference and introduce an O(nao^2) intermediate. Borrow its
early restricted-variable specialization instead, while retaining all of our
moving-grid terms and the current integral/Pulay/nuclear owners.

## SCF work is a separate algorithmic axis

The installed GPU4PySCF `lib/diis.py:219` applies an absolute `1e-14` threshold
to eigenvalues of the augmented DIIS system. Our earlier controlled diagnosis
shows iteration inflation can depend on that conditioning policy. This is not
evidence that one JK kernel performs more work per build. Keep the stock
comparator, its seed and stopping criterion unchanged; report actual full and
incremental Fock work. See `2026-10-09-pbe0-gpu4pyscf-diis-tail.md` for the
separate normalized-DIIS diagnostic and its qualification limits.

The restricted geometry point changes no SCF iteration policy or direct/DF
choice. Complete endpoint comparisons must use matched PBE0/RKS, spherical
def2-SVP, direct/noDF, FP64 and identical grid semantics, retaining host return,
actual selected point counts and SCF/Fock work. Independent gates remain
1e-8 Eh and 1e-7 Eh/Bohr. Static register reductions or synthetic routing parity
cannot establish an endpoint improvement or authorize a performance PR.

## Consequence

First target genuinely redundant full-spin XC work, not another warp-publication
micro-optimization or a wholesale switch to the comparator's dense force path.
General PBE/UKS/HVP boundaries remain outside this PBE0-only experiment; their
existing independent failures are not hidden by successful restricted tests.

## Measured follow-up

The producer-bound specialization now passes official snapshot complete endpoint
gates: five interleaved off/on energy+moving-grid-force calls per geometry on a
Slurm-assigned RTX 5090 reduce medians by 5.281%/4.307% at 48 atoms and
3.418%/3.068% at 96 atoms (warm/moved-warm). All timed calls retain one SCF/Fock
build and identical AO/force semantic work. These are complete host-return
timings, not point microbenchmarks. Each independent arm's seed remains frozen;
the two converged seeds are not bit-identical to each other.

The default-off control has no observed >2% slowdown in a separately scheduled,
different-visible-device guard; do not use that noisy cross-allocation difference
to claim further acceleration. Source base, independent E/F gates, actual bound
point counts, raw evidence, failed harness attempts and promotion prerequisites
are detailed in `2026-10-10-rks-point-producer-binding.md`. Default remains off
and the measured #2185 snapshot is not current master. Submission-source
reconciliation remains necessary before a phase PR.
