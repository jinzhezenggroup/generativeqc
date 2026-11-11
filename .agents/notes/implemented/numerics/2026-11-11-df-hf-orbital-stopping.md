# Decision: align compact DF RHF stopping with stock GPU4PySCF

Status: implemented; independent device/endpoint qualification complete
Date: 2026-11-11

## Problem

#3065 aligned Direct RHF only. Compact DF RHF still required density RMS and
an AO-maximum commutator, unlike the stock E/MO-gradient conjunction. The
aligned-RKS README cohort retained a DF HF96 cold loss (18.182064924389124 s
versus 17.741809528321028 s), with 11 target iterations / 12 physical Focks
versus 10 iterations / 11 effective potentials. This is not a causal timing
ablation: initial guesses and execution owners also differ.

## Decision

Reuse `GENERATIVEQC_CUDA_RHF_CONVERGENCE` for qualified all-electron FP64 compact
DF RHF with physical DIIS residual storage. After DIIS retains its unchanged
AO residual history, transform only the disposable stopping copy to
`X^T(FDS-SDF)X` using already reserved scratch. Its Frobenius norm divided by
sqrt(2) is the restricted physical MO-gradient norm. Default candidate admission
requires exact `abs(delta E) < energy_tolerance` and gradient norm below
`sqrt(energy_tolerance)`. Density change remains a finite diagnostic, not an
additional gate. Remove the same unintended density/energy-roundoff additions
from the shared Direct/DF orbital predicate so the contract has one meaning.

Explicit strict, tighter public E/D, ECP, mixed/application-budget, physical
reference and no-DIIS owners retain their existing strict/bounded behavior.
Explicit orbital requests on unsupported owners fail closed. Prepared Python
resource/checkpoint identities already bind the shared request; compact graph
reuse additionally binds the resolved policy. Final physical Fock, projector,
trace/idempotency, canonical proof and response-token checks are unchanged.
Neither the force accuracy gates nor GPU4PySCF defaults are modified.

## Validation contract

Execute actual Direct/DF reduction in ordinary and captured CUDA calls,
including large density change, localized/distributed gradients, nonfinite
entries and inactive neighbors. Independently compare complete energy/force
endpoints, cold/frozen warm and moved geometry for several restricted models,
including strict controls. Qualify large DF HF endpoints against a tightly
converged same-fitted-model oracle; retain the separate dense-fitting failures.
Measure strict/default endpoints in balanced same-card order, retain all work
and do not replace a complete README cohort with cherry-picked timing points.

## Rejected alternative

Do not treat the previous DF HF loss as irreducible merely because the owner
was outside #3065, and do not retune GPU4PySCF or relax independent accuracy
gates to manufacture a win.

## Source-matched qualification and same-image ablation

Measured implementation `0b74eb57f33bb64d3e859276f267b041a45e2e54` has native
input identity `7a7ff32cb750ea418cb55c39064dd786c2d102c10c76d4f907c4bc82fee96a23`
and production ELF SHA256
`bd4d8929df9f8b3f77a8477784303d8ea2c37e46f67c6bc8a24414df474b47c6`.
Build and all real-device work use finite node1/main Slurm allocations with
assigned visibility, verified sccache 0.16.0, Release/SM120, fast compile off and
device link on. The build retains 1,261 cache hits and 93 misses, including the
misses needed for changed CUDA policy consumers; no cache was cleared.

The image passes 276 host/resource/checkpoint/inventory tests (17 explicitly
optional skips), actual Direct/DF CUDA scalar/graph acceptance tests and 47
DF final-state/eigen/projection/force-transition cases. Independent complete
E/F qualification passes 28 cases initially; two newly added N/C DF fixtures
initially lack auxiliary tables because the README input freezer covers only
H/O. Exporting the independent oracle's exact default auxiliary basis for
those test geometries fixes the fixture, and both additional cases pass on the
same image. No scientific gate, production input or measurement is replaced.
The initial fixture failures and a separate native-test SONAME loader failure
are retained as prerequisite failures, not hidden numerical passes. The latter
is fixed by retaining the ELF's ordinary `.so.0` loader link in the frozen image.

A separate three-round balanced strict/default/unchanged-reference ablation
has 36 processes / 144 complete E+F calls across all four DF HF sizes. Every
native strict/default pair has the same target iteration/Fock counts:

| Atoms | Strict cold median (s) | Default cold median (s) | GPU4PySCF cold median (s) | Native iterations / Focks |
| --- | ---: | ---: | ---: | ---: |
| 3 | 0.676368 | 0.664875 | 3.014433 | 9 / 10 |
| 12 | 0.940864 | 0.933920 | 3.157010 | 10 / 11 |
| 48 | 2.875211 | 2.949060 | 4.718348 | 10 / 11 |
| 96 | 17.815167 | 17.786771 | 18.484750 | 11 / 12 |

HF96 warm process-median medians are 3.727561 s strict, 3.745421 s default and
6.861900 s GPU4PySCF. The retained rows contain all samples, not only medians.
These measurements do **not** establish a stopping-induced speedup: work is
unchanged, initial guesses and DIIS trajectories still differ between engines,
and small timing differences have not been shown statistically significant.
In particular the old 0.44 s cold loss cannot be attributed to #3065 or fixed
by threshold alignment from these data alone. Matching convergence semantics
does not require matching iteration counts.

Untimed tightly converged GPU4PySCF same-fitted-model oracles (`conv_tol=1e-12`,
`conv_tol_grad=1e-9`) independently qualify every ablation endpoint at 48/96
atoms under quarter-budget E/F gates. Default HF96 worst energy error is
`4.4858931384320994e-12` Eh/atom, force max `7.666242640702592e-9` Eh/Bohr and
force RMS `2.6324066985588076e-9` Eh/Bohr. The reference auxiliary tables are
hashed and matched to the actual measured GPU4PySCF defaults. This same-model
qualification does not waive the separate dense-reference fitting failures.

All raw ablation, independent reference, build and validation receipts are in
ignored `.artifacts/df-hf-aligned-20261111/` and the matching
`n1:/data/jzzeng/df-hf-aligned-20261111/` tree. The complete README cohort is a
separate predeclared acquisition; ablation observations never substitute for
its timing points.

## Final publication source freeze

Concurrent pre-commit.ci commit `aa259d2dc113118e1c6b30c6f8a1ce101872fe10`
formats two existing KS expressions. Preserve it rather than overwrite another
writer. The pinned clang-format hook also formats the new HF code. Although
these are non-functional edits, they change the native source identity, so the
initial `0b74eb57f` cohort is retained as an unselected complete campaign and
never relabelled as current-image data. The decision to reacquire is made for
source identity, before reducing that cohort's timing rankings.

The new publication source is
`5a1ae34addac124423a22490d1d7dda0f90fc0f4`, native inputs
`cf7b1e62f4502195b1983fc6124ca26d4aa7cdd7810c4b030d97e3827bff7986`, ELF
SHA256 `783388d764cda2cd687b0d192b161f16f66176013b479a07fdc46fa0895a2adc`.
The final image independently passes all 30 complete E/F cases in one run,
276 host/resource/checkpoint/inventory cases (17 optional skips), the actual
CUDA Direct/DF ordinary/graph reductions and all 47 DF physical final-state,
projection, eigen and force-transition cases. The repaired N/C auxiliary
fixtures are part of this frozen source. New receipts are under ignored
`.artifacts/df-hf-aligned-final-20261111/` and the matching node1 tree.

## Final-source complete README acquisition

The final-source campaign completes all 128 processes / 512 complete E+force
endpoints, with no failed processes or retries. Direct and DF each observe
32/32 faster cold pairs and 32/32 faster warm pairs: 64/64 in either phase.
Independent accuracy gates still qualify 58/64 pairs for each engine, and all
58 qualified pairs are faster in both phases. All four DF HF dense-reference
fitting failures and Direct/DF M06-2X3 grid failures remain plotted and excluded
from qualified wins. All 1024 cross-engine full-vector comparisons pass, but
do not replace independent qualification. Warm Direct WB97M-V96 speedup is
9.368630765349003. These observations are not statistical rankings.

Final-image DF HF96 cold takes 17.94930298998952 s at 11 target iterations /
12 physical Focks, versus unchanged GPU4PySCF's 18.577793136239052 s at 10
iterations / 11 effective-potential evaluations. Its three warm calls take
3.743092503398657, 3.725467275828123 and 3.746906951069832 s, versus
6.840868979692459, 6.801037695258856 and 6.859910197556019 s. The separate
same-image ablation above found identical native strict/default work counts;
neither the new aggregate wins nor reversal of the old single cold loss is
causal evidence that the stopping-policy change reduced HF work.

The final-image HF48/96 endpoints also pass comparison against the retained
untimed tight same-fitted-model oracles after exact geometry, fitted-input,
default auxiliary-table hash and force-contract checks. Across their eight
cold/warm samples, E/atom error is at most 4.4858931384320994e-12 Eh, Fmax
5.2886765189441576e-8 and Frms 2.5530113735054997e-8 Eh/Bohr. This checks
ordinary stopping accuracy, not the separate dense-reference fitting gates.

Normal and optimized Python independently replay the complete vector audit
and scalar publication capsule; both scalar replays reproduce the tracked
SVGs byte-for-byte. Visual inspection, 512 offline tests, Ruff, default-policy
inventory, CUDA ownership and fixed-cap evidence checks pass. Native source
identity remains `cf7b1e62f4502195b1983fc6124ca26d4aa7cdd7810c4b030d97e3827bff7986`;
CBS bytes are unchanged. Libxc's historical evidence remains stale with zero
current-source qualified products; passing catalog tests do not promote it.

## Public default follow-up

The separate
`../compatibility/2026-11-11-public-default-standard.md` decision aligns the
public precision facade and default policy object with the measured Standard
endpoint and removes the benchmark's explicit profile selection. It retains
this source-bound campaign rather than relabeling it, with a new complete
source/image acquisition for the replacement publication. Neither this public
dispatch correction nor a later timing cohort changes the same-image HF
strict/default ablation or establishes a causal HF work reduction.
