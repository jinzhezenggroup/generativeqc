# Decision: qualify a DF density guess without changing the CCSD(T) reference

Status: implemented
Date: 2026-10-09

The initial experiment below is preserved as measured. Its benchmark-only
default decision is superseded by the separately qualified
[native guarded default](2026-10-09-df-rhf-preconvergence-default.md).

## Problem

Conventional unscreened FP64 RHF consumes about 129 seconds in the 230-AO
ethane DF-CCSD(T) endpoint. Correlation-only DF does not require its *initial
guess* to use four-center integrals. Changing the accepted reference to DF-HF,
however, would change the method and require consistent response/gradient
validation rather than inheriting the conventional-reference force contract.

## Decision

Add a benchmark-only native CUDA experiment: DF-JK RHF produces a detached AO
density; a fresh Direct RHF solver starts from that density with fresh DIIS.
The final solver still uses zero screening, FP64, energy tolerance `1e-12`
and density RMS tolerance `1e-11`, and must export its validated physical
canonical reference. Only then may the existing correlation-only DF energy
and complete analytic-force owner run. Public method defaults are unchanged.

Use `aug-cc-pvtz-jkfit` (484 auxiliary functions) for the guess, independently
of the original `aug-cc-pvtz-ri` correlation basis (488 functions). The native
orbital basis has 230 spherical AOs and nine occupied orbitals. Neither a
PySCF density nor any oracle integral/amplitude enters the measured calculation.
PySCF is an explicit offline basis-metadata provider and independent oracle.

The seed's capacity is charged in the simultaneous numeric reservation. DF
preparation/refusal and the exact seeded solve belong to endpoint timing.
Preliminary nonconvergence or resource refusal retains cold Direct fallback;
the correlated owner's existing seeded-reference fallback remains intact.
An experiment's previous output reference must be released before reuse.
Optional-cache retries clear diagnostic reference ownership before restarting.

## Rejected alternatives

- Treating one exact Fock validation as an exact SCF solve: the candidate must
  complete the ordinary Direct convergence loop and canonical export.
- Quietly replacing the accepted reference by DF-HF: this would change the
  scientific Hamiltonian and require a separately defined response contract.
- Reusing the correlation RI basis as a JK-fit basis without validation.
- Importing a PySCF density into the production endpoint or counting only
  the final Direct solve while omitting preliminary DF preparation.
- Promoting a default from this one molecular case and two confirmatory pairs.

## Invariants

The initial guess has no derivative contribution to the fully converged
stationary exact-reference endpoint. This is not permission to stop early:
the original exact SCF, canonicality, Lambda and Z-vector gates remain.
Compare densities and orbital energies, not individual coefficients within
degenerate orbital subspaces. A common exact Hamiltonian alone does not prove
that arbitrary seeds reach the same RHF branch or the global minimum.

Complete Direct Fock counts include post-SCF builds. DF cycle counts are not
a complete DF physical-operator census; the latter remains unavailable/null.
Do not reconstruct uncounted attempts from successful-attempt diagnostics.
Nonfinite diagnostics of a refused preliminary iteration serialize as null,
not invalid JSON `inf`. Final physical arrays/scalars must remain finite.

## Evidence

The frozen base is master `ab5282f74c0acf98c18ec05b343a5494de85cf49`.
The input SHA-256 is
`9428f2b1d1db38ffa374387705099e8d57fde98e0e068faed2861b04604a1c6e`;
its numeric budget is 64 GiB, not an observed VRAM peak. The separate JK input
digest is `f3aac69713fb69c025580064892e192d0991e5638a67889f39948fc3b35d5de4`.

Screen Slurm 2758 compares two cold Direct HF runs with one candidate each at
`1e-4`, `1e-6`, `1e-8` and `1e-10`. All four pass the fresh independent
PySCF 2.14.0 RHF audit. Direct uses 19 SCF cycles and **20 physical Focks**;
all seeded candidates use 12 Direct cycles and **13 physical Focks**.
The three looser DF settings use 13 DF cycles, versus 16 at `1e-10`, without
saving another Direct cycle. The screening times are not pooled with the
confirmatory E+F measurements or used to claim a uniquely optimal threshold.

Complete E+F Slurm 2759 fixes `1e-4` after the independent screening gate and
runs cold Direct/seeded/seeded/Direct in one allocation. Slurm visibility is
`2`, identifying PRO 6000 UUID
`GPU-cacd0aaf-c80f-41eb-d7a2-4c3a5970f282`. No visibility override is used.
Both methods share the frozen library
`81ca1958f9230ae6ab8090c2762edbf43d53fd77d51409a865dd97648a43c05f`
and executable
`29aa24ec93947f2a41cd737c55309f0e96cddec4f5c3cf419b016dc7125621c7`.

The first complete pair is 515.891731 / 471.296221 seconds, with RHF
128.272012 / 85.974324 seconds including 3.016883 seconds of DF work.
Its maximum energy difference is `4.405e-13 Eh`, density difference
`6.617e-11`, orbital-energy difference `3.838e-12 Eh`, and all-component
force difference `3.487e-10 Eh/bohr` against the Direct control.
All numerical gates pass, including retained independent same-Hamiltonian
CCSD(T) energy and two-step directional force differences (`1e-4`, `3e-5`
Bohr). Those differences are not an independent analytic oracle for every
Cartesian component. The final paired summary is retained with the raw runs.

### Completed paired measurement

All four cold E+F endpoints pass all-repeat-pair and independent gates. The
two observations per method are descriptive evidence, not the shared
five-pair default-promotion qualification:

| Quantity | Direct control | DF density -> Direct |
| --- | ---: | ---: |
| DF cycles | 0 | 13 |
| DF preparation/SCF median, s | 0 | 3.008333 |
| Direct SCF cycles | 19 | 12 |
| Direct physical Fock builds | 20 | 13 |
| RHF median including DF, s | 128.148333 | 85.936760 |
| Complete cold E+F median, s | 515.554616 | 471.211025 |

RHF saves 42.211573 seconds (**32.9396%**); complete E+F saves 44.343591
seconds (**8.6011%**). Complete endpoint ranges are 515.217501--515.891731
seconds for Direct and 471.125829--471.296221 for the candidate. These are
full native energy-and-host-returned-force endpoints, not only the last
Direct solve or one response kernel. The experiment does not assign the
small remaining phase/time differences to any kernel optimization.

Maximum candidate/control total-energy error is `4.405365e-13 Eh` and
all-component force error is `3.490586e-10 Eh/bohr` (gate `3e-9`). The fresh
independent RHF energy/density/orbital-energy maxima are `6.821e-13 Eh`,
`3.316e-10`, and `6.086e-10 Eh`. Independent correlation energy error is
`2.473e-12 Eh`; the two directional-force errors are `7.560e-9` and
`3.280e-9 Eh/bohr` (gate `3e-7`). Largest physical response residual is
`8.421e-13`; translation defect is `1.254e-12 Eh/bohr`. No seed fallback
occurs in any of the paired measurements.

The final rebuilt native library separately passes 28 focused tests under
Slurm 2763, including fresh-PySCF methane HF/E+F, a deliberately exhausted
one-cycle DF guess, CLI rejection and memory/optional-cache ownership
contracts. Eight additional host capability/ownership tests pass. Slurm
2764 independently runs a complete final-build 230-AO candidate and passes
all physical, same-Hamiltonian energy, full-component control and two-step
directional-force gates. Its diagnostic flags certify zero Direct execution
retries and no discarded primal/resident-JK attempts. This validation uses
a different binary/allocation and is deliberately not pooled into the timing
table. Numerical agreement is tolerance-based, not a claim of bitwise identity.

Raw source/binary/input identities, logs, arrays and summaries live on
`n2:/data/jzzeng/qc-df-preconverge-experiment-20261009/`, mirrored locally under
ignored `.artifacts/df-hf-preconvergence/`. `source/` and `build/probe` preserve
the frozen timing prototype; `final-source/` and `final-build/probe` validate
the final diagnostic/fallback fixes. Do not pool binaries' timing samples.
The prototype native sources are separately mirrored so formatting and the
later refused-iteration JSON/lifetime fixes cannot erase measurement provenance.

## Consequences

This removes genuine Direct SCF work, rather than speeding up an isolated
kernel or hiding preliminary setup. Twelve Direct cycles still remain; the
experiment does not claim that a single exact build suffices. Tightening the
DF seed beyond the fitting error does not necessarily reduce target work.
The final method remains the conventional-reference correlation-only DF method.

## Revisit when

Promote an explicit production policy only after broader molecular/branch,
changed-geometry, warm/batch and constrained-memory qualification, and the
shared five-pair endpoint promotion gate. Consider a separately selectable
full DF-HF method only with independently validated DF-consistent SCF,
orbital response and three-center/metric nuclear derivatives.

## References

- `benchmarks/df_hf_preconvergence.cpp` and its preparation/audit Python driver.
- `docs/developer/df_hf_preconvergence.md` for current reproduction and boundaries.
- `benchmarks/results/df-lambda-cost-2136-20261009/oracle-energy.json` for the
  retained same-Hamiltonian independent energy.
- `benchmarks/results/df-lambda-gemm-20261004/oracle-energy-fd.json` for the
  retained independent directional finite differences.
