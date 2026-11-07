# Complete Se/Br fitting spaces and conditioned cold UKS proposals

Status: implemented
Date: 2026-11-05

## Problem

Issues #3000 and #3001 retain six all-electron spherical def2-SVP B3LYP-D3(BJ)
UKS energy/force inputs. Their unmodified def2-universal-JKFIT Se/Br records
include h shells. Rejecting these shells, or deleting them to fit the previous
auxiliary-g limit, does not implement the requested density-fitted Hamiltonian.
The larger Direct inputs also expose cold occupation cycles and a redundant
final closure that can destabilize an already canonical physical state.

Fixed-density Direct J/K comparisons agreed with independent CPU contractions
to approximately 3e-11 Eh. Diagnostic snapshots also retained metric-orthogonal
orbitals, density reconstruction and small eigensolve residuals. These controls
did not identify a generic J/K or eigensolver arithmetic defect. They instead
motivated conditioning proposals without changing the physical F[D] or its
acceptance contract; they do not establish a unique cause for all SCF failures.

## Decision

### Role-bounded auxiliary h

Admit auxiliary h values and first center derivatives only for full-range CUDA
SCF density fitting. Extend the role-specific generated Gaussian-moment domain,
native Cartesian/real-spherical normalization and six-term AO metadata together.
The separate h value and response generators need F11 and F12, respectively;
the established lower-angular-momentum generators retain their arithmetic.

The public shell-expansion caller must carry the same backend/operator/order/
role/consumer request used for capability admission. Otherwise an admitted SCF
auxiliary basis is incorrectly revalidated as an ordinary CPU orbital basis.
Only the SCF auxiliary normalization bridge opts into l=5. Correlation rejects
auxiliary h before reference or cache work, even though it shares lower-level
SCF source infrastructure.

H response retains the bounded polynomial-panel implementation. It does not
enter g-only promoted shell schedules, mixed shell response, packed occupied
pair response, or optional Hermite accelerators. Long-range h, orbital-h
production execution, auxiliary i and higher response orders remain rejected.
No contraction, primitive or fitting shell is removed.

### Native cold proposal conditioning

For strict-FP64, all-electron, full-range native-cold CUDA B3LYP UKS containing
an element beyond Ar, reserve sixteen startup proposals when the original
iteration budget has room for at least four subsequent sweeps. The first and
last proposals use current F[D]; intervening proposals average current F[D]
equally with the preceding proposal. The first idle DIIS history slot owns
this temporary proposal, with zero admitted Pulay history until startup ends.
A GPU kernel performs the update without host copies or extra matrix storage.

Subsequent Pulay history uses X^T(FDS-SDF)X, where X^T S X=I. This avoids fitting
rotations according to the scale of compact core AO coordinates. Both existing
temporary matrices are reused. The raw AO commutator remains untouched for
physical diagnostics and publication, and DIIS history always pairs a current
physical Fock with its corresponding transformed residual.

Imported, transported and authenticated warm seeds retain their previous path.
Short budgets, approximate Focks, ECP, range-separated, nonlocal and
lower-precision paths do not acquire this startup. Every sweep still counts
against max200 in the issue protocol; no CPU solution or oracle seeds production.

Extend the existing fitted-state canonical proof to eligible full-range Direct
UKS global hybrids. When the actual physical D/F already passes the original
energy, raw residual, electron, idempotency and canonical-density gates, publish
that proved state instead of forcing another unextrapolated closure. Rejection
continues the original solver and budget without a valid state token. This
extends, rather than replaces, the rationale in
[the fitted-KS decision](2026-11-05-df-ks-cold-gauge-and-canonical-proof.md).

## Rejected alternatives

- Raising generic angular guards alone leaves missing h values, derivatives,
  normalization and metadata; trimming the fitting basis changes the method.
- Eight startup proposals repaired some endpoints but not all larger cases.
  Sixteen proposals with raw-AO DIIS still failed case-05 in both Direct and DF
  at the original 200-sweep ceiling. Damping alone is not the retained repair.
- Diagnostic level shifting and ordinary DIIS from the original native cold
  density did not consistently repair the cycles. CPU damping/EDIIS trials are
  investigative controls, not production code or evidence for a CPU fallback.
- Increasing the ceiling, relaxing physical or force gates, or returning an
  energy-only state does not satisfy either issue.

## Independent reference details

The AO-to-libcint transformation is inferred only inside the independent test
oracle. Unit-radius sampling misses tight Se/Br core AOs and produced a false
rank deficiency (36 of 37 for SeH). Sample each atomic center at six geometric
scales down to the tightest primitive radius, normalize sample columns during
least squares, undo that scaling and verify full rank and reconstructed values.
This changes neither native initialization nor production AO evaluation.

Some inputs have distinct stationary roots. Reconverge only the independent
PySCF calculation from the native density on identical quadrature, and also
evaluate independent F[D] at that unmodified native density. Report the selected
root rather than comparing unrelated cold energies or asserting a ground state.

## Evidence and invariants

`tests/reference_data/ks-se-br-3000/` pins the original Gist revision, ordered
geometries, populations, shell records and controls. Its historical protocol
still describes the formerly unsupported auxiliary limit; that text must not
be mistaken for a qualification that drops h. The original reproducer's fixed
`software_commit` string is not the identity of a newly tested native library.
Qualification must retain actual source identity and library SHA-256 separately.

Compiled host/CUDA primitive tests compare Cartesian and spherical h blocks and
center response to libcint; an additional compiled probe exercises actual native
`basis.cpp` normalization. Source preflight keeps higher-shell, correlation and
long-range refusals. Native CUDA controls check startup tail elements, dirty
proposal initialization and preservation of physical Fock input.

Complete endpoint qualification is `test_se_br_ks_cuda.py` with
`GENERATIVEQC_SE_BR_KS_CUDA_TEST=1` inside finite Slurm allocations. All six
inputs run Direct and full-space DF, cold and warm, with native E/F only. Gates
are independent total energy 3e-7 Eh, force 1e-6 Eh/bohr, independent F[D] 5e-9
Eh, raw AO stationarity 1e-9, and native central differences at 2e-4/1e-4 bohr
within 3e-6 Eh/bohr. Cold/warm energies and forces must agree within 1e-9 Eh
and 1e-7 Eh/bohr. Endpoint times, iterations and Fock builds accompany errors;
these are correctness observations, not comparative performance claims.

The final source-matched run passed all twelve case/route combinations on n2
RTX PRO 6000 Blackwell devices (Slurm 3963--3966, preserving assigned device
visibility). Cold iterations were Direct `[48, 71, 31, 47, 54, 59]` and DF
`[24, 57, 31, 47, 54, 63]`; every same-geometry warm replay took one sweep.
Maximum independent energy error was 6.63e-11 Eh, maximum force error 4.91e-9
Eh/bohr, and maximum two-step finite-difference error 5.76e-8 Eh/bohr. These
observations include 24 native E/F executions and 48 reconverged displaced
energies, with the original max200 and unchanged tolerances.

The source/library identities, per-case timings, Fock builds, selected-root
energies and errors are retained in
[`qualification.json.xz`](../../../../benchmarks/results/se-br-ks-3000-3001/qualification.json.xz).
Compiled CUDA primitive qualification (Slurm 3958) passed 187 tests; the twelve
skips deliberately request h-only response from the g-only evaluator. Host
primitive/codegen, compiler/policy and basis/source suites passed 203, 101 and
153 tests, respectively. Compiler ownership checked 584 modules with no errors.
Native startup/recovery controls and matched range-exchange/CPU controls passed.

The local build disables automatic XC CUDA AOT inventory. The unrelated host
KS final-state test expects M06-2X/MN15 registries unconditionally and reports
"split-hybrid final-state admission disagrees with its physical J/K model" in
both the pre-repair and repaired libraries. Do not weaken that contract or
interpret that configuration failure as Se/Br endpoint evidence.

A wider #2984 rerun on this node passed five tests but the separate O2 singlet
reached max200. Slurm 3968 ran the pre-change library and current candidate on
the same allocated GPU; both returned exactly -150.12683548967266 Eh with
energy change 8.674502111202606e-8 Eh and the same rejection. This is a retained
baseline limitation, not a qualified endpoint or a reason to broaden the
heavy-element UKS repair to restricted oxygen. The compact receipt records both
library identities and this failure alongside the successful Se/Br observations.

## Revisit when

Qualify any broader startup domain, long-range h, promoted h schedules or
correlation consumer independently with complete E/F, response, resource and
identity evidence. A second-order solver or a different electronic-root policy
requires its own scientific decision; these fixes do not promise globally
convergent SCF or the lowest-energy root.
