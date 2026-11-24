# Decision: Use the linked PBE response for the named PBE-D3(BJ) Hessian

Status: implemented
Date: 2026-11-23

## Decision

Resolve the exact named PBE-D3(BJ) RKS graph to the existing PBE point and
Coulomb-only response binding. The native prepared item retains the correction
parameters. The same stationary driver adds the existing D3 curvature once,
with no extra electronic response solve. Scalar dense, HVP and recycled dense
use this owner for both Cartesian and real-spherical Direct FP64 all-electron
contexts. No AOT leaves or scientific equations are added.

Correction eligibility is checked by the same native predicate in catalog
lookup/preflight and execution. It admits uncut two-body D3(BJ), ATM off, for
PBE without changing LDA, PBE0 or B3LYP rules. D3(0), D4, custom graphs and
other unqualified contexts do not acquire a published derivative by sharing
a PBE carrier. Python cannot enter its electronic compatibility path for
corrected PBE, even for an explicitly unavailable solver schedule.

## Evidence

On base `cae2959e1`, an independent PySCF PBE SCF and moving-grid gradient
plus the `dftd3` 1.2.1 gradient with its independent `pbe` damping parameters
match the accepted central energy/force. Raw dense Hessian maximum errors at
a `3e-4` Bohr displacement are `1.416e-7` (Cartesian STO-3G) and `2.524e-7`
(spherical def2-SVP), in Hartree/Bohr². The separately resolved correction
action differs by less than `1.835e-14` Hartree/Bohr² in both fixtures. It
does not change the electronic response iterations or induced Fock calls.

Installed SDK, CLI and Python compare both products and the recycled result.
CLI packed energy inventory includes the native correction reservation;
composed host admission succeeds at the exact limit, and one byte less or an
insufficient D3 cap fails before any SCF trace. D3(0) and D4 are refused rather
than publishing bare PBE curvature. Raw artifacts and compiler-cache receipts
are under `/data/jzzeng/gqc-2227-sdk-JsefEq0x/cpu-pbe-bj-hessian`.
All 152 CPU CTests, 38 selected installed Python tests and the strict Sphinx
build passed. No CUDA admission or device evidence is implied.

Five interleaved process timings per variant included SCF, response, curvature
and publication on one CPU core. Eight unchanged PBE, PBE0-D3(BJ) and
B3LYP-D3(BJ) controls matched energies, derivatives and work exactly; median
times changed by −0.07% to +0.29%. The newly admitted PBE-D3(BJ) water
endpoints took `0.292 / 1.751 s` (HVP / dense, Cartesian STO-3G) and
`1.600 / 4.019 s` (spherical def2-SVP). Compared with bare PBE on the same
binary, median time changed by −0.12% to +0.11%. Electronic work matched
exactly: STO-3G used 11 SCF iterations and 13 Fock builds, def2-SVP 18 and 22;
induced Fock actions were 20/168 and 24/198 for HVP/dense, respectively. The
correction added 9/81 pair visits (three phases per pair and direction), not
additional response or integral work.
