# Decision: qualify the public through-f WB97M-V force composition

Status: implemented
Date: 2026-10-01

## Problem

PR #1637 separates WB97M-V geometry composition from the generic SPD integral
descriptor inventory, but two admission/dispatch defects remain. The calculator
still rejects named def2-TZVP and local def2-TZVPD force requests. Correcting
capability discovery exposes `nonfinite or invalid stationary CUDA source`.

Synchronous diagnostics identify `stationary_nuclear_all`, not a nonfinite XC
value or invalid f-shell integral. The geometry owner compiles a nuclear-only
dispatcher accepting plain kind zero. Its AO topology nevertheless enables
Cartesian component mode for d/f shells, encoding center/axis metadata into the
nuclear kind. The nuclear-only dispatcher does not decode these bits.

## Decision

Match public CUDA WB97M-V capability discovery to the existing through-f geometry
domain. Keep STO-3G, def2-SVP and def2-TZVP as explicit bundled names; diffuse
bases require canonical local records. ECP, density-fitting, precision, shape
and capacity restrictions are unchanged.

Enable component dispatch only for owners of integral derivatives. Geometry-only
owners retain their complete packed AO topology but use plain nuclear kinds.
Generic SPD owners retain encoded bindings; geometry-only owners still reject
integral tasks.

## Rejected alternatives

- Capability admission alone hides the downstream nuclear-dispatch failure.
- Expanding generic descriptors to f creates an unrelated combinatorial
  inventory: WB97M-V already borrows native integral derivatives.
- Dropping f/diffuse shells, clipping values, weakening gates or using reference
  densities in production does not fix this ABI defect.
- Removing AO/resource caps needs separate capacity/work qualification.

## Evidence

- Slurm 11930 identifies `stationary_nuclear_all` as the failing entry.
- Slurm 11931 passes an independent nuclear-pair regression with real spherical
  def2-TZVP and a nuclear-only compiled geometry owner.
- Host tests exercise the constructor's actual mode expression for geometry-only
  and integral owners and preserve SPD task encoding: 138 passed, 11 GPU skips.
- Public CUDA tests retain independent GPU4PySCF, translation, warm replay and
  reconverged directional finite-difference gates. Local def2-TZVPD shares the
  exact canonical snapshot with its comparator.
- Through-f tests allow the existing bounded native one-electron metadata
  staging fallback; zero H2D is an SPD retained-owner optimization, not a
  scientific force requirement. Resident density and weighted-density bindings
  and the zero detached final-state export assertions remain unchanged.
- The [VV10 comparator correction](../numerics/2026-10-01-matched-vv10-comparator-domain.md)
  matches independent SCF and force domains without weakening numerical gates.

## Invariants and revisit conditions

Preserve native-only derivatives, complete source coverage, live-generation
checks, bounded scratch and failed-owner cleanup. Registry discovery remains
conservative; calculator/batch capabilities are execution-context views.
Generic integral qualification remains SPD. Revisit encoding only if the
nuclear-only ABI changes, not when AO angular momentum changes.

## References

- PR #1637: `6795097b003a30f9f5e9ab8c34e0705dddbf2427`
- Base master: `5891a0e0253b2ad6f74447a808d7b3271f0952b2`
- [Earlier benchmark diagnosis](../performance/2026-10-01-omol25-matched-grid-benchmark.md)

## Appended complete-endpoint qualification

The nuclear pair and named spherical def2-TZVP RKS water cases pass under Slurm
11938. That allocation was canceled during the next case, not counted as a
wholly passing test job. Full local spherical def2-TZVPD RKS water passes under
11940, including all six independently reconverged displaced energies. The
following OH reference SCF fails; this does not erase the preceding accepted
case or qualify OH. Named spherical def2-TZVP UKS NH₂ subsequently passes the
entire same comparison/translation/replay/finite-difference test under 11947
(2121.59 seconds).

The initial OH-doublet GPU4PySCF fixture does not converge at 180 cycles.
Independent 400-cycle, 0.2 level-shift/damping and finer-grid probes also fail.
Do not relax its gradient tolerance or borrow native densities to manufacture
an oracle. NH₂ provides an independently convergent through-f unrestricted
fixture on the original 12×4×8 grid with unchanged stopping/acceptance gates.
OH is explicitly excluded from numerical qualification.

The focused host suite passes 107 tests with 16 opt-in skips. A broader CPU-only
water finite-difference case (`grid_shape1-water-rks`) fails SCF convergence on
both patched and archived unmodified PR Python sources with the same library
and eight threads. This pre-existing solver behavior is outside the CUDA
admission/dispatch correction.

The accepted-test library digest is
`76c5c3cc04637e96eb6c41b5d150ba0a05521c2abe59fe369c51acade2960290`.
The retained
[qualification record](../../../../benchmarks/results/omol25-wb97mv-20261001/force-qualification.json)
binds each case to its raw log hash and enclosing job outcome. Native-only
production, the bounded one-electron staging charge, zero final-state exports,
and existing precision/ECP/DF/resource restrictions remain acceptance invariants.
