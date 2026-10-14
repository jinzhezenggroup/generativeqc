# Candidate: split-hybrid and canonical 3c budget endpoints

Status: bounded numerical qualification; full README/performance scope unfinished
Date: 2026-10-14

## Missing ownership, not a tolerance failure

Earlier M06-2X probes failed before either derivative executed: the packaged
stationary inventory contained only LDA, PBE, r2SCAN, PBE0 and B3LYP. Adding a
method-name whitelist could not supply the missing owner. The inventory now
resolves codes from the generated split-hybrid registry and can explicitly
package one spin. M06-2X packages restricted only; existing profiles retain
their historical spin inventory and legacy point transport.

The subsequent integration onto master `c2c670e2f` supersedes this packaging
choice: upstream already supplies `m06_2x_rks` and `m06_2x_uks`. Both upstream
packages and names are retained; the provisional restricted-only inventory
extension is removed. Arithmetic admission remains independently restricted.

Native binding derives the exact M06-2X component graph, domain version 4,
unit semilocal scales and full-range exchange 27/50. It rejects a reused
component pair with another exchange weight. Descriptive labels cannot grant
admission. Production retains the bounded all-electron direct 24--96 AO,
through-d, 2--32 atom shape. Merely linking the profile does not promote UKS
or the historical small-s/p default. Both ordinary FP64 and Fast use the same
qualified native owner; only a separately admitted budget narrows arithmetic.

Qualification ID 6 names M06-2X. Generated point interiors, AO/grid projection,
Becke and selected weighted derivatives can use float. Point conditioning,
Cartesian conversion, unsafe tails, SCF, Boys, one-electron/Pulay and global
reductions remain strict. This is not all-FP32 SCF or a complete SCF work census.

## Independent checks and replay diagnosis

Slurm 8307, library
`c62bceedf63428f852f3ca32193d93add042264df6eb42acc4e2a6a63a44254f`, passed
both 24/96-AO same-state derivative probes against independent PySCF moving-grid
forces. Strict gates remain 1e-7 Eh and 1e-6 Eh/bohr; mixed gates remain 1e-5
maximum and 3e-6 RMS Eh/bohr, including translational invariance and strict replay.

Slurm 8311, library
`4e7ee5d1ed4bdab9335d8f6779d265aac3a7c9737f792513f5bb5ba47b1b42fb`, passed
the two native probes and the prepared/frozen/moved/failed-peer test. All four
public cases reached their successful complete-force assertions, but failed
on the following energy-only replay: Python advertised AUTO energies while
the native split-hybrid SCF owner explicitly rejects AUTO. The constructor
now rejects this unsupported route early so Fast chooses strict SCF with its
independent derivative budget. Do not bypass the native restriction or hide
the replay failure by removing that assertion. Host checks also verify that
other split hybrids retain strict fallback and that force capability is not
lost. Final scheduled replay/timing validation must follow this Python change.

Slurm 8310 retains the pre-promotion ordinary FP64 timings. An earlier attempt
copied only the main shared library outside its package directory, causing a
missing-sibling M06-2X AOT artifact error. The corrected run uses the original
build directory; no runtime loader fallback or artifact identity was weakened.
Keep `qualify-m06-baseline-missing-sibling.log` distinct from that valid baseline.

## Canonical 3c composition

The public composite already splits its electronic graph from immutable
geometry-only corrections. Consequently the new r2SCAN budget transport also
reached canonical 3c, although previous prose described it as unqualified.
An electronic native record must not silently certify arbitrary corrections.
The facade now binds the canonical complete MethodIR and successful typed D4
and gCP results before naming a full 3c qualification. Other corrected models
retain the observable audit, including when their XC warmup alone is qualified.
The native gradient record remains explicitly electronic; no composite ID or
FP32 correction work is fabricated in native provenance.

Slurm 8312 with the same `4e7ee5...` library passed four independent tests:
water (34 AOs), ammonia (39 AOs), water dimer (68 AOs), and prepared frozen/moved
replay with an invalid peer. Basis inputs are the defining def2-mTZVPP snapshot.
PySCF supplies moving-grid r2SCAN; upstream dftd4 4.3.0 and the separate CPU gCP
reference supply corrections. Correction energy/gradient gates are 1e-10 Eh
and 1e-9 Eh/bohr; full endpoint gates are unchanged from M06-2X. Production D4
remains FP64 CUDA and gCP remains the existing native CPU consumer, never the
Python/reference oracle. Final facade-label tests and timing remain to be run.

## Remaining scope

Keep complete construction/singlepoint/cleanup timing separate from prepared
E/F timings. Prepared runs must freeze the post-cold and post-move seeds and
explicitly pass moved coordinates on every replay. A private prepared budget
test does not implement public `PrecisionCalculator.prepare_batch`.
Do not quote pre-sharing strict timings as precision speedups. DF, WB97M-V,
larger shapes, general HF acceleration and the full README benchmark matrix
remain unresolved. See the HF note for its rejected shared-owner replacement.

Raw receipts, XML, binaries/source snapshots and timing scripts are retained in
`n1:/data/jzzeng/readme-budget-fast-20261013/` and copied small receipts in the
ignored local directory with the same suffix.

## Successful replay and the fair-comparator issue

Slurm 8315 passed all seven M06 public/private/replay checks after the early AUTO
rejection fix. Slurm 8317 passed all four canonical-3c tests with the final
facade qualification. Their original timing receipts remain retained, but their
FP64 comparator was missing an exact equal-spin optimization in inline geometry.
The FP32 point owner already used this algebraic simplification; only the strict
precomputed consumer inherited the producer's generation/paired-feature proof.

Inline strict geometry now consumes that same proof, without admitting UKS or
external nonlocal seeds. A first build failed because the kernel declarations
and launches did not all carry the additional internal argument. Updating the
forward declarations and executable host fixtures fixed it; 66 host tests
passed, with two optional oracle tests skipped. Slurm 8323 built library
`deeafec1fb03de25d3f36e690ebb10833ea256c82281443694cf52fb22eaa61a` and passed
170 CUDA tests, explicitly deselecting the two previously reproduced unrelated
resource fixtures. The same-state oracle test also compares strict execution
with the producer-proof optimization disabled, at 1e-10 force tolerance.

The updated complete M06 singlepoint medians were 1.3640/1.2578 s (24 AOs) and
3.8179/3.3489 s (96 AOs), FP64/candidate. Frozen medians were 0.28469/0.17057 s
and 1.30138/0.85108 s. **The 24-AO strict frozen endpoint still regresses from
the pre-promotion 0.25365 s**; the 1.67 ratio against the slower strict owner
must not be advertised as the achieved Fast gain. Sharing the proof reduced
the regression but did not resolve it. Retain the ordinary compatibility owner
as a same-binary diagnostic comparator and profile before further promotion.

Updated 3c singlepoints were 0.65632/0.62718 s (34 AOs) and 2.91319/2.82008 s
(68 AOs); frozen medians were 0.17980/0.13592 s and 0.59525/0.43493 s. These are
bounded small-case measurements with unchanged strict corrections, not the
full README matrix or a claim that all arithmetic is FP32. The receipts retain
every cold, moved and frozen attempt, exact coordinates, independent errors,
work records and preparation cost.

## Same-binary diagnosis, not a new promotion

Slurm 8329 profiled three frozen endpoints for each of native FP64, forced
ordinary compatibility FP64, and native candidate, using the same `deeafec...`
library and independently checking forces. Native strict remained approximately
0.288 s versus 0.250 s compatibility and 0.172 s candidate under intrusive
profiling. Compatibility actually selected a **serial 512-point** force plan;
native selected **cooperative 256-point** geometry. Across three endpoints the
native geometry consumer took 447 ms in 864 launches versus 236 ms in 432
compatibility launches. Both used the equal-spin proof. This refutes treating
the residual regression as merely a missing producer proof; source tiling and
small-shape scheduling must be considered before speculative scalar rewrites.

Slurm 8333 varied the existing public XC tile option (which changes SCF as well
as native force tiling), retaining ordinary FP64 and candidate controls. At
24 AOs, 256-to-512 reduced M06 frozen FP64 from 0.2881 to 0.2651 s and candidate
from 0.1724 to 0.1436 s; at 96 AOs, 1.3251 to 1.2110 s and 0.8472 to 0.7717 s.
PBE0 also benefited: 24 AO FP64 0.1370 to 0.1133 s and candidate 0.1158 to
0.0946 s; 96 AO FP64 0.8260 to 0.7713 s and candidate 0.7088 to 0.6689 s.
These are diagnostics, not authority to change a global tile default or to
attribute all savings solely to force tiling. M06 small strict still trails the
optimized compatibility consumer. No new tile/schedule default is promoted.

The local-RKS remeasurement in Slurm 8323 also retained a losing population:
96-AO PBE complete singlepoint FP64/candidate was 3.4970/3.5519 s (0.985 ratio).
Do not omit it or describe all qualified arithmetic routes as faster. Complete
endpoint profitability remains a separate unresolved gate.

## Rejected: globally restore the phased M06 strict route

An experiment required the admitted application budget in the split-hybrid controller
(or the existing private diagnostic override). Ordinary FP64 M06-2X retained its
incumbent phased owner instead of the measured slower native inline schedule.
This is not a new acceleration claim or a removed scientific capability: both
owners were independently qualified, and the 8329 same-binary experiment already
established the ordinary owner's better small frozen endpoint cost. Narrowed geometry kept
its native controller and bounded numerical replay; private strict probes remain
available to revisit the scheduling issue. No global tile default changes.

A scheduled test checks both prepared routes and their complete force agreement.
Slurm 8345 passed 176 GPU tests with the same two unrelated resource fixtures
explicitly deselected. It then measured the restored route. At 24 AOs, frozen
strict improved to 0.24593 s (candidate 0.16975 s), but complete strict
singlepoints regressed to 2.69001 s (candidate 1.27731 s), versus approximately
1.36 s with the native controller. At 96 AOs, both strict singlepoint and
frozen timing regressed, to 5.26574 and 1.36853 s. The reported 2.106x and 1.564x
singlepoint ratios therefore use a worse cold comparator and are **not**
accepted Fast speedups. Preserve the full `m06-incumbent-restored-timing.json`.

The blanket routing change was rejected and reverted locally. A test retains
explicit diagnostic comparison against the compatibility owner without
promoting it as the default. Native strict warm scheduling still needs a
force-only measured fix; simply disabling its public capability sacrifices
too much cold/moved endpoint performance. This experiment's integrated binary
`c991c0c6bf4f6871759bf6cacf7a98fdc8070353e6f0921476ae50973d26d240`
remains distinct from the subsequent rebuild restoring native routing.

Slurm 8358 validated the restored native routing with final library
`5b7fc0ef88c97ee56246c91c55428b3cb8df972ca5d68fe576a1ce5792bd9599`.
Six focused direct M06 checks passed, followed by 16 DF/composite checks.
Complete native-default FP64/Fast singlepoint medians returned to
1.37122/1.26636 s at 24 AOs and 3.81809/3.36493 s at 96 AOs. Frozen medians were
0.28599/0.17111 s and 1.30163/0.86964 s. The small native strict warm regression
remains: do not advertise its 1.67x ratio without acknowledging the faster
approximately 0.246 s compatibility warm comparator. Raw complete attempts,
movement and independent gates are in `m06-final-native-routing-timing.json`.
