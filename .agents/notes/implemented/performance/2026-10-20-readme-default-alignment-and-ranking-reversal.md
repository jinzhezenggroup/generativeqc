# Decision: qualify default README endpoints by common observables

Status: implemented
Date: 2026-10-20

Supersedes the earlier
[stock endpoint campaign](2026-10-20-readme-default-endpoints.md), whose original
receipts and rationale remain unchanged.

## Problem

Replacing historical grid and solver overrides with engine defaults reversed
several apparent 96-atom speed rankings. Treating this as a native performance
regression, or copying residual thresholds numerically between engines, confuses
different scientific and iterative work. Density RMS and orbital-gradient norm
are not interchangeable stopping criteria.

## Decision

The new campaign keeps GPU4PySCF's default XC/VV10 quadratures, convergence,
screening, initial guess and incremental Fock behavior. Native Standard/Fast use
their public defaults and the matching Treutler/Lebedev/NWChem quadratures. Both
include complete moving-grid force response. DF auxiliaries come from the stock
GPU4PySCF selection and are frozen before native timing. There is one reference
profile, no GPU4PySCF Fast, and no displaced-geometry timing.

Common acceptance gates, declared before this campaign, are energy error per
atom <= 1e-5 Eh, maximum force-component error <= 1e-4 Eh/Bohr and force RMS
error <= 3e-5 Eh/Bohr. Fresh tighter direct references at XC levels 4/5 and VV10
levels 2/3 establish a common target including the DF approximation error.
Reference-grid differences and CPU/GPU small-system spot differences must meet
one quarter of these gates. Every timed warm repeat is checked. Missing or
unstable references leave results unverified; failed populations retain their
actual measurements but do not become qualified plot points.

This qualification is limited to the sampled static geometries. It does not
establish displaced-geometry force accuracy, a globally optimal configuration,
or universal Fast speedup. Native WB97M-V DF uses exact long-range exchange while
stock GPU4PySCF fits it; shared direct-target error gates do not make those two
approximate operators identical.

## Controlled configuration evidence

Twenty-nine diagnostic configurations use the same production source
`e5fa4d2f0cd13775327261a0b6ced1eab6006c62` and binary, on one n1 RTX 5090 per
Slurm job. These are diagnostic single cold/warm pairs, not the new README
population medians. Native absolute latency improved under defaults, while the
reference improved much more:

| Method | Native historical cold/warm (s) | Reference historical (s) | Native default (s) | Reference default (s) |
| --- | ---: | ---: | ---: | ---: |
| PBE | 42.06 / 8.84 | 79.63 / 11.88 | 40.72 / 6.72 | 15.38 / 3.61 |
| PBE0 | 61.16 / 10.45 | 79.58 / 13.84 | 49.33 / 8.37 | 19.15 / 4.88 |
| r2SCAN-3c | 118.56 / 24.79 | 146.25 / 33.83 | 83.35 / 18.94 | 29.06 / 10.91 |

The historical grid contains 2,359,296 points. The default pruned quadrature has
about 1,078,000 points. Historical reference settings also forced energy 1e-12,
orbital gradient 1e-10, screening 1e-14, full-current-density Fock rebuilding,
zero density pruning and grid response. The diagnostic reference-default column
uses its default `grid_response=False`; the new README explicitly enables it.

PBE0 leave-one-setting-out measurements distinguish important interactions:

| Configuration | Cold / warm (s) | SCF cycles |
| --- | ---: | ---: |
| All historical overrides | 79.58 / 13.84 | 37 / 3 |
| Restore default grid only | 66.49 / 10.44 | 37 / 3 |
| Restore default convergence only | 29.24 / 10.27 | 9 / 1 |
| Restore default screening only | 76.32 / 13.17 | 37 / 3 |
| Restore default grid response only | 84.39 / 6.40 | 42 / 1 |
| Restore incremental Fock only | Failed | 100-cycle limit |

These are not additive timing contributions: changing a setting can change the
SCF trajectory. A later pilot at energy 1e-10 and orbital gradient 1e-8 with
default incremental Fock failed PBE/PBE0 warm and r2SCAN-3c cold convergence at
50 cycles. Artificially tightening the timed reference or disabling its
optimization would not resolve the intended comparison. Full-density rebuilding
belongs only to the separately labeled strict accuracy oracle.

## Additional configuration audit

The following distinctions were checked against the executed launchers,
measurement receipts and installed GPU4PySCF 1.8.1 source. They must accompany
the diagnosis rather than be silently treated as convergence effects:

- Native Standard uses energy 1e-10, density change 1e-8 and, for the CUDA KS
  physical closure, RMS/maximum residual gates bounded by 1e-9. GPU4PySCF uses
  energy 1e-9 and an unset `conv_tol_grad`, resolved by its solver to
  `sqrt(conv_tol)` (approximately 3.16e-5). These measure different quantities;
  its extra convergence check also has its own relaxed thresholds. Native
  Direct budget-Fock Fast selects energy 1e-6, density 1e-4 and screening 1e-10;
  these controls do not apply to every HF/DF fallback.
- Native Huckel versus reference MINAO initialization remains an inherent
  engine difference, not a newly changed setting. Default native incremental
  Direct J/K is disabled; GPU4PySCF retains its default incremental behavior.
- Matching VV10 quadrature does **not** match its active density domain.
  Native `MOLECULAR_VV10_DENSITY_THRESHOLD` is 1e-8, while installed GPU4PySCF
  `NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD` is 1e-10 in both SCF and forces. The old
  matched WB97M-V comparator explicitly imposed 1e-8 on both engines. The
  current driver leaves the reference default untouched. This difference
  remains unresolved and must not be described as complete VV10 alignment.
  Actual current WB97M-V reference receipts report level 3 for both XC and
  nonlocal grids; do not substitute a remembered default from another version.
- Ordinary DFT `small_rho_cutoff` is 0 in both historical and current reference
  receipts. Removing its explicit assignment causes no effective change here;
  it is separate from the VV10 active-density threshold above.
- The historical refresh launcher and WB97M-V reference supplement set
  OMP/OpenBLAS/MKL threads to 8. The current aligned launcher sets all three
  to 1 for every engine. This is a real historical-protocol difference whose
  performance contribution has not been isolated. The same-binary ablations
  keep their environment fixed, so this does not explain their within-run
  configuration effects.
- The historical build records GCC 11.4, whereas the current frozen build
  uses GCC 13.3; both use CUDA 12.9.1, Release/sm120 and FastCompile OFF. The
  current build explicitly reports scalar CPU linear algebra. No claim about
  a historical BLAS regression follows without corresponding old linkage
  evidence and a controlled experiment.
- Historical sampling used five warm repeats plus moved populations and
  separate small-system priming. The replacement uses three frozen-density
  warm repeats and omits moved populations. Both count complete E+F endpoints
  with setup in cold. Historical explicit WB97M-V DF/96 resource overrides
  are not part of the new default-resource population.

These observations are an audit of existing measurements, not permission to
alter their recorded controls or source identities. Common observable gates
still require completed qualification; they do not imply identical operators,
active domains, stopping algorithms or work.

### Follow-up thread and oracle checks

Paired n1 jobs 11052/11053 use the same frozen binary and default controls at
one and eight CPU threads, one cold and three frozen-density warm repeats.
Native PBE cold changes 41.39 to 41.08 s and warm median 6.82 to 6.54 s;
PBE0 cold changes 49.37 to 49.31 s and warm 8.39 to 8.14 s. Reference PBE
changes 16.31 to 15.96 s cold and 5.61 to 5.60 s warm. Thread count does not
reverse those rankings. These are diagnostic populations, not replacements for
the recorded one-thread campaign. Raw paired receipts and launchers remain in
`.artifacts/settings-audit/` and n1 `/data/jzzeng/readme-settings-audit-20261020/`.

The CPU WB97M-V spot oracle initially failed because PySCF's range-only `get_k`
needs its direct-SCF prescreener even during full-density rebuilding. Preserve
the wrapper that omits previous-density/potential inputs, but keep CPU
`direct_scf=True` to construct that optimizer. The separate rerun passes the
CPU/GPU spot gate (maximum force difference about 2.01e-9 Eh/Bohr). With the
unchanged dense-grid stability and observable gates, the 96-atom Direct
WB97M-V Standard, Fast and reference populations all qualify. Standard warm
45.44 s versus reference 120.96 s supports a roughly 2.66x speedup specifically
for this static warm population. Cold and other-method limitations remain.

## Fast work and remaining performance limits

At frozen source `02337e0cf3d94915a352d5056f6d09d27fe353f2`, 96-atom Direct
Fast PBE/PBE0/WB97M-V warm receipts perform two mixed and two strict Fock builds;
r2SCAN-3c performs two mixed and three strict builds. Standard warm performs
one. The public Fast profile restarts its mixed/refinement sequence, so its
looser threshold alone does not imply less work for an already converged seed.
HF Fast additionally runs a complete independent tighter audit. The CUDA warm
density admission and separate audit-seed fix in PR #2647 removes the previous
host eigensolver/cold-audit overhead without deleting those scientific checks.

Default auxiliary g support removes an admission gate; it does not qualify all
sizes for every resource budget. The 96-atom WB97M-V DF force still reports a
prepared response resource limit. Other 96-atom DF force endpoints can dominate
latency despite a one-iteration warm SCF. These must not be described as
over-convergence or hidden behind substituted auxiliaries.

## Invariants and revisit conditions

- Freeze measured production source/binary and retain exact executed drivers.
- Preserve original failed and superseded receipts rather than relabeling them.
- Bind qualification to complete receipts, geometry, versions and reference
  identities; reject malformed force shapes before NumPy subtraction.
- Keep README prose short and place controls, diagnosis and failure details in
  the evidence report. Restore the earlier 2x4 figure arrangement; leave CBS alone.
- Consider new stopping/precision policies only after independent E/F gates and
  complete endpoint work demonstrate the benefit. Do not tune defaults to win
  selected water sizes or suppress genuine slower Fast results.

## References

- README measurement/qualification drivers under `benchmarks/readme_defaults*`.
- Original default campaign: `benchmarks/results/readme-defaults-20261020/`.
- PR #2647: default auxiliary admission and bounded warm/auditor density reuse.
- `docs/maintainer/performance_engineering.md` for production qualification scope.

## Follow-up: isolate default auxiliary and VV10 active-domain effects

Same-binary n1 PBE-DF/48 diagnostic profiles retain source `02337e0cf`, the
same orbital basis, quadrature, strict native controls and CPU thread settings.
Changing only the auxiliary input from historical cc-pVDZ-JKFIT to the reference
selection def2-SVP-JKFIT changes cold/warm from 13.320/2.571 s to 31.530/19.083 s.
The old fit contains 1856 AOs and auxiliary angular momentum through f; the
reference default contains 1808 AOs and oxygen g. Executed shell response changes
from enabled to disabled. Three-center response increases from 0.134 to 16.602 s;
SCF in the default warm endpoint takes only 0.237 s. This isolates an algorithmic
fallback triggered by the default angular domain, not too many SCF iterations or
simply a larger auxiliary dimension. Neither diagnostic fit replaces the default
README auxiliary. Native g support is numerically complete but not yet fast.

The separate WB97M-V Direct/48 reference diagnostic preserves default grids,
convergence, incremental Fock and complete force response. Changing both SCF and
force VV10 density cutoffs from GPU4PySCF's 1e-10 to native's current 1e-8 changes
cold 96.362 to 94.722 s and warm median 32.342 to 31.812 s, with identical 9/1
cold/warm SCF cycles. The roughly 1.6% warm effect does not explain the measured
several-fold native warm advantage. The active-domain mismatch still exists;
this experiment does not authorize describing the defaults as identical.

Supplementary references refine unstable XC pairs to 5/6 (M06-2X) and 7/8
(r2SCAN-3c) without changing any observable gate. WB97M-V adds VV10 4/5 at XC5/6;
the larger reference is still bounded by its Slurm allocation. Offline
qualification accepts complete consecutive denser pairs and compares CPU/GPU
spots only at identical XC/VV10 levels. Earlier unstable, failed and partial
receipts remain retained. The main 192-population campaign and its slower Fast
baseline remain immutable; new Fast implementation results use distinct source
and binary identities.

## Follow-up: verified fixes and remaining comparison differences

The original campaign above remains immutable. Subsequent implementations have
their own source/binary identities and cannot replace selected old points under
the old revision label.

At `33461b49e`, the DF consumer partitions auxiliary angular momentum: existing
s/p/d/f shell sharing runs alongside a generic g remainder. Slurm 11404 passes
34 complete endpoint/translation tests. All repeats of PBE/PBE0 DF48/96 and
r2SCAN-3c DF48 pass unchanged observable, grid-stability and CPU/GPU-spot gates.
PBE96 warm decreases from 129.270 to 29.708 s, and PBE0 from 255.253 to 56.309 s.
Those candidate times still exceed the corresponding default reference, so this
fix does not establish general DF leadership.

At `bb25db89e`, native's molecular VV10 density cutoff changes to rho >= 1e-10
under a new policy identity. 133 host and 39 GPU force-seed tests pass, as do
every cold/three-warm sample of Standard 3/12/48 and Fast 12 against stable
XC 5/6 + VV10 4/5 direct references. Standard WB97M-V48 warm is 7.035 s,
consistent with the earlier warm advantage. This branch does not contain the
separate Fast warm fix. GPU4PySCF additionally removes abs(weight) <= 1e-14 while
native keeps nonzero signed weights and their derivatives. Aligning the density
cutoff must not be called identical active-domain screening.

A bounded r2SCAN-3c DF96 diagnostic at `33461b49e` isolates a second work cliff
before the force calculation. The first RI-J evaluation takes 135.361 s; metric
factorization takes 1.797 s and complete preparation 6.558 s. The plan assigns
about 14.54 GB to values and 7.14 GB to response, uses 1,183,744 ordered AO pairs
and an auxiliary tile of 185, and repeatedly regenerates raw three-center tiles.
The diagnostic intentionally stops after 180 s and is not a performance
population. This case cannot be explained by precision auditing or g force
response alone. Revisit value representation, source reuse and bounded resource
planning.

For publication, keep these distinctions explicit in evidence while leaving
README prose short:

| Condition | GenerativeQC | GPU4PySCF 1.8.1 |
| --- | --- | --- |
| Scientific inputs | Same geometry, spin, orbital basis and method corrections | Same |
| XC/VV10 quadrature | Treutler/Lebedev/NWChem level 3 | Installed defaults: level 3/3 |
| Moving-grid force response | Complete | Explicitly enabled for the shared requested force definition |
| VV10 density cutoff | Baseline 1e-8; correction 1e-10 | Default 1e-10 |
| VV10 weight screening | Retain nonzero signed weights | Remove abs(weight) <= 1e-14 |
| Standard stopping | Energy 1e-10, density RMS 1e-8; physical residual closure | Energy 1e-9, orbital gradient approximately 3.16e-5 |
| Integral screening | Standard 1e-12; admitted Direct Fast 1e-10 | Default 1e-13 |
| Initial density | Huckel | MINAO |
| Direct incremental J/K | Disabled by default | Default incremental implementation retained |
| DF auxiliary | Frozen exact tables selected by reference | Default def2-SVP-JKFIT or 3c even-tempered |
| WB97M-V DF range exchange | Exact long-range correction | Fitted long-range exchange |
| Precision profile | Native Standard/Fast; fallback work retained | One default profile |
| Cold/warm boundaries | Constructor/preparation + E/F; frozen own cold density | Same declared boundaries |
| Resources | One n1 RTX 5090, one CPU math thread, finite Slurm job | Same |

The two stopping residuals are different mathematical quantities; their numeric
thresholds cannot be equated directly. Complete observable gates remain common.
Old reference full-Fock/tight-convergence overrides, thread counts, compiler
versions and sampling changes are described above with controlled evidence.

## Completed baseline accounting

Slurm 11405 completed the independent WB97M-V/96 XC5/6, VV10 4/5 reference.
All repeats of the original three Direct96 populations pass the unchanged
observable gates, with quarter-gate reference stability and CPU/GPU spot
agreement. Standard warm 45.438 s versus default reference 120.961 s is a
qualified 2.662x baseline observation. Its native 1e-8 VV10 density domain
remains explicitly different from the reference's 1e-10; it is not a timing
claim for the subsequent domain-alignment fix.

The complete frozen campaign has 188 completed populations, two WB97M-V DF96
resource failures, and two r2SCAN-3c DF96 four-hour timeouts. Standard 3c DF96
completed cold at 6546.999 s and only two of three warm calls, at
3686.229/3686.284 s (each one SCF iteration). Fast produced no sample before
the limit. Original partial receipts retain their status and are paired with
scheduler termination logs rather than relabeled as complete. Updated baseline
qualification totals are 170 passed, 18 failed and four incomplete/unverified;
HF DF fitting and M06-2X water-3 grid failures remain failures.

[Retained large-case evidence](../../../../benchmarks/results/readme-alignment-audit-20261021/README.md)
contains the independent reference, original WB96 measurements, all-repeat
gates and lossless timeout receipts. These observations do not justify a
universal default-performance headline or lowering the common accuracy gates.
