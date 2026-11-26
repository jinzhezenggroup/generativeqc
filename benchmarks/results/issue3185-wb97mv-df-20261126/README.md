# H100 WB97M-V DF geometry replay (#3185)

The original native release at `53442660ae08d965890632b6bb538c2f45aa253d`
failed at water96 step 8 and n-C32H66 step 3 on both repeats. A fresh owner
completes either geometry. Rebinding from the preceding geometry triggers
`DF metric rank crossing: retained/discarded subspaces are unresolved` with
or without the preceding density. The extra SCF retry cannot change the
prepared LR provider. The independent spectrum keeps the full-range metric
well separated. The failing LR modes are 0.426 and 0.758 resolution bands
from the existing cutoff.

Branch base `a7d4d0e5d886fe89398dfaa12cdfdc48cbbc245c`.
Measured fix source `e8049fa996846c9a64d9b02416fa96527a874dbc`.
The fix rechecks automatic LR DF admission before solving each moved geometry
while the item is fitted. An unresolved fit selects exact LR before the SCF.
The item retains exact LR across subsequent replays. The auxiliary basis,
full-range fit, thresholds and quadratures are unchanged. Explicit fitted LR
keeps its unresolved analytic-response error.

## Acquisition

Use Release/sm90 AOT builds with the same C++ and CUDA options for both arms,
one H100, one numerical thread, and the exact frozen coordinate sequences.
The [regression fixtures](../../../tests/reference_data/wb97mv-df-3185/README.md)
retain the two minimal pairs and the dense Direct references at the failing
steps. The replay driver authenticates the original frozen sequence and loaded
build before using each sample. It times preparation, geometry rebind, SCF,
analytic force and return to host. After a failed endpoint, record the failure
and use only the last successful density if attempting subsequent steps.
A failed endpoint cannot contribute a qualified time.

Run `acquire_qz.sh TASK_ROOT HISTORICAL_ROOT COLD3_INPUTS ACTION` in a finite
qz H100 batch Job, with the two release arms at `TASK_ROOT/master` and
`TASK_ROOT/candidate`. Each arm contains a source archive expanded into
`source` and a verified production `build` plus release `build-receipts`.
Set `GQC_QZ_JOB_NAME` and `GQC_QZ_PROJECT` from the actual allocation.
`prove` checks the failing geometry in fresh, moved and moved-cold owners and
runs the regression tests staged under `TASK_ROOT/regression/tests`.
`measure` alternates the baseline/candidate processes on one card with two
full trajectory repeats and three cold3 water96 processes per arm, each with
one cold and three warm calls. Prewarm processes are kept separate.
`oracle` independently reconverges the failing geometries at levels 4/2 and
5/3 (XC/VV10). The common absolute gates are 1e-5 Eh per atom, 1e-4 Eh/Bohr
maximum Cartesian force error and 3e-5 Eh/Bohr force RMS. Grid stability uses
one quarter of each gate.

The source-matched H100 trajectory and cold3 timing campaign is in progress.
