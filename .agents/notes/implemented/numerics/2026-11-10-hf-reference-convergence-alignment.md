# Decision: explicit native MO-gradient convergence for default-reference comparisons

Status: implemented; complete-endpoint RTX5090 qualification at unchanged common gates
Date: 2026-11-10

The opt-in-only/default-retention decision below is superseded by the separately
authorized [owner-bound default promotion](2026-11-10-default-hf-orbital-gradient.md).
This note's source identities and measurements remain historical observations.

## Problem

The [cold HF diagnosis](../performance/2026-11-10-hf-cold-force-contract-cost.md)
found unlike output-accuracy/stationarity contracts, not a new duplicate-Fock
bug. Native Direct HF forces use an AO-maximum physical commutator ceiling of
1e-8. Default GPU4PySCF 1.9.0 instead checks the occupied/virtual physical
orbital-gradient Frobenius norm against sqrt(conv_tol), about 3.16228e-5 for
conv_tol=1e-9. These are different norms; changing a density tolerance cannot
make their numerical thresholds interchangeable. The user explicitly requests
changing only our settings, never the reference engine's defaults.

## Decision and boundaries

Add the explicit native execution request
`GENERATIVEQC_CUDA_RHF_CONVERGENCE=orbital-gradient`. Unset, empty and `strict`
preserve the existing solver/strict-force owner. Unknown values fail closed.
The new request belongs to the Direct CUDA RHF bucket, not compact DF or KS.
It is not a return to the historical relaxed AO ceiling and is not a new
production default. Existing strict force qualification remains required.

For an idempotent restricted determinant D=2 C_occ C_occ^T and X^T S X=I,

    ||2 C_vir^T F[D] C_occ||_F = ||X^T(FDS-SDF)X||_F / sqrt(2).

The implementation borrows the existing matrix-product scratch, after raw AO
DIIS history and the eigensolve/density proposal have consumed it. Only then
does it transform the residual in place. It changes neither the DIIS norm nor
the physical Fock/density pair. Each subsequent iteration rebuilds the AO
residual before DIIS. Screened provisional iterations retain their original
stopping policy and bounded proposal allowance; the alternative applies only
to the physical target, never to an approximate operator. The physical
iteration and rebuilt-force determinant
share one scalar comparison authority and the same transformed norm. Graph
capture, bounded allocation and final force-token ownership are retained.

The explicit alternative retains the existing energy comparison (including
its bounded FP64 roundoff guard), extra density-RMS condition, operator
screening, initial guess, pure FP64 arithmetic, trace/idempotency and force
evaluation. Thus it matches the reference's physical MO-gradient threshold,
not the reference algorithm or number of Focks. Its strict `<` orbital test
does not change the old inclusive AO-maximum ceiling.

Only all-electron restricted FP64 owners are admitted. UHF, ECP, mixed
arithmetic, application-budget/Fast owners and physical-reference exports
reject the explicit request. Either tighter energy or density controls
restore the historical strict physical ceiling; in particular, a stricter
audit must not inherit a relaxed physical reference from an environment
setting. This setting is part of Python CUDA resource/checkpoint identity
and native captured-plan admission, rather than an untracked benchmark hack.

## Qualification protocol

Use the frozen master805 cohort's environment, frozen auxiliary inputs and
import-rejecting native wrapper. Build a separate normal Release SM120 image
through srun and verified sccache; do not modify the earlier homepage image.
Retain source/ELF identities, finite Slurm allocation, cache statistics and
complete numerical/work-count receipts. The build initially pays new absolute
source paths; cache use must not be described as a high hit rate.

On one scheduled RTX5090, measure three balanced-order rounds of strict native,
explicit orbital-gradient native and unchanged-default GPU4PySCF at HF48/96.
Also include both small Direct cases and the four unchanged-default DF cases;
no shape-dependent settings or selected faster replacement points. Every
fresh process owns constructor/preparation/SCF/analytic-force/host-export cold
and three calls from its own frozen cold density. Raw failed trials must remain.

Independently resubtract all force vectors against the unchanged dense README
oracles. HF48/96 additionally use the earlier authenticated tighter references.
Keep the common README gates (E1e-5 Eh/atom, Fmax1e-4 Eh/Bohr, Frms3e-5) distinct
from the additional E1e-8/Fmax1e-7 diagnostic. The alternative must not be called
strict-force-qualified merely because it passes the common gate; reference
defaults must not be tightened to manufacture an accuracy or timing result.

Retained raw/analysis/build receipts:
`.artifacts/hf-convergence-alignment-20261110/` locally and
`n1:/data/jzzeng/hf-convergence-alignment-20261110/` remotely.
The new setting does not authorize mixing a few aligned results into the
existing strict default-vs-default README cohort. A replacement figure requires
a complete, explicitly labeled and qualified comparison population.

## Results and acceptance

Slurm17107, finite30min, n1/main, one RTX5090,4CPUs/24GiB, assigned visibility1.
GPU UUID `GPU-d0b9e1b7-2083-6198-da50-c3e55693691c`.
The same normal Release SM120 image owns every native comparison:

- source identity `dbc15ae252d40b75ac3fc5afd3320e353006ec3eef09ac478793c0d97adc378a`;
- ELF SHA256 `39ba7eb7811aecb73beb80b011a73e42e97bb81278a303c1a6e0df891f41e1a8`.

There are32 fresh processes and128 complete cold/frozen-warm endpoints.
All processes exit0; native processes reject reference-engine imports.
HF48/96 each have three strict-native/aligned-native/default-reference rounds
in balanced Latin-square order. Small Direct HF3/12 and all four unchanged DF
sizes each have one trial per included configuration; those single observations
are not a statistical ranking. No timing or independent force vector is reused.

| Direct HF | Native strict cold median [min,max], s | Native MO cold median [min,max], s | GPU4 default cold median [min,max], s | Native Focks |
| --- | --- | --- | --- | --- |
| 48 | 7.949604 [7.942287,7.983040] | 6.281873 [6.250466,6.286546] | 8.039068 [7.919885,8.113000] | 14 to10 |
| 96 | 16.929665 [16.917003,17.030949] | 11.520701 [11.496915,11.550690] | 11.818994 [11.768609,11.823688] | 15 to11 |

Native cold time falls20.98%/31.95%; this population's comparison speedups are
1.2797x/1.0259x. Each native warm still has one physical Fock; the screened HF96
proposal remains8 and its physical target tail falls7 to3. Independent
constructor/preparation/forces/host-export costs remain inside every endpoint.
The extra two matrix products per physical iteration are paid, not hidden in
an isolated-kernel timing. All native cold starts retain seven preliminary
guess Focks; Direct post-SCF Focks remain0 with one skipped final rebuild.
Reference counters count get_veff calls, not a claim of identical full-Fock work.

Every Direct endpoint passes the unchanged common gates, including all four
sizes in explicit MO mode. Independent worst force errors at HF48/HF96 are:

| Owner | HF48 Fmax, Eh/Bohr | HF96 Fmax, Eh/Bohr |
| --- | ---: | ---: |
| Native strict | 1.651547e-9 | 1.547548e-9 |
| Native explicit MO | 1.818374e-7 | 3.019337e-7 |
| GPU4 default | 1.376760e-7 | 1.534040e-7 |

Both explicit-MO native and default reference fail the additional1e-7 diagnostic;
neither may be labeled strict-force-qualified. Native default retains that gate
and passes. Equal MO-gradient thresholds do not guarantee equal force errors;
the observed native/reference differences must remain visible. HF3/12 also pass
the common gates but have single observations and larger finite-SCF force errors.

DF is not changed by this setting. Both engines retain the pre-existing dense
energy-gate failures at all four DF HF sizes (roughly1.85e-5 to2.11e-5 Eh/atom,
above1e-5); this is not a new force-convergence repair or a qualified DF win.
Every reference settings dictionary exactly matches its original current-version
default cohort, not just conv_tol. In particular, stock density_fit changes
direct_scf toFalse; Direct'sTrue is not a universal reference default.

## Validation and failed harness attempts

The actual device convergence/finalization program passes both ordinary and
32/256-thread captured/replay controls, including nonfinite and inactive peers.
The selected Python regression set passes267 tests with4 CUDA restart opt-ins
skipped. Additional scheduled native-only guards verify tighter energy/density
requests against the independent HF48 force oracle, invalid/unsupported mode
rejection, and prepared resource-policy immutability. Offline checks pass122
benchmark/code-generation tests and189 standalone host/structure tests. Local
CMake discovery needed the already installed Conda prefix for OpenSSL/LibArchive;
the corrected environment passes without changing project code.

The first guard in Slurm17101 accidentally used Calculator's Cartesian basis
default against the spherical reference and aborted before any benchmark trial.
Its log and earlier passing kernel/Python tests remain in
`qualification-failed-17101/`; the corrected guard explicitly selects spherical
and retains its full vector before checking the gate. An initial offline auditor
also incorrectly required direct_scf=True for the DF wrapper; the original
reference receipts and installed primary source establishFalse as the unchanged
DF default. The failed audit log remains. Neither incident changes the reference
engine or a numerical gate.

The first source-path-separated full build uses Slurm17093; final incremental
builds and every source/ELF/cache receipt remain. A comment-only source-identity
change is actually rebuilt and qualified in17107, rather than exempted from
image binding. No incomplete/failed run supplies a headline timing. The offline
analyzer authenticates all32 raw receipts, exact scientific inputs, one card/job,
source/ELF identity, complete populations and unchanged reference defaults, then
resubtracts every Cartesian component. Normal and optimized-Python replays agree.

## Retention and replay

The complete private evidence archive is
`.artifacts/hf-convergence-alignment-20261110/evidence.tar.gz`
(1,182,712 bytes;160 members), SHA256
`aa9ec54b3b9bf05af7135a622fa314a88711c44e67f4e4de42f14b4bf662e878`.
An identical, checksum-verified copy is retained at
`n1:/data/jzzeng/hf-convergence-alignment-20261110/evidence.tar.gz`.
It contains all32 raw receipts/logs, failed harness receipts, exact acquisition
scripts, installed reference primary sources, compiler/cache/allocation receipts,
full force-vector guards, standalone tests, analysis and focused source patches.
It does not add to the bounded tracked benchmark-evidence population.

Extract the archive at the repository root, verify its `SHA256SUMS`, then run
`python .artifacts/hf-convergence-alignment-20261110/analyze.py` and the same
command with `python -O`. Both regenerate analysis SHA256
`90bc13c4fdaf548e8eade1137b76709024862bd84e95fafd19c93c83045abd7c`
and compact-summary SHA256
`0fa3517cdd5d093bd6ff4d26f0426bc5f39cb578d30e574c7a97d0e8e8a11778`.
`provenance.json` authenticates every member and names the tracked independent
oracle/replay dependencies. The two prior unchanged-default aggregate receipts
are included at their original ignored paths, not assumed to exist elsewhere.

Base commit `805de425a7c9cd323bc3353d466493cf4eadd397` plus
`native-source.patch` reconstructs the exact qualified source identity above;
this is verified by unpacking the base and recomputing the full source inventory.
Perform reconstruction outside an enclosing Git worktree: an initial audit
inside a worktree subdirectory skipped the patch paths and failed closed at the
base identity. Its log is retained. The large initial prototype source archive
is not labeled clean master or used as final source provenance. Large ELF
images remain private on n1, with hashes and final-image binding retained here.
Slurm accounting storage is disabled; the allocation/start receipts and complete
process exit receipts remain available rather than an invented accounting record.

## Consequences

This removes unnecessary strict-force tail work only for the explicitly chosen
comparison contract. It does not make every native default faster, make the
screened proposal cheaper, repair DF's energy-oracle mismatch, or establish a
new universal force-accuracy guarantee. Keep the README's existing strict/default
population separate; a convergence-matched figure replacement needs its own
complete, clearly labeled cohort and unchanged independent acceptance gates.
