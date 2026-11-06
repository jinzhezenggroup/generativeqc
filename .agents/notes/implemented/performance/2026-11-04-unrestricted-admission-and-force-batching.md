# Decision: explicit unrestricted work policy and bounded UMP2 force batching

Status: implemented
Date: 2026-11-04

The three-index provider/response decision in
[the superseding note](2026-11-04-three-index-unrestricted-df-response.md)
replaces this note's incremental fitted-force storage/scheduling limitation.
The work-policy and Br SCF diagnosis remain applicable. Measurements below are
historical qualification of the intermediate patch, not the final factor path.

## Problem

Issues #2978–#2980 distinguish three different failures: hard-coded work-policy
refusal mislabeled as memory exhaustion, DF-HF provider/convergence/export
failures hidden by one exception, and repeated whole-AO transformations for
singleton UMP2 force integrals. Their historical L20 observations are not
measurements of this revision or numerical qualification of changed schedules.

## Decision

Keep the existing default finite work limit, with one explicitly configurable
process policy for unrestricted primal, DF source/metric and CC response. Its
dedicated exception survives method adapters and maps to public resource-limit
status 10 rather than out-of-memory. Memory, overflow and scientific gates are
independent. Report phase, spin dimensions, iteration ceiling and estimates on
refusal rather than inferring runtime from work units.

Preserve DF-HF backend status and detail before checking convergence and physical
reference publication. A successful backend status alone is not a usable
reference. No external reference replaces a failed native calculation.

The Br reproduction isolated an ordinary-eigensolver input refusal after Pulay
mixing. Averaging only validated raw-Fock roundoff restores exact symmetry before
history storage, but does not alone cure the cold-core charge oscillation. The
CUDA DF bucket retry therefore uses the existing normalized, affine-checked DIIS
policy and at most 32 half-density startup steps, retiring startup history before
undamped iteration. Convergence always checks the undamped proposal and the
existing final physical-state gates; no SCF threshold is relaxed. CPU reference
solves and successful compact-device solves are unchanged. UHF also preserves
raw-Fock symmetry and uses normalized history; the new damped startup is RHF-only.

UMP2 initial adjoints transform a complete virtual pair for each occupied pair.
Same-spin exchange is an index permutation of that same ordered block. Orbital
gradient assembly may transform a complete spin channel once instead of four
independent whole-AO scans for every orbital. Additional provider and returned
block storage are admitted with the complete endpoint. If unavailable, the
original four-slice schedule remains available; its result must agree within
numerical gates. Optional residency is not a mandatory capacity requirement.

## Rejected alternatives and limits

Simply increasing the original primal constant would hide both response policy
and real execution costs. Reclassifying all DF failures as nonconvergence would
continue to obscure provider/physical-state failures. Neither is acceptable.

This bounded scheduling change is not a three-index DF force implementation.
The fitted interaction/adjoint still has dense AO fourth-order storage, and UCC
triples retain their own execution granularity. It does not establish a scalable
production domain or a speed ratio against RCC. A three-index unrestricted
provider and direct factor adjoints remain a distinct architectural follow-up.

## Evidence and invariants

`generativeqc_correlation_admission_tests` exercises actual NO/DZ admission,
explicit override, independent numeric memory refusal, malformed policies and
public status mapping. It also injects provider/convergence/export failures.
`generativeqc_ump2_force_contract` compares resident and bounded schedules,
spin exchange, stale reference rejection and resource/source failures.
Independent UMP2 analytic and reconverged finite-difference checks remain in
`tests/python/test_ump2_forces.py`. Real GPU validation must run through Slurm,
preserve scheduler device visibility and retain complete-endpoint receipts.

Diagnostic-only n1/RTX 5090 Slurm probes linked the updated RHF translation unit
against the previously qualified CUDA library while the complete new library
compiled. They reproduced the three Br refusals and the successful BrOP/DZ
contrast. Symmetry projection alone and normalized DIIS alone still failed
convergence. With bounded startup damping, all four reference energies agreed
with the pinned issue's independent PySCF values within 1.5e-11 Eh. This mixed
diagnostic executable is not complete new-library or correlated-force
qualification. `tests/python/test_df_bromine_reference.py` separately requires
complete native CUDA MP2/HF forces, independent energies and two-step directional
force checks for all four inputs.

### Complete-library qualification

The subsequent full-library build supersedes the mixed diagnostic probes as
acceptance evidence. Source base: `d023599fed06e078b28137eb582c5732f7e213b9`,
plus the retained working-tree patch and new files. All changed C/C++, Python,
test and CMake file hashes were compared with the n1 qualification checkout.
The CUDA library SHA-256 is
`9dca11bb83ebddf1541089b4bae7e3dce7731069d66522a3dbe7e7c08badf486`.
Ignored local receipts under `build/issues-2978-2980-evidence/` retain source
provenance, scripts, input/basis snapshots, numerical results, raw traces,
device samples, build logs and compiler-cache statistics. The machine-readable
`qualification-summary.json.xz` is derived from those complete endpoint receipts.

Qualification used n1's RTX 5090 (32607 MiB, driver 580.95.05), Release,
CUDA 12.9.1 / sm_120, GCC 13 and verified sccache 0.16.0. AOT shell/stationary
artifacts were disabled; native OpenBLAS was unavailable, so host linear algebra
used the scalar implementation. Native thread limits were one. Every real-GPU
execution used a finite-time Slurm allocation on `main` with
`--gres=gpu:5090:1`, preserving assigned device visibility.

- All four original Br geometries/basis combinations passed complete native CUDA
  DF-MP2/HF energy and analytic-force tests, with 100 SCF iterations available.
  Tests retained independent energies, force translation and two-step directional
  finite differences; the four-case suite took 130.96 s.
- Four physical H2/H3 DZ/TZ CUDA DF-UMP2 force cases passed independent energy
  and reconverged finite-difference gates, including the zero-beta triplet. The
  four-case suite took 65.59 s.
- Both original open-shell issue inputs completed the entire MP2-assisted CBS
  energy/force workflow (`standard`, `auto`, explicit matching JKFIT snapshots,
  32 GiB numeric budget, SCF ceiling 200). Each ran three times with tracing off
  and separately three times with tracing on. Separate PySCF 2.14.0 CPU processes
  reproduced the CBS energies and two-step force differences. Maximum energy
  errors were 5.01e-12 / 2.99e-11 Eh for H2/H3; maximum checked force errors were
  8.87e-8 / 3.21e-8 Eh/Bohr. Gates were 3e-8 Eh and 5e-7 Eh/Bohr.

Complete CBS timings below are trace-off measurements. Setup includes reading
the already-local input/basis snapshots and calculator construction, not Python
imports or compilation. Warm calls request and compute energy and forces again.

| Original case | Setup + first endpoint (s) | Two warm endpoints (s) | Sampled device peak (MiB) | Maximum planned numeric bytes |
| --- | ---: | --- | ---: | ---: |
| H2 triplet, 0.85 Angstrom | 1.8747 | 0.57594, 0.51444 | 6173 | 11,791,228,031 |
| H3 doublet, original bent geometry | 25.4554 | 24.1258, 24.1762 | 6257 | 11,983,002,948 |

Device peaks are 200 ms samples from separate trace-on runs, not allocator-exact
peaks or the planned simultaneous numeric capacity. Trace-on/off variability
does not establish an instrumentation speed benefit. These RTX 5090 results are
not a controlled speedup comparison with the historical L20 observations.

In one traced TZ endpoint, initial-weight assembly read the whole AO source four
times for H2 (2,458,624 values) and seven times for H3 (21,781,872 values).
Each orbital-gradient evaluation read it once for H2 (614,656 values) and three
times for H3 (9,335,088 values). These are observed semantic source reads, not
FLOPs, GPU kernel counts or permanent exact-call-count CI requirements.

CPU validation additionally passed 42 tests with six device cases skipped, four
compact-DF independent-oracle tests and both focused native admission/force
contracts. The broader existing UMP2 native test still contains an obsolete
CUDA-rejection expectation ("native UMP2 silently admitted CUDA backend"); that
failure is not evidence against the separately passing force contract. No full
native-suite pass is claimed.

## Revisit when

Three-index unrestricted source/response ownership is implemented and qualified,
or full-endpoint evidence justifies a different default finite work policy.
