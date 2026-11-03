# Required native SCF-default owner: retained qualification

This is bounded ownership-retirement evidence for #2227, not new scientific
capability or whole-parent acceptance. The candidate requires the selected
native default-controls query; unsupported older installations fail explicitly.
The successful current-library policy and native equations/defaults are unchanged.

## Initial source and scope

- Baseline Python/source: `9cfb77727661cc64e47938d9d0ba65a157427439`.
- Candidate implementation: `e4072ae6f`; the retained receipt records its full SHA
  and an empty uncommitted patch at measurement time.
- Both frontends use one CPU native library. Native source identity,
  library SHA-256, scripts, launcher verification and shared cache snapshots are
  retained in `qualification.json.gz`. No native/compiler source changes occur
  between these revisions.
- The library's embedded full-input identity also covers Python policy. Its
  retained `9e62f8cd...` identity differs from the finalized candidate's
  `51f23ad6...` identity. Unchanged C++/compiler sources do not establish an exact
  full-input match: retain this campaign as historical paired evidence, not
  exact-candidate qualification. The integration campaign below supersedes it.
- CPU RHF and PBE0 RKS, restricted Direct STO-3G H2, explicit full E/F publication.
  The PBE0 quadrature is level-0 Treutler: this is not grid-convergence evidence.
- Each timed endpoint includes Calculator construction and complete synchronous
singlepoint E/F result publication. No prepared solve replaces a full endpoint.
Immutable atom and KS option/grid inputs are supplied before the repeated calls;
their setup is outside the measured constructor/singlepoint boundary in both variants.

## Initial frozen gates and observations

Each ABBA and BAAB order has four process blocks, one separately retained first
call, three warmups and 21 timed endpoints per block. All 16 blocks and 400 full
publications are retained; 336 publications are timed. CPU affinity and native
library/thread controls match between baseline and candidate.

| Model | Order | Candidate/baseline median ratio | Gate |
| --- | --- | ---: | ---: |
| RHF | ABBA | 1.005940 | 1.05 |
| RHF | BAAB | 1.002352 | 1.05 |
| PBE0 RKS | ABBA | 0.990314 | 1.05 |
| PBE0 RKS | BAAB | 0.988401 | 1.05 |

All 400 publications replay with zero energy/force difference, equal SCF iteration
counts and equal native initial-guess work/capacity records. Only elapsed
`preparation_seconds` is excluded from work equality; it remains in the raw data.
Opaque CPU Fock-build counts are **unavailable**, not reconstructed from iterations
or reported as zero. These ratios establish no speedup or other-model guarantee.

The companion source validation passes 582 regression tests plus 6 subtests,
with one explicit real-GPU/Slurm-only skip; the public native SDK defaults test;
12 independent PySCF CPU LDA/PBE/PBE0 spin/Direct-DF force cases with Python force
consumers forbidden; and 6 CPU precision/reference cases. Host policy queries for
CUDA metadata are not real-GPU endpoint evidence. All applicable pre-commit hooks
pass. No universal memory/provider/HVP or SDK/CLI/Python qualification is claimed.
An additional constructor/resource regression set passes 367 tests, with 20 skips;
those skips are not counted as executed endpoint or numerical evidence.

## Source-matched integration campaign

`integration.json.gz` retains a separate campaign after integrating master
`604c04259e216bd08123ce2873459eb152218c90`. Baseline frontend/source is that
revision; candidate is the committed merge
`d72546b0db65c4cded1f9a4078d5cd1a36e4a95b`, with an empty measurement patch.
Neither the original raw receipts nor their numerical/timing gates are replaced.

The rebuilt CPU library's actual `generativeqc_get_source_identity()` result
matches the full candidate input inventory before measurement:
`3c68b052381daacac5230a75dfa88edda59635db2d323dba547a3a88c6fb8bbd`.
Native/compiler sources are equal between this baseline and candidate. Both
frontends use that one library, whose SHA-256 and verified compiler-cache
commands/statistics are retained. Shared cache activity is not a per-build hit
rate. The measurement boundary, inputs, counts and frozen 1.05 gate are unchanged.

| Model | Order | Candidate/baseline median ratio | Gate |
| --- | --- | ---: | ---: |
| RHF | ABBA | 0.999819 | 1.05 |
| RHF | BAAB | 1.000922 | 1.05 |
| PBE0 RKS | ABBA | 0.987592 | 1.05 |
| PBE0 RKS | BAAB | 0.984354 | 1.05 |

All 400 full publications, including 336 timed samples, replay with zero
energy/force difference and equal iterations/initial-guess semantic work.
Opaque CPU Fock-build counts remain unavailable. This is bounded CPU
non-regression evidence, not a speedup or additional-domain claim.

This source-matched library also passes 609 policy/transport/ownership/doc tests
plus 6 subtests, with one explicit real-GPU-only skip; 112 catalog/ownership
tests; 52 integrated resource tests; the native SDK defaults test; 12 independent
PySCF CPU LDA/PBE/PBE0 force cases; and 6 independent-reference precision cases.
Some test groups overlap and are not a combined unique-test count. The skipped
GPU case and host-only CUDA metadata queries are not real-GPU evidence.

The native Libxc catalog integration also retains its historical campaign,
exact capability summary, blockers, empty admitted endpoint sets and stale
status. Only its two checked current-input identities change; no stale
measurement becomes current qualification.

## Reproduction and history

The compressed JSON contains the exact one-off `probe.py` and `compare.py`
scripts, raw first/warm/timed publications, native provenance and replay summary.
Restore the scripts under `.artifacts/required-native-defaults/`. Restore baseline
`python`, `manifests`, `upstream` and `CMakeLists.txt` using `git archive` at the
baseline SHA into `.artifacts/required-native-defaults/baseline/`.

Build `generativeqc` and `generativeqc_cli` in `build/native-runtime-policy` with
CUDA disabled, Release, native CPU force AOT enabled and verified `sccache`
configured as the C++ compiler launcher. Run the archived comparator from the
checkout root with single-threaded OpenBLAS/OMP. It applies the unchanged 1.05
gate and checks science/work equality rather than assuming it from timing.

The source checkout also retains `failed-protocol-v1/v2` and
`accepted-precommit-v3` under the ignored artifact directory. Initial setup lacked
baseline compiler assets/source-root metadata, and the first comparator wrongly
included a duration in semantic work equality. Those errors and raw records are
retained; neither a completed timing failure nor a scientific gate was discarded.
The committed candidate's complete campaign uses the repaired protocol v3.
For the integration campaign, restore its scripts and baseline beneath
`.artifacts/required-native-defaults/integration-20261104/` instead. The comparator
requires a committed candidate and refuses a library/input identity mismatch
before timing. Rebuild from that candidate before reproducing its measurements.
