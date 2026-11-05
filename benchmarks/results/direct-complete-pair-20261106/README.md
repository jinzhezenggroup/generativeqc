# Complete-pair Direct force projection — qualified default

Issue #2976 / PR #3014. This cohort extends exact external-pair contraction
to seven low-order pure-J classes, on top of the qualified compact cooperative
MD baseline. It does not change Direct into DF, scientific settings, or GPU
model dispatch. See the [decision](../../../.agents/notes/implemented/performance/2026-11-06-direct-complete-pair-projection.md)
and [current contract](../../../docs/developer/weighted_eri.md#complete-pair-direct-force-projection).

## Complete endpoint results

Water96, 1088 spherical def2-mTZVPP AOs, r2SCAN-3c exact Direct E+analytic
forces including native D4/gCP; constructor/prepare/SCF/host export included
in cold, same-owner frozen engine-local cold density in warm. Baseline then
candidate, one cold plus three warm each, reused compiler/artifact caches,
no timing traces/census. No moved-geometry population in this cohort.

| Actual allocation | Baseline warm median | Candidate warm median | Change |
| --- | --- | --- | --- |
| A100, ECNU LSF225765 | 7.962698835879564 s | 6.952339671552181 s | -12.688652% / -1.010359 s |
| RTX5090, Slurm16007 | 11.085675675421953 s | 10.544288743287325 s | -4.883662% / -0.541387 s |

Every warm corresponding5090 ratio is 0.9491–0.9523, below the unchanged1.05
non-regression limit. Cold single observations improve18.35358353704214→
17.39413031935692 s (A100) and29.316666338592768→28.656750690191984 s
(5090); these are not cold distributions. `summary.json` reports every sample.
Cold10/10 iterations/Focks and warm1/1, recorded settings, anchor/delta/closure
counts and admission semantics match. Sixteen all-pair native/native observable
comparisons per device pass1e-9 Eh/atom,1e-7 max and3e-8 RMS Eh/Bohr gates.

The prior [5090 reference](../direct-cooperative-md-20261105/README.md) is reused
for independent common observable gates only (64 further pairings), not for
cross-device speed rankings. There is no fresh same-A100 GPU4PySCF timing.
Do not add savings across campaigns to declare the original1.576s gap closed,
or claim wb97m-v hybrid performance from a pure-J cohort.

## Science and safety

- 585 host cases pass;164 invalid same-shell/pair signature combinations skip
  explicitly, not numerical or GPU acceptance cases.
- Six actual5090 Libcint cases pass: RKS/UKS, Cartesian/spherical,
  Separate/Combined/replay. Pure-J work admits exactly classes5,8,9,11,12,13,16;
  hybrids retain ordinary cotangents. Executed test source is retained losslessly.
- Memcheck passes one full seven-class case, zero reported errors, exit0.
- Racecheck reaches its180s cap, exit124, without a diagnostic: **inconclusive**.
  The combined auxiliary Slurm16008 runner exits1, not success. Its FAILED
  scheduler state was inspected during the run; Slurm later purged the job.
  No terminal scheduler text is fabricated from the runner status.
- The extension changes thread-private expressions and pair arrays, not shared
  storage, barriers or scatter synchronization. Promotion is scoped to the
  independent numerical, memory and endpoint gates; it is not sanitizer-green.
- 1165 focused host default-policy/allocation/resource-identity/tile tests pass
  after promotion. An initial wider local resource suite has11 native-library
  discovery failures (no local `libgenerativeqc`); those do not become passes.

## Source and build identity

The same normal Release O3 SM80+SM120 candidate ELF runs on both devices.
Actual build uses verified sccache0.16.0 and CUDA12.9.86, with no compilation
failures. Immutable build/runner/version receipts are included.

| Arm | Native source identity | ELF SHA256 |
| --- | --- | --- |
| Baseline | `b78434b9416f124db8bf73b3b339bae894027959ed0220f6731e1bb1b10c2293` | `05319ea1eada9df72daa64fe05d131c553de582211b6b594d4d2f44f5fba8871` |
| Candidate | `4c54ccfb3d38e5ec86b147a019d8ef99e87c8b8247ca5a42bf50a84027dc61ed` | `3142176ecd915a1fe5eaaced6aa5cb4b3da3af6a1ff7b71b4cf6a2255f199799` |

`measured-source.patch.xz` reconstructs the three measured compiler/native
changes relative to base `7fc4ae8f6a6e4e31509f0fa7b3409d7daa72d83d`.
Measured activation uses `GENERATIVEQC_EXPERIMENT_DIRECT_FORCE_PAIR_SEPARABLE=1`.
`integration-delta.patch.xz` retains master08c9's disjoint nonlocal batching
change; the semilocal first branch exercised here is unchanged.
`promotion-delta.patch.xz` records subsequent whitespace and formal policy/
resource-identity changes to measured production files. Final policy is unset/1
on,0/empty/unknown off via `GENERATIVEQC_DIRECT_FORCE_PAIR_SEPARABLE`, with no
old alias. These deltas are audited/host-tested, not a new full GPU build;
neither final commit nor generated catalog relabels the measured ELF.

Raw compiler/profiler inventories and pre-link resource dumps remain ignored.
Modeled live-state reduction is not a measured register reduction; pre-link
object metadata is not final-linked stack evidence. No binary, source/runtime
tarball, external backup, Release or release asset is published.

## Offline replay

From the repository root, without either engine or GPU:

```bash
python benchmarks/results/direct-complete-pair-20261106/replay.py
cd benchmarks/results/direct-complete-pair-20261106
sha256sum -c SHA256SUMS
```

Replay checks full populations, source/core receipts, authentic scheduler
visibility, settings/work equality, unchanged numerical gates,5090
non-regression, every declared independent XML case and the explicit failed/
inconclusive safety boundary. Output must reproduce `summary.json`.
This bundle also needs the prior compact-MD `replay.py` and its original5090
reference population/exit receipt, all retained in Git. `.xz` producers and
XML decompress to the actual original bytes, not reformatted reconstructions.
Two tiny original sanitizer logs use exact-byte policy exceptions; broad raw
logs remain ignored. Original completion times and fail-fast runner exits
prove process completion; purged scheduler records are not invented.
