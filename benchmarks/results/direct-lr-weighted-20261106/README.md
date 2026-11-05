# Direct LR weighted forces — measured candidate

Issue #2976. This is a separate follow-up to PR #3014, not another measurement
of its pure-J pair projection. The exact order-four/five LR exchange weights
are contracted before the existing weighted-ERI derivative graph. No DF,
low-rank assumption, GPU-name dispatch, CPU production oracle, new recurrence,
precision reduction, grid or screening change is introduced.

## Complete endpoint measurements

Water96, 768 spherical def2-SVP AOs, WB97M-V full-Direct energy and analytic
forces, including moving quadrature and VV10. Cold includes construction,
preparation, SCF and host export. Warm uses the same prepared owner and its own
frozen cold density. Each arm has one cold plus three warm calls; samples are
retained in `measurements.json.xz`, not selected by speed. There is no moved
geometry population or cold distribution in this cohort.

The **LR-only attribution** is the same candidate ELF with captured selector
`GENERATIVEQC_DIRECT_LR_WEIGHTED_FORCES=0` versus `1`:

| Actual allocation | Cold off → on (single observations) | Warm median off → on |
| --- | --- | --- |
| A100, ECNU LSF225805 | 117.483224 → 113.216387 s | 19.467532 → 15.051496 s, **-22.68%, -4.416036 s** |
| RTX5090, Slurm16016 | 197.351884 → 195.343316 s | 16.520271 → 14.406079 s, **-12.80%, -2.114192 s** |

Cold has 18 iterations/18 Focks; every warm call has 1/1. All recorded settings,
VV10 sweeps/cache probes/provisional/target Focks and J/K applications match
within each same-image pair. Paired gates remain 1e-9 Eh/atom, 1e-7 force max
and 3e-8 force RMS Eh/Bohr; maximum force differences are below 1e-12.

### Previous-image non-regression control

The separately qualified prior image remains an additional whole-endpoint
control, not isolated LR attribution. Its WB force tile is 256 rather than the
already-synchronized upstream default 1024. That schedule-only difference is
explicitly retained by replay; all other settings and semantic counts match.
Do not attribute its whole warm improvement (A100 21.611487 → 15.051496 s;
5090 18.731600 → 14.406079 s) to this LR change or call it an isolated compiler
frame improvement.

RTX5090 r²SCAN-3c exact Direct, 1088 spherical def2-mTZVPP AOs and native D4/gCP,
also retains its complete previous-image control: cold 29.009568 → 28.863739 s;
three-warm median 10.614057 → 10.613558 s (ratio 0.999953). Settings and work
match. This protects the unrelated semilocal route against regression.

## Independent science and explicit failures

- 392 new actual-native host LR cases pass: eight classes, RKS/UKS, omega
  0.03/0.3/1.7, signed exchange, orientations, legal shell/pair triangles,
  repeated atoms, sparse/equality/empty masks, zero exchange and Inf/NaN retry.
- Shared pair/allocation regression: 511 prior pair cases and 1157 allocation
  cases pass; 164 invalid physical-signature combinations skip explicitly.
- Eight actual5090 independent Libcint cases pass: RKS/UKS, cartesian/spherical,
  selector0/1, independent J/SR/LR channels and repeated stationary exports.
- A wider local host run has 2076 passes and 11 native-library-discovery
  failures. All27 resource cases subsequently pass with the actual built
  library in Slurm16016. `science/test-cases.json.xz` retains every test name,
  outcome, failure/skip detail and original XML digest, including both failed
  attempts. Raw transient XML is not published as new repository evidence.
- An initial fault fixture incorrectly expected a huge finite relative result
  to be nonfinite: 8 failures/106 passes. Its Inf/NaN correction has114 passes;
  the separate pair-projection finite-magnitude boundary is unchanged.
- Another154 host/default/source-contract cases pass. New-source type checks
  and applicable pre-commit checks pass; unrelated broader test-module typing
  errors are not rewritten to claim a blanket type-suite pass.
- The original independent GPU4PySCF static reference is reused only for48
  observable pairings, with its original source/package/control receipts and
  exact cohort binding. `reference-reuse.json` retains its digest and unchanged
  common gates. No historical reference timing is reused as a current ranking.
- No new sanitizer pass is claimed. The change adds no shared memory or barrier;
  prior racecheck timeout remains inconclusive, not a blanket sanitizer receipt.

## Provenance and integration boundary

Measured base: `b27caeec846b6fd28597c06cbbb55498c1e3d730`.
`measured-source.patch.xz` reconstructs the exact seven production overlays.

| Image | Source identity | ELF SHA256 |
| --- | --- | --- |
| Previous qualified | `4c54ccfb3d38e5ec86b147a019d8ef99e87c8b8247ca5a42bf50a84027dc61ed` | `3142176ecd915a1fe5eaaced6aa5cb4b3da3af6a2255f199799` |
| Measured LR candidate | `c313decb8884e69c3b44b6fe2e72c9491869b64313db3e4fcdd56f8e5f52a2a1` | `d9b33f54155186cc3240ecd325a290bcd1fecadcc136ed9c8ce72107f88f3486` |

Normal Release O3, CUDA12.9.86, verified sccache0.16.0, SM80-real+SM120-real,
successful actual link. Final linked resource inventories retain all216 bounded
kernel rows per image: only18 change. Default full-range stacks do not grow;
explicit angular LR4/5 workers do have larger stack reservations and are not
independently promoted by these default measurements. Resource reservations do
not establish runtime traffic or speed.

Master advanced during measurement to `1943a24b7` (#3008, uniform SCF defaults).
The follow-up branch preserves PR3014's reviewed rollback fix and follows its
new head `9a72d770a`. The integrated source is
`ff6117e3ba5b481e6124f3bf80f4613fa27586d77bff88f5187f9290abbf5afa`;
its rebuild/default endpoint verification is **pending**, distinct from the
sealed measured image. This is not yet a final integrated-image qualification.

LSF225805 is DONE and its runner exits0. SCP copied the complete archive but
did not terminate normally; a separate remote SHA256 receipt authenticates the
bytes before use. No GPU cohort was repeated because of transport errors. The
task-owned finite transfer server is stopped; shared VPN/SSH sessions remain.
No binary, tarball, Release, tag, release asset or external backup is published.

## Offline replay

```bash
python benchmarks/results/direct-lr-weighted-20261106/replay.py
cd benchmarks/results/direct-lr-weighted-20261106
sha256sum -c SHA256SUMS
```

Replay checks all original populations, image/scheduler identities, settings,
work, numerical gates, every declared failure boundary and the non-regression
controls. It reproduces `summary.json` without either engine, GPU or network.
The independent reference remains owned by
`../direct-range-warm-anchor-20261105/publication.json.xz` in Git. Historical
measurements are never relabelled as the later master-integrated image.
