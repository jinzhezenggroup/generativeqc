# Source-scoped incremental KS qualification and negatives

This is draft #1803 evidence, not a default promotion or a resolution of the
retained OH convergence failure. Each campaign retains cold, five warm, moved
and five moved-warm complete energy/analytic-force calls, without filtering
iterations, retries or slower samples. There are 120 native and 60 independent
reference calls and 720 matching-geometry native/reference comparisons.

| Source | Matching commit | Large endpoint scope |
| --- | --- | --- |
| Initial `8269ab44…` | `e52b13c0f` | 48/96 atoms |
| Audited-full-RKS reuse `fe6aea98…` | `891b533b3` | 48/96 atoms plus H2CO |
| Observer/master integration `61c68d2c…` | `80fa2844c` | Energy-only 96-atom work trace, not large E/F qualification |

Original preparation receipts, loaded-library hashes and dirty checkout bases
remain unchanged. The observer integrates master `79418329e`; the later
`d9431c913` documentation update and subsequent J-screening experiments are not
measured here. Device/toolchain and actual GPU4PySCF XCfun routing remain in
each endpoint receipt. Full-grid PBE0, def2-SVP spherical, exact direct J/K,
strict FP64 and original 1e-8 Eh / 1e-7 Eh/Bohr E/F gates remain matched.

## Endpoint observations

| Campaign/size | OFF warm median | ON warm median | OFF/ON moved |
| --- | ---: | ---: | --- |
| Initial 48 | 17.860668 s | 22.421234 s | 70.893/84.116 s; 12/16 iterations |
| Initial 96 | 77.213533 s | 123.139572 s | 258.265/324.892 s; 12/17 iterations |
| Audit 48 | 17.853868 s | 17.850991 s | 70.829/75.922 s; 12/14 iterations |
| Audit 96 | 78.024929 s | 78.010056 s | 260.733/297.989 s; 12/15 iterations |

Two initial 48-atom warm ON calls retry cold and take 143.253/148.156 seconds.
Returned-attempt counters omit those discarded attempts: they are not complete
endpoint work counters. The separate warm diagnostic must not erase these
observed retries. Reusing an already full-density converged RKS build restores
one-build warm parity in the audit campaign, not a speedup over OFF. UKS and
delta-built convergence retain the extra full closure.

Cold ordering shares caches; differing SCF trajectories and moved regressions
remain visible. These are not interleaved causal speedup measurements. Every
E/F comparison passes, with maximum error about 1.08e-10 Eh / 3.29e-11 Eh/Bohr.

## Actual provider work

Jobs 5678/5679 observe separate J/K class counters for a fresh 96-atom SCF solve
and two frozen warm replays. All 21 present s/p/d classes are covered, including
native DDDD. Unit: **admitted streaming shell-task dispatch**, not candidates,
rejections, primitive recurrence, roots or FLOPs. Extra atomics/readbacks make
these energy-only runs unsuitable for clean timing. All six final energies pass
against all six matching-geometry reference energies.

J admits **142,757,104 tasks on every build**, including all 13 delta builds.
OFF cold uses 26 builds; ON uses 14 full plus 13 delta builds. K delta admission
falls from 108,832,786 tasks to 5,197 at the last delta. Thus actual K work reacts
to delta density while this J admission gate does not. Both warm replays use
one full build, with 142,757,104 J and 72,116,584 K dispatches. No new screening
optimization or primitive-work reduction is claimed by this publication.

The 29,997-byte input contains only geometry and basis. The probe solves SCF
anew; it neither consumes nor claims replay of the unpublished historical force
density. The measured probe's leading comment still mentions an ignored density
suffix; the retained file has none and the parser reads only geometry/basis.
Jobs 5675/5676 were cancelled after diagnosing an eager CPU grid-construction
harness mismatch, before any SCF iteration record. Their source and cancellation
receipt are retained; the repaired probe uses production's resident grid factory.

## Controls and open gates

`controls-and-failures.json.gz` retains the initial FD-test adapter failures,
their successful corrected runs, native/linear sanitizer controls, the first
two failed observer fixtures and successful device-input repair. The unchanged
whole-HF density-bound reduction reports 96 synccheck errors, independently
reproduced on the retained unmodified-master control. Passing isolated linear
kernels is not a blanket sanitizer certification. Broad UKS, OH convergence,
latest-source large E/F, tighter resource controls and genuine review remain open.

## Reproduction

Run `python benchmarks/results/pbe0-incremental-ks-20261004/verify.py` for offline
all-repeat numerical/work conservation checks. Scripts with `.txt` suffix retain
the exact runtime harnesses, including site-specific paths; adapt those paths,
not scientific settings, for a new run. The ordinary endpoint command is
`python -m benchmarks.readme_pbe0 reference --atoms 96 --basis-file
benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json --repeats 5 --output
.artifacts/reference.json`, followed by `native --reference` and the same options.
Set `GENERATIVEQC_INCREMENTAL_DIRECT_JK=0` or `1` explicitly. Build the matching
source with CUDA 12.9.1, sm_120 AOT, explicit CXX/CUDA ccache launchers and a
checkout-root `CCACHE_BASEDIR`; record the new library identity rather than
assuming bitwise reproducibility. Every GPU command must run through a finite
`srun --partition=main --gres=gpu:5090:1` allocation. No external archive or
release asset is required; no storage cap is raised.
