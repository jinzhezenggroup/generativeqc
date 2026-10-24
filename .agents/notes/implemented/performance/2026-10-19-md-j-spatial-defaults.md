# Decision: default to complete primitive J and spatial semilocal XC

Status: implemented
Date: 2026-10-19

## Decision

An unset `GENERATIVEQC_MD_J_ENGINE` requests the complete `primitive` recipe.
An unset `GENERATIVEQC_CUDA_XC_SPATIAL_ORDER` requests spatial execution in the
already qualified resident semilocal/local-AO domain. The user explicitly
requested promotion after the matched endpoint qualification. This supersedes
the opt-in decisions in [primitive J](2026-10-19-primitive-complete-md-j.md) and
[spatial XC](2026-10-19-xc-spatial-grid-order.md), without rewriting their
historical source identities or measurements.

## Boundaries and rollback

Primitive J uses independent conservative Coulomb norms and a cumulative
per-entry omission bound, rather than the old exact per-AO integral mask.
`GENERATIVEQC_MD_J_ENGINE=reference` and `incumbent` retain that exact-mask
contract. Insufficient hierarchy capacity retains incumbent MD; insufficient
base storage or optional allocation failure retains normal J. The owner's
actual remaining allowance controls admission, not a second 128-MiB clamp.
`GENERATIVEQC_DISABLE_MD_J=1` still disables all MD preparation. Public-ledger,
unsupported-basis, precision, response and derivative boundaries are unchanged.

Spatial XC permutes the exact points and weights before local AO discovery.
The canonical resident grid remains authoritative for moving-grid forces.
Explicit `GENERATIVEQC_CUDA_XC_SPATIAL_ORDER=0`, insufficient allowances or
optional device OOM retain canonical execution. Nonlocal and public-ledger
owners remain excluded. No AO threshold, precision, quadrature, DIIS default,
K provider or derivative kernel is changed by this promotion.

## Qualification

The preceding source-matched qualification reduced 96-atom r2SCAN-3c complete
cold E/F from 236.820267 s after the capacity fix to 157.723791 s, versus
153.635193 s GPU4PySCF 1.8.1 (fresh repeat 153.314540 s). PBE0 complete cold
was 62.549000 s versus 76.804166 s upstream. These are matched FP64/direct/noDF
48x16x32 unpruned moving-grid implementation diagnostics, not independently
grid-converged production rankings. The linked notes retain all work counts,
population sizes, independent numerical gates and limits of the comparison.

Default promotion additionally checks the unset selectors through executable
resource/failure probes, independent raw-J and captured changed-density GPU
tests, retained exact-mask tests, strict/fast RHF/UHF/PBE0 energy+force endpoints,
and clean complete 96-atom r2SCAN-3c/PBE0 E/F with both enabling selectors absent.
Source-matched receipts are retained under `.artifacts/pr2609/defaults/` and
the isolated `/data/jzzeng/qc-pr2609-defaults-20261019` remote snapshots.

The promoted source identity is
`63aa902bfba5163c8f0745dc099b6e3b9f135f51f7a9f20ffc4f650941ac7f35`;
the matching library SHA256 is
`b1eb32f609bd6c006fb333135de6ca3af57b31019b63bd7473cdb012f8e7915c`.
Both n2 and n4 receipts verify the archived source against the loaded library.
The compiler closure is unchanged from the preceding qualification; all six
stationary-force manifest targets were checked and the matching artifacts
deployed. The incremental native build used verified ccache 4.5.1; its retained
before/after statistics show 5 hits and 23 misses, with no additional cache
errors. These are shared-cache observations, not exclusive per-build counters.

Host math/lifecycle/grid integration: 921 passed; selection inventory: 128
passed; benchmark/reference harnesses: 208 passed. The final resource suite
was rerun after adding real 48/96-atom default exact-capacity checks: 17 passed.
The obsolete inventory assertion requiring a literal 128-MiB hard cap was
removed; executable exact-fit, one-byte-short, ledger and failure rollback
coverage remains authoritative.

n2:3483 passed 112 GPU cases. Two legacy reciprocal/oriented ablations still
implicitly selected the old default, so they compared the same primitive
recipe after promotion. They now explicitly select `incumbent`; n2:3484 passes
both corrected cases. The original failed receipt is retained. Together these
qualify 114 distinct GPU cases; one full-pinned-GPU4PySCF-package case remains
deselected because that complete package was not built. The default primitive
changed-density/tail graph and reference tail probes pass memcheck, initcheck,
racecheck and synccheck on n2:3484, with zero errors and zero race hazards or
warnings. These small-case sanitizer receipts do not qualify large-system
sanitizer execution. n4:1185 also passes the independent spatial-grid E/V,
captured replay, resource and canonical-view native suite.

### Ordinary default endpoint results

n4:1185 runs with both enabling selectors and both MD counter controls absent.
All 18 complete E/F endpoints pass the independent `1e-8 Eh` energy and
`1e-7 Eh/bohr` maximum-force gates. The maximum observed r2SCAN-3c errors are
`8.64e-12 Eh` / `4.72e-11 Eh/bohr`; PBE0 errors are `6.82e-12` / `2.45e-11`.
The scope includes preparation, SCF, full moving-grid force, corrections and
host publication. Warm samples freeze the post-cold density update policy.

| Endpoint | Default complete seconds | Samples | Fock builds per sample |
|---|---:|---:|---:|
| r2SCAN-3c cold | 158.655704 | 1 | 24 |
| r2SCAN-3c warm median | 25.963102 | 5 | 1 |
| r2SCAN-3c moved | 97.468332 | 1 | 13 |
| r2SCAN-3c moved-warm median | 25.944059 | 5 | 1 |
| PBE0 cold | 62.452559 | 1 | 17 |
| PBE0 warm median | 10.373085 | 5 | 1 |

Every endpoint's spatial AO inventory matches the preceding explicitly enabled
route. For original r2SCAN geometry, maximum block AO count is 450 and
point-AO-square work is 122,486,172,672. The moved solve takes 13 Fock builds
versus 14 in the earlier explicit-path sample; do not call that an additional
kernel speedup. Default r2SCAN cold is 33.0% faster than the capacity-fixed
reference sample and remains 3.5% slower than the fresh GPU4PySCF cold sample.
The preceding audit's remaining upstream differences are not declared solved.
All repository pre-commit checks pass. Raw receipts and the executable summary
are retained in `.artifacts/pr2609/defaults/`.

## Revisit when

Broader basis, precision, nonlocal, public-ledger or multiple-independent-density
admission requires new independent science, lifetime/resource and endpoint
qualification. Preserve explicit rollback recipes and semantic work counts;
raw-kernel wins alone do not justify another default change. Remaining upstream
differences and rejected alternatives are recorded in the
[complete source audit](2026-10-19-gpu4pyscf-cold-source-audit.md).
