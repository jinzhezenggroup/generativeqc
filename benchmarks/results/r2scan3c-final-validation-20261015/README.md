# r²SCAN-3c resident final-state proof, 2026-10-15

This campaign measures complete **energy + analytic forces**, not isolated
matrix products or force kernels, from task-start master
`cb8781fb1bdcfb9166ef07cfed463c0482dea752` plus the retained source patch.
Both README panels use the defining spherical **def2-mTZVPP** basis for
r²SCAN-3c; other methods retain their historical def2-SVP observations.
Changing to a smaller basis is not this optimization.

The superseded README plots are retained losslessly as
[direct](dft-cold-warm.svg.xz) and [DF](df-cold-warm.svg.xz) archives.
Use the checksum-verified recovery below for viewable SVGs. The verifier rejects
the retained Direct/96 moved endpoint, so `refresh.py --output` does not render.
Observations, source identities and the acceptance hold remain unchanged.

From this directory, recover both figures into a new temporary directory:

```bash
python - <<'PYCODE'
import hashlib, json, lzma, tempfile
from pathlib import Path
out = Path(tempfile.mkdtemp(prefix="r2scan3c-figures-"))
manifest = json.loads(Path("manifest.json").read_text())
for name in ("dft-cold-warm.svg.xz", "df-cold-warm.svg.xz"):
    entry = manifest["files"][name]
    data = Path(name).read_bytes()
    for receipt in (entry, entry["uncompressed"]):
        if len(data) != receipt["bytes"] or hashlib.sha256(data).hexdigest() != receipt["sha256"]:
            raise ValueError(name)
        if receipt is entry:
            data = lzma.decompress(data)
    with (out / name[:-3]).open("xb") as stream:
        stream.write(data)
print(out)
PYCODE
```

The implementation automatically admits strict-FP64, all-electron,
restricted, full-range pure-J owners with at least **544 AOs**, with either
exact-direct or density-fitted J. It reuses the existing resident final-state
proof, removing two redundant full-AO products (eleven to nine) and running
the remaining products on the existing matrix-library owner. The separate
occupied-rank density reconstruction remains: the trace counts ten products.
Scientific acceptance, dense host exports, weighted density, D4/gCP, grid,
screening and SCF stopping rules are unchanged.

## Comparison and enablement

Control forces `GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION=0`; candidate
leaves it genuinely **unset**. Both leave the force-domain selector unset.
Every arm starts a fresh process and Calculator, includes preparation in cold,
and retains one cold, five fixed-density warm, one moved and five moved-warm
endpoints. Compiler caches are retained, not cleared or subtracted from timing.
SCF iteration/Fock-build counts are preserved without normalization.

| Route / atoms (AOs) | Warm control → default (s) | Reduction | Moved-warm reduction |
| --- | ---: | ---: | ---: |
| Direct / 48 (544) | 22.409 → 21.951 | 2.045% | 1.868% |
| Direct / 96 (1088) | 92.089 → 88.312 | 4.101% | 4.068% |
| DF / 48 (544) | 6.350 → 5.941 | 6.439% | 6.318% |
| DF / 96 (1088) | 21.476 → 18.809 | 12.417% | 12.498% |

The descriptive robust-benefit label is
`max(2%, 2*(relative_MAD_control + relative_MAD_candidate))`, **not an
enablement threshold**. Direct/48 moved-warm honestly remains below that label;
its small positive improvement is retained and enabled. The user explicitly
rejected disabling useful improvements merely because they are below 2%.
Complete numerical correctness, resource/lease guards and complete-endpoint
nonregression beyond `max(5%, the same noise gate)` remain blocking checks.
Changed SCF work limits causal attribution; it does not waive a failed endpoint
nonregression gate. Warm comparisons additionally require matched SCF work. Cold/moved have one
sample per arm and do not support a significance claim. Small 3/12-atom owners
retain the reference proof and serve as non-admission controls.

Direct/96 cold performs 30 versus 26 SCF iterations, and moved 17 versus 21.
Their complete times are 516.372 → 456.842 s and 327.889 → 380.302 s,
respectively. The moved sample keeps its failed descriptive nonregression
flag. Both phases are work-confounded, so causal attribution to the validation
route is unresolved; no time is normalized by iterations or removed. The moved
endpoint is nevertheless an observed slowdown and blocks performance acceptance.
Warm/moved-warm retain all five matched-work endpoints for the route comparison.
The retained observation archive preserves its original descriptive classification.
The replay verifier now recomputes phase statistics and fails on the Direct/96
moved nonregression result; repairing that verifier does not resolve the slowdown
or qualify default promotion.

## Source and numerical qualification

- All **192 complete comparison calls** pass the independent energy/force
  gates. Maximum errors are 4.093e-11 Eh and 1.095e-10 Eh/Bohr. The two refreshed
  panels publish the eight candidate populations (96 complete native calls).
- All comparison arms use the same native SHA256
  `ef73b7bd00652e5850737f03d7dcfdab1f738ba5a8dd6d0d5c5429688f88cd40`.
- The binary advertises native source identity
  `953a3addc555388d8f025f0e0eeb7e02af6b3033775d8ad050bc53e582a21d66`,
  matching the frozen 1742-input inventory and all 71 benchmark consumers.
- Independent GPU4PySCF energy/force acceptance remains **1e-8 Eh /
  1e-7 Eh/Bohr for every returned call**, including moved geometries. DF uses
  the retained explicit cc-pVDZ-JKFIT auxiliary basis and its response.
- Finite Slurm allocations preserve device visibility: node5 job 1619 covers
  resident analytic/negative cases, real KS producers, repeated stationary
  leases and four changed-resource-schedule guards. The retained KS fixture
  adapts six diagnostic calls to master's extracted
  `BatchKsTransportDiagnostics` capability; production headers are unmodified.
  Its exact adapter and standalone build recipe are retained in receipts.
- Node1 job 8728 independently traces **actual unset-selector resident proof**
  for both Direct/48 and DF/48. Each cold/first-warm diagnostic has two valid
  544-AO proof events. These intrusive captures are not clean timing samples.
- Main comparisons run on RTX 5090: node5 job 1620 for Direct, node1 job 8726
  for DF. Points through 48 atoms retain 1800-second limits; 96 atoms use
  3600 seconds to collect all twelve calls without relaxing any scientific gate.
- Focused host tests: 62 pass. Production core and official stationary AOT
  manifests are built with the existing compiler cache; cache/build receipts
  are retained. The restored local core is byte-identical to the measured image.

## Retained evidence and replay

`observations.json.xz` retains all comparisons, per-call work and independent
cold/moved geometry oracles. `summary.json.xz` and `df-summary.json.xz` retain
only the refreshed r²SCAN-3c points. `refresh.py` composes other functionals
from `../readme-wb97mv-20261014/` without relabeling their historical identities.
`manifest.json` binds every archive, table and plot. `receipts.json.xz` retains
source/build/cache/test/scheduler records and the actual-route diagnostics.

The earlier explicit `0` versus `1` DF/48 pilot uses its own older frozen image
and is retained in `pilot-observations.json.xz`; it is not mixed into default
qualification. Both rejected cooperative force experiments remain separate
under `../r2scan3c-order6-20261015/` and `../r2scan3c-degree7-20261015/`.

With raw campaign artifacts present, independently recheck before reducing:

```bash
PYTHONPATH=python:. python benchmarks/results/r2scan3c-final-validation-20261015/reduce.py \
  .artifacts/r2scan3c-final-proof --output /tmp/r2scan3c-observations.json.xz
PYTHONPATH=python:. python benchmarks/results/r2scan3c-final-validation-20261015/refresh.py \
  --output /tmp/r2scan3c-readme
```

Raw directories and the frozen matching source/runtime remain locally under
`.artifacts/r2scan3c-final-proof/` and remotely under
`/data/jzzeng/gqc-r2scan3c-final-proof-20261015` on node1/node5. No binary,
release asset or tag is published. See the
[decision](../../../.agents/notes/implemented/performance/2026-10-15-pure-j-resident-final-validation.md)
for fallback, lifetime and scope boundaries.
