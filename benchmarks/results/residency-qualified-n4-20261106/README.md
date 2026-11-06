# Qualified scientific DF links and named HF replay boundaries

This completes #1629's actual-path qualification using the existing CUPTI/source
observer, not a second recorder. It adds a real scientific GPU→host-transform→GPU
chain and narrowly declared post-prepare replay boundary contracts. **Global
residency remains INCOMPLETE** for unqualified implicit/library/graph waits;
the scoped result is not a universal residency PASS or a GPU speedup claim.

## Allocation and source/build identity

Final Slurm job **1511** runs on SSH `n4` / node `node4`, `main`, `gpu:5090:1`,
with an explicit **15-minute** limit and assigned `CUDA_VISIBLE_DEVICES=0`.
RTX 5090 UUID is `GPU-5430ba2e-194a-b6d1-f12c-6b99f6d34efd`, driver 580.178.04.
No GPU runs bypass Slurm or use maintenance node n3.

Frozen source is base commit `12d5142a3c0c15d803704bfa2a4d6c8765b1ab1e` plus
`source.patch.xz`, Git tree **6412823832ea0cb532972b57fa441230bea33aa4**.
The ignored full run archive includes its file/blob manifest; installed source is
checked inside the job before capture. Later documentation/tests and stronger CPU replay
checks are not relabeled as that frozen tree. Native build inputs are unchanged
between the retained build/capture manifests. Ignored build evidence includes CMake
caches, actual cached compiler commands and logs. Release SM120 uses GCC 11.4,
CUDA 12.9.86 and verified sccache 0.16.0, with optional CUDA shell/stationary AOT
disabled. Native DSO SHA-256:
`20457bdee1b33294b9d8bb416fde59b07453009f529de329681e69f726dd0401`.
Collector SHA-256:
`42236a485cb2a28f9c22212135b282b5184676ef6679081af5e9f564c7ac0618`.

## Actual scientific host transform

The **public CUDA DF-MP2** endpoint performs real metric eigensolver eigenvalue
D2H, existing completion fence, checked CPU cutoff/inverse square root, scales
H2D, then the existing device scaling consumer. Explicit source generations and
named transform completion establish the link; equal bytes or timestamps do not.
Each H2/LiH/water capture qualifies both HF and RI-MP2 metric preparation chains.

| Fixture | Metric-owner H2D / D2H bytes | Stream fences | Maximum energy control error |
| --- | ---: | ---: | ---: |
| H2 | 96 / 40 | 2 | 1.87e-14 |
| LiH | 672 / 104 | 2 | 1.89e-13 |
| Water | 896 / 120 | 2 | 2.85e-13 |

The unchanged 1e-9 energy gate compares captured/ordinary CUDA, native CPU and
committed independent same-Hamiltonian fixture energies. This is permitted
preparation residency traffic. It does not invent an H2D dependency for the older
generated-DF-to-host-MP2 compatibility journals, whose unknown links remain
unknown. Those four independent energy/amplitude controls, plus real synthetic
iteration/tile/publication and extra-fence negatives, remain retained in
[the earlier campaign](../residency-n4-20261106/README.md).

## Declared resident replay scope

Prepared RHF and UHF execute `[H2, water, H2]` first/warm energy, first/warm E+F,
and an internal moved-geometry E+F, with clean ordinary-CUDA and independent
native-CPU histories outside capture. Energy/density solver tolerances are
identically pinned at **1e-12 / 1e-11** for all histories. The original acceptance
gates remain **1e-10 Eh / 1e-9 Eh/Bohr**.

| Method | Maximum independent energy / force error | Observed boundary ratchet |
| --- | ---: | --- |
| RHF | 1.28e-13 / 8.79e-13 | WITHIN_OBSERVED_RATCHET |
| UHF | 7.11e-14 / 5.60e-13 | WITHIN_OBSERVED_RATCHET |

The version-3 policy was declared from **historical job 1506**, not learned from
1511. It pins warm energy/force explicit work for `hf_bucket` and named
region/role/site/payload/kind/direction boundaries. Unexpected transfers/waits
have zero allowance; equal-byte scientific-payload substitution is rejected.
`declared_resident_scopes.zero_unexpected_post_prepare_host_round_trips=true`
applies only to these declared observed explicit boundaries. Global
`zero_round_trip_assertion=null`, false dependency/implicit/graph completeness
flags and capture exit **2** remain correct. Offline ratchet replay exits **0**
for its explicitly scoped work gate, not for the strict global residency audit.

## Preserved failures

Job 1509's harness/linker failures and job 1510's additional **F-HeH full-SCF
CUPTI launch failure** are retained as logs. F-HeH is not a supported passing
scientific capture here; its historical generated-provider compatibility test
is a different endpoint. The earlier capture harness failed before writing raw
F-HeH journals; none are fabricated. Failure handling now writes available raw
journals/status and preserves the original endpoint exception even when the
observer fence also fails. The earlier RHF moved-force failure is preserved in
the historical campaign. Tighter common solver controls qualify the present
workload, not a general change to production defaults.

## Artifacts and reproduction

Permanent `tests/reference_data/cupti/n4-1511/{h2,lih,water,rhf,uhf}.tar.xz`
fixtures contain each CPU replay test's exact raw API/activity/source/graph
journals, numerical controls and pinned contracts. A per-member digest migration
receipt at `manifests/evidence_migrations/mp2-residency-20261106.json` verifies
lossless storage; all fixtures remain below the unchanged 1 MiB cap. Full mixed
run/build bundles, failure logs, frozen manifest and finite-job runner remain in
ignored local `.artifacts/issue-1626-1629/evidence-before-compaction/` and
`.artifacts/issue-1626-1629/completion/`, not a benchmark archive publication.
`source.patch.xz` reconstructs the frozen tree from its base commit.
`provenance.json`, `qualification.json.xz` and `checksums.sha256` retain identities,
rederived work and artifact integrity. No aggregate storage budget or scientific
gate is raised. Raw journals are the authority, not saved diagnostic verdicts.
No binary libraries or release assets are posted.

Reproduce on a clean frozen workspace using the retained `run-gpu.sh`:

```bash
ssh n4 'cd <workspace> && srun --partition=main --gres=gpu:5090:1 \
  --nodes=1 --ntasks=1 --time=00:15:00 bash run-gpu.sh'
PYTHONPATH=python:. python -m pytest -q \
  tests/python/test_cupti_qualified_production_replay.py \
  tests/python/test_cupti_df_metric_lineage.py \
  tests/python/test_cupti_source_ratchet.py
```

The CPU tests reconstruct source/API joins and independent fixture comparisons,
reject broken dependency identities and same-count payload substitution, and
retain the earlier real iteration/tile/extra-fence regression controls. They do
not authenticate device provenance offline or claim unavailable wait coverage.
