# Native Modified Hückel complete-endpoint qualification

This reviewed copy retains compact scalar gates, exact timing samples, scientific
work counts and the measured production patch. `qualification.json.gz` contains
the compact records; `recovery.json.gz` binds their original full records and source
audit to existing Git history, preserving exact bytes without exceeding the
checkout evidence budget. Dense density exports remain in ignored local run
storage; regenerate them for a full replay audit. The historical policy is superseded
by the later user-authorized default/DF decision.

Retained `.gz` files decode byte-for-byte to their original payloads. Read them
with `gzip -dc <path.gz>`; decompress patches/scripts before applying or running
them. The [storage inventory](../default-huckel-df-20261012/storage.json) binds
both encoded and decoded identities. Recovery inventory paths still name the
original files at the historical revision, not their current gzip containers.

This campaign measures the native provider, not the earlier PySCF-backed
Modified-Hückel experiment. It qualifies the frozen source described here,
not the complete dirty checkout or the subsequent CUDA HF integration.

## Frozen source and execution

- Base: `4ac8d4517f3323c5b7ba567270bdb2bd3323cde7`, plus the four-file
  eigenpair-presentation repair and native Hückel implementation.
- Library SHA-256:
  `61c6a2f222b11b8a5b679b38683a70b1fd9cbc4b6b9ecfb2450b65977449b9fb`.
- The recoverable `source-audit.json` hashes the complete production inventory
  and identifies all 15 changed production files. `measured-production.patch.gz`
  retains the final built production source. The complete measured-source patch
  and exact runner/tests are recoverable through `recovery.json.gz`.
- Build jobs 7806 and 7810; dedicated endpoint jobs 7830 (48 atoms) and 7831
  (96 atoms), node1/n1, one Slurm-assigned RTX 5090 and eight CPU cores each.
  Device visibility is preserved, finite limits are 15/25 minutes, and each
  case retains the same device across its arms and repeats.
- Verified ccache 4.5.1, checkout-root `CCACHE_BASEDIR`, reused compiler caches.
  Actual CXX/CUDA commands and before/after cache receipts are retained.
- Exact FP64 PBE0/RKS, spherical offline def2-SVP, unpruned moving
  `GridSpec(48,16,32)`, identical target convergence and screening controls.
  Independent reference records are reused byte-for-byte from the earlier
  campaign and their hashes are checked by every sample.
- Three fresh-process/owner samples per arm, rotating `core` / `minao` /
  `huckel` order. The endpoint includes preparation, native atomic construction,
  strict seed admission, synchronized energy and analytic forces, and any retry.

The build-plus-test job 7810 reached its finite limit after starting an
incomplete endpoint. Its interrupted receipt/log is retained separately under
`receipts/interrupted-7810`; it contributes no observation to the fresh campaign.
The native contract suite and 111 native/CPU Python tests passed before that
interruption. The host regression and benchmark-checker receipts are separate.

## Results and decision

| Arm | 48 median seconds | 96 median seconds | Reduction versus MINAO, 48 / 96 |
| --- | ---: | ---: | ---: |
| Hcore | 43.682298 | 112.777171 | control |
| Native MINAO | 37.589319 | 88.251705 | existing-default baseline |
| Native Hückel | 34.154173 | 86.213632 | 9.14% / 2.31% |

All 18 native endpoints pass independent energy/force, physical residual and
paired final-density gates; all six Hückel seeds are used with complete atomic
work counts and no target retry. Maximum errors are `6.82e-12 Eh` in energy,
`3.77e-11 Eh/Bohr` in force components, `1.77e-12` physical residual RMS and
`5.48e-11` paired AO-density error.

Hückel passes the strictly >5% complete-endpoint gate at 48 atoms only. Its
paired reductions there are 9.14%, 9.08% and 9.53%. At 96 atoms they are 2.25%,
-1.77% and 19.49%; the last large gain accompanies a 22-Fock MINAO observation,
which is retained rather than cherry-picked as a default-policy justification.
The 96-atom median reduction is only 2.31%. **The guarded MINAO/Hcore default is
unchanged.** Two water fixtures do not justify a size whitelist or universal
promotion to unmeasured methods/backends.

| Arm | 48 target iterations/Fock builds | 96 target iterations/Fock builds |
| --- | --- | --- |
| Hcore | 23, 23, 23 | 24, 24, 24 |
| MINAO | 19, 19, 19 | 17, 17, 22 |
| Hückel | 17, 17, 17 | 17, 18, 17 |

Hückel preparation/admission medians are 0.129302 / 0.927277 seconds versus
MINAO's 0.317283 / 2.359682 seconds. Every Hückel preparation counts seven
atomic iterations/Fock builds; hydrogen performs no J/K. Native construction is
already cheaper in both cases, but the 96-atom target work does not consistently
decrease. The recoverable `work-ledger.json` retains all phase timings, atomic and target work,
AO/XC census and complete analytic-force work records.

## Evidence and reproduction

The original 20 JSON gzip records, independent references and full work ledger
are recoverable byte-for-byte from the revision and hashes in `recovery.json.gz`.
The exact runner is also `benchmarks/pbe0_initial_guesses.py` in this checkout;
its historical identity is retained in every sample. The original local run
retains 18 lossless NumPy gzip density exports. Recover/regenerate them into a
scratch directory without renaming the resulting records; density
hashes refer to the original `.npy` bytes. Summary generation is host-only:

```bash
python -m benchmarks.pbe0_initial_guesses summarize --native-huckel-only \
  --directory /path/to/decompressed-records --output /path/to/summary.json
```

For any listed original payload, use
`git show <git_revision>:<path> > <scratch-path>` and verify its SHA-256 from
`recovery.json.gz` before replay. Transient endpoint/build scripts and compiler-cache
receipts remain in local run storage, not this retained copy.
Adapt only machine-local environment paths for reproduction. Real-device replay
must remain inside a compatible finite `srun`, preserve its assigned visibility,
and set `RUN_GPU_ENDPOINTS=1`; run summarization only after both case jobs finish.
