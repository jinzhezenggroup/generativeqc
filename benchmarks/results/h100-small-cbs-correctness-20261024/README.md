# H100 small CBS bounded correctness outcome

Refs #2727 and #2724. The **whole fixed32-slot matrix is INCOMPLETE**. This
publication retains actual H100 results and refusals, not a completed parent,
performance comparison, current-master SDK qualification or default promotion.

## Actual outcome

| Fixture/model | Requested | Returned | Remaining fixed slots | Outcome |
| --- | ---: | ---: | ---: | --- |
| Physical H2 DZ/TZ, conventional/exact | 1 E+F | 0 | 5 E-only NOT_RUN | Provider retained-allocation refusal |
| Synthetic #2681 water algebra, conventional/exact | 1 E+F +5 E-only | 6 | 0 | Original independent E/F, native parity/directions PASS |
| Archived water, exact-HF/correlation-only DF | 1 E+F | 0 | 9 E+F NOT_RUN | Same provider retained-allocation refusal |
| Archived water, full RI-JK | 10 E+F | 10 | 0 | Both geometries, Standard/Fast/repeat/256MiB original gates PASS |

Actual totals are **18 requests, 16 returns (11 E+F +5 E-only), two refused E+F
requests and 14 NOT_RUN slots**. No request was replaced or retried. The two
failures stopped their dependent workflows; independent models continued under
the frozen finite plan. The platform Job naturally FAILED. All four scientific
process subtrees were reaped with no remaining owned descendants.

The ten full-RI-JK returns cover original/displaced geometries, each with
Standard/2048MiB repeats0/1, Fast/2048MiB repeats0/1 and Fast/256MiB repeat0.
Warm repeats are dependent workflow observations, not independent timing samples.
`internal-policy-comparison.json` separately records same-geometry policy parity
and raw dependent timing groups; neither substitutes for independent acceptance.
The two water fixtures have different basis identities and cannot be exchanged.
Archived `hf_reference=exact` still uses DF correlation; conventional/exact does
not. Matched independent DF/RI-JK H2 references remain UNAVAILABLE.

## Independent numerical replay

All 16 returned samples pass their applicable original gates. Conventional
energy/force allowances remain 1e-9 Eh/2e-6 Eh/Bohr. Archive Standard keeps strict
`<1e-8 Eh`, `<3e-6 Eh/Bohr` max/RMS force. Fast keeps its original strict
1e-5/1e-4/3e-5 composite allowances and coefficient-L1 component allowances.

| Returned independent scope | Maximum E error (Eh) | Maximum force error (Eh/Bohr) | Maximum force RMS error |
| --- | ---: | ---: | ---: |
| Conventional water center | 3.268496584496461e-13 | 1.7446770932127986e-7 | 9.866686619367448e-8 |
| Full RI-JK water, all10 E+F | 6.289866405495559e-9 | 2.9215061877021272e-8 | 1.6406127215679416e-8 |

Conventional water native energy parity is exactly zero. Native reconverged
direction errors at 3e-4/1e-4 Bohr are 4.512348189544824e-9 and
3.9120031519734866e-10 Eh/Bohr. Comparison with the separately retained PySCF
directions remains independent of this native self-consistency check.

The archived water reference retains all360 displaced constituent energies.
Full vectors are reconstructed from these originals, never from native forces.
`model_contract.py` independently derives the 2/3-cardinal, HF-exponent1.63,
MP2-exponent3 composite using 80-digit Decimal/exact rational weights. Returned
coefficients only undergo a separate four-ULP metadata identity check; they do
not generate the oracle or alter any E/F gate. Semantic role/method/basis mapping
accepts component reordering and preserves the mp2_low/correction_mp2 alias.

## Exact source and execution provenance

Measured source is frozen `976f756b391bc65edb79f76df1a8c3fe807518a7` plus retained
qualified patch `5a1a974bab09e0fe31e6186e6ec2129abf26961113aafcd3d9c81616eb2d7cad`.
Portable SM90 library SHA256 is
`fea001df0f1a7c8bd6cbf5188fa0decb818e349da7e64d38465adb0b5aad57b6`.
Profile is portable_cuda/portable match/tuned=false/hardware_only=false.
Delivery-base changes do not relabel this as a current-master installed build.

The actual Job is `h100-small-cbs-20261024-e7f7-v2`, creation string
`1792864826000`. Root validated H10080/SM90 on qb-prod-gpu029, PCI2A:00.0,
UUID SHA `e888c12306245d8878f94e88f23f104ab821cb8a723f1ce80139b04380c18cda`,
driver595.58.03 and cgroup20CPU/200GiB before its one release at18:02:39Z.
Marker SHA is `f24ae323d379f20d7c639075d70d9f73a0a9ab4bf089395895896a485e5d2c7d`.
Actual source/env/DSO and selected device identities match before/after.

Runtime identity `86d641d1581eea3f7d2fe9d4bdf1da2bf36ebff588f02bfbece5c7623243aca3`
binds 40662 regular files, 8 links and19 external dependencies. It retains the
original07b manifest plus one exact source-equivalent `_distutils_hack` pyc,
not an ignored cache directory. The first Job/v1 refused the late cache name-set
change before release: zero scientific calls. Source/PEP552/compile-marshal
inspection, original failure and all host RED history remain in immutable
packets. The possible old HOST-grandchild origin is supported by timestamps,
not asserted as unique file-writer attribution. Existing earlier three setup
measurements are not retrospectively invalidated.

## Exact refusal interpretation

H2 index0 requested32GiB; archived exact-HF/DF index12 requested Standard2GiB.
Both threw `CUDA RHF provider exceeds retained allocation allowance` without
returning E/F. The frozen `reference_export.cuh` guard compares a device free-
memory delta against `(use_cublas?96MiB:0)+(use_cusolver?96MiB:0)` around vendor
handle preparation. This is distinct from its later numeric-memory-budget guard.
Actual flags, free-memory values, delta, selected allowance and failing internal
basis group were not returned and remain UNAVAILABLE. Status enum7 alone proves
neither physical CUDA OOM nor exhaustion of the32GiB mathematical budget. No
budget/source/SDK fix or new attempt was made to obtain a passing result.

## Original recovery and bounded offline verification

`actual-terminal-originals-v2.json.xz` retains all255 actual members/1795518 raw
bytes, including all16 observable/enriched IEEE/index returns, refusals,
attempts, logs, source scripts, preflight/release and cleanup outcomes. SHA256:
`2e9a5aa83bd998bcf2b63703f590f97c93a67b87fc0e573c034fbb438510816f`.
Raw originals retain byte counts/hashes and Linux mode witnesses; recovery applies
modes on POSIX and labels Windows mode application unavailable. The independent
owner/root collections have identical bytes, hashes and modes for every common
member. JSON envelope sizes are separate from raw member sums.

From this directory, using new destinations:

```text
python verify_actual.py --output /new/originals --report /new/offline-report.json
python test_restore_actual.py
```

The first command verifies bounded XZ framing, safe no-clobber recovery, all
255 byte/mode identities, actual frozen caller hashes, both primary/enriched
IEEE/index representations, before/after closure and every original numerical
gate. Exit0 means **offline storage/outcome closure PASS** while its scientific
matrix explicitly remains FAIL_OR_INCOMPLETE. Optimization is refused. These
commands never load the native library, run an oracle/solver, fetch, submit,
restart or publish a marker. Historical operational scripts are frozen producer
evidence, not authorization to launch them.

The standard publication decision is `inconclusive`, numerical/endpoint stages
are failed/incomplete and performance/production are not-run. Full operator
precision, semantic/Fock work, measured peak and fallback coverage remain
INCOMPLETE where the frozen ABI does not expose them. Zero/null counters never
mean zero work or measured zero memory. H2 cannot prove nonzero(T); synthetic
water is not physical water CBS-limit accuracy; C2's old24 displaced energies
and independent full vector remain UNAVAILABLE. No cross-machine speedup,
all-core acceptance, issue closure or parent completion is claimed.
