# Same-binary VV10 reuse ablation

Selected numerical dataset: **PASS**, with mixed protocol and hardware.
The original four-owner protocol remains **FAILED / INCOMPLETE**. Publication,
independent PR approval and merge are separate gates. Refs #2677; no default,
global README ranking or parent-issue completion follows.

## Measured scope

WB97M-V, spherical def2-SVP, water48/96, implicit XC/VV10 grids and complete
moving-grid E+F response. Standard/Fast x Direct/DF x default/SCF-off/force-off/
both-off: one cold plus three calls from each arm's immutable cold density.
All Fast candidate/audit/fallback SCF and force work is retained beyond public
iteration/Fock counts.

Frozen source `4a783a27e104e9392a7227b7efca79abcfba60ae`, native source identity
`2b5116bad6474aeaed5a0caff9216217c2ca7e8812761978ec5e93fe94e6614a`, core SHA-256
`2e0efd6b481c603fba6eed18b09c2cf1525190c42566b6075d35aba3598ec1b0`.
The recorded build/source/DSO/AOT inventories are retained; qualification does
not transfer to a later default-branch build.

The original five complete cohorts supplied80 samples on H100 `qb-prod-gpu046`,
PCI `0000:2A:00.0`. After timeout and platform reclamation, the authorized
[independent-arm batch](protocol-addendum-single-owner.md) used the normally
restarted Notebook on H100 `qb-prod-gpu095`, PCI `0000:18:00.0`. It completed
48 requests/zero retries at17:53 CST2026-10-24, replacing the entire incomplete
Standard/DF96 cohort and supplying Fast96. The three old partial cold returns
remain failure history. Protocol/hardware timing groups cannot be pooled.

## Numerical and work evidence

| Observable | Original absolute gate | Worst selected error |
| --- | ---: | ---: |
| Energy per atom, Eh | 1e-5 | 3.0313102807e-6 |
| Maximum force, Eh/Bohr | 1e-4 | 2.1186440036e-5 |
| Per-sample RMS force, Eh/Bohr | 3e-5 | 1.0775033025e-5 |

All128 selected formal and64 pilot samples pass. Independent PySCF/GPU4PySCF
producer identities and exact H/O orbital/DF tables are retained. CPU/GPU
water-3 and grid stability at3/12/48/96 atoms pass quarter gates. Historical
arrays with unavailable producers are excluded as references.

[Timings](timings.md) include cold construction/preparation and synchronized
warm public E+F; [work](work.md) retains histories, consumers and operator scopes.
Opaque J/K counts, isolated compile time and measured complete-endpoint peak
memory remain UNAVAILABLE. Device total is not peak memory. First-use effects,
resource plans and shared SSD/cache overlap prevent cache-only causal attribution
or general performance promotion. Consumer sampling is not a reservation.

## Offline reproduction

Use Linux Python3.11.16/NumPy2.4.6; no GPU/native scientific calculation:

```sh
python benchmarks/results/vv10-reuse-ablation-20261023/reproduce_recovery.py --output .artifacts/vv10-reuse-offline-replay
python -m unittest discover -s benchmarks/results/vv10-reuse-ablation-20261023 -p 'test_*.py'
```

A fresh destination restores716 original files and134 whole-file aliases with
exact bytes/SHA-256/POSIX modes, retired Job receipts, source/runtime/producer
identities and claim/admission/return bindings. Frozen replay regenerates reports.
Checksum-bound UTF8 helper sources in `offline-source.json.xz` restore for inspection.
Original local/remote files remain retained. The reader limits XZ decoder memory
to96MiB and expansion to64MiB/member, rejecting truncated or extra streams.

Verify Git-LF blobs or a `core.autocrlf=false` archive. Windows3.13.14/NumPy2.3.2
passed raw/provenance checks but strict RMS replay differed in32 last bits, at
most5.083e-21. This negative host result remains; stored values/gates are unchanged.

## Retained failures

The original Standard/DF96 child hit10800s at02:46:58 CST2026-10-24. Its failed
claim and83 returns survive; only80 enter complete cohorts. No checkpoint held
missing owner/device state; post-failure source integrity does not complete it.

The authorized intrusive both-off diagnostic returned, then failed a list/tuple
harness comparison before saving full E/F and outer timing. Those remain
UNAVAILABLE; its journal, callback work, rc1 and failed claim survive. Intrusive
timing cannot replace endpoints or explain the original timeout.

Cross-node flock failure, replacement atomic-mkdir guards and authorized H200
cancellation remain recorded. Original four-owner admission and broader DFT
acceptance remain unproven. See the
[Agent Note](../../../../.agents/notes/proposed/2026-10-23-vv10-same-binary-ablation.md)
for durable rationale and recovery limits.
