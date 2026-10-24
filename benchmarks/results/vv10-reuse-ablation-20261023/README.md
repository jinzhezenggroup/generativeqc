# Same-binary VV10 reuse ablation

Selected numerical dataset: **PASS**, with explicit mixed protocol and hardware.
The original four-owner protocol remains **FAILED / INCOMPLETE**. Publication,
independent PR approval and merge are separate delivery gates. Refs #2677;
no production default, global README ranking or parent issue is completed.

## Measured scope

WB97M-V, spherical def2-SVP, water48/96, implicit XC/VV10 grids and complete
moving-grid E+F response. Standard/Fast x Direct/DF x four arms (default,
SCF-off, force-off, both-off), one cold plus three calls from each arm's own
immutable cold density. All Fast candidate/audit/fallback SCF and force work
is retained, including work beyond the public result's iteration/Fock count.

All selected calls use frozen source
`4a783a27e104e9392a7227b7efca79abcfba60ae`, native source identity
`2b5116bad6474aeaed5a0caff9216217c2ca7e8812761978ec5e93fe94e6614a`, and core SHA-256
`2e0efd6b481c603fba6eed18b09c2cf1525190c42566b6075d35aba3598ec1b0`.
The clean Git-LF inventory has 12,222 entries; 25 DSOs, 24 stationary AOT
manifests and the runtime/producer/build receipts are retained. Release,
CUDA 12.9, architecture90, portable_cuda, FastCompile off. This qualifies the
recorded build, not a later default-branch native build.

The original campaign contributed five complete cohorts (80 samples) on H100
`qb-prod-gpu046`, PCI `0000:2A:00.0`. After its timeout, the user approved
[independent arm processes](protocol-addendum-single-owner.md) for the remaining
three cohorts. The same Notebook object was normally restarted after platform
reclamation, on a different physical H100: `qb-prod-gpu095`, PCI `0000:18:00.0`.
One serial batch made exactly48 new public requests with zero retries, replacing
the entire incomplete Standard/DF cohort. It completed at17:53 CST on2026-10-24.
The three old partial cold calls are failure history and are never stitched in.

## Numerical and work evidence

| Observable | Original absolute gate | Worst selected error |
| --- | ---: | ---: |
| Energy per atom, Eh | 1e-5 | 3.0313102807e-6 |
| Maximum force, Eh/Bohr | 1e-4 | 2.1186440036e-5 |
| Per-sample RMS force, Eh/Bohr | 3e-5 | 1.0775033025e-5 |

All128 selected formal and64 pilot samples pass. Independent PySCF/GPU4PySCF
producer identities and exact H/O orbital/DF tables are retained. CPU/GPU
water-3 and reference-grid stability at3/12/48/96 atoms pass quarter gates.
Historical arrays with unavailable producers are not substituted as references.

[Timings](timings.md) and [work](work.md) keep the two protocol/hardware groups
separate. Cold includes construction/preparation; warm includes the complete
public E+F endpoint and required synchronization. Full SCF histories, force
consumers, operator scopes and raw vectors remain recoverable; the compressed
sample index is additional derived data. Unexposed J/K counts, isolated compile
time and measured complete-endpoint peak memory remain UNAVAILABLE. Device
total is not a measured peak. First-use effects and resource-plan differences
prevent cache-only causal attribution or general performance promotion.

## Offline reproduction

Use the recorded Linux Python3.11.16 / NumPy2.4.6 environment; no GPU or native
scientific calculation is needed:

```sh
python benchmarks/results/vv10-reuse-ablation-20261023/reproduce_recovery.py --output .artifacts/vv10-reuse-offline-replay
python -m unittest discover -s benchmarks/results/vv10-reuse-ablation-20261023 -p 'test_*storage.py'
```

The reader requires a fresh destination. It restores716 original files,
134 whole-file aliases and the retired Job receipts; checks original bytes,
SHA-256 and POSIX modes, claim/admission/return bindings, source/runtime and
producer identities; executes the frozen replay; then regenerates reports.
Metadata compression preserves exact original bytes. Complete UTF8 helper
sources are checksum-bound in `offline-source.json.xz` and materialized for
inspection in the replay output. The reader, codec and storage tests remain
directly readable. Original local/remote files are retained.

Check delivered Git-LF blobs or a `core.autocrlf=false` Git archive. Transformed
Windows working-copy bytes are not delivery evidence. Windows Python3.13.14 /
NumPy2.3.2 preserved raw-byte/provenance checks but strict scalar replay differed
only in32 force-RMS last bits (maximum5.083e-21); this negative host result is
retained. No stored value, frozen gate or tolerance was changed.

## Retained failures and limits

The original Standard/DF96 four-owner child reached its unchanged10800s limit
at02:46:58 CST on2026-10-24. Its failed claim and83 returned samples are kept;
only80 belong to complete selected cohorts. No checkpoint held the missing
owner/device state. Source integrity after failure does not complete that run.

One separately authorized both-off diagnostic returned under intrusive DF
tracing, then failed a harness list/tuple comparison before saving full E/F and
outer timing. Those observables remain UNAVAILABLE. Its journal, callback work,
rc1 and failed claim are retained, without borrowing values from other samples.
BSD flock failed cross-node exclusion; the retained atomic-mkdir controls passed
two-way/race/failure guards. The unstarted original H200 Job was normally
cancelled under its specific authorization. No fake Slurm allocation was used.

The new48 stay on one stable physical GPU and one frozen binary, with sampled
foreign-consumer checks. Sampling is not a physical reservation. Shared SSD/cache
overlap was disclosed, not causally isolated. The original four-owner admission,
cross-protocol speedup and broader DFT acceptance remain unproven.
