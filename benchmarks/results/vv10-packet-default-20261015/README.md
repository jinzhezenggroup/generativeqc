# Guarded automatic VV10 packet qualification

This separate campaign qualifies default dispatch, not a change to pair algebra.
The user requested default promotion after reviewing the original-limit 96-atom
comparison. Automatic selection keeps the previous admission: molecular VV10,
sm_120, at least 65,536 nominal grid points, the numerical-domain guard and
disjoint output leases. Unset and empty selectors both choose the packet route
when admitted. `parallel` restores the previous bilateral schedule and `ordered`
retains an explicit ordered fallback. Small grids and other architectures retain
their previous automatic fallback.

## Provenance and validation

The Release build is based on master
`cb8781fb1bdcfb9166ef07cfed463c0482dea752`, using a verified reusable ccache
launcher. Native source identity is
`9f72f8400dd4c6a2fdc8967a91b855f30c3765db1c9a49c54e7c9bea1fcfcbb4`;
library SHA-256 is
`6d9c1fd0cb2face995e15ff9881615417c9bf8495880be29f8ec8296f2b1eb80`.
Full deployment source/runtime hashes are retained and verified before device
execution. n4 Slurm 955 uses an exclusive RTX 5090 allocation, eight CPUs/task,
the assigned device visibility and a finite 20-minute allocation.

- Host: 238 passed; 56 optional scheduled-device cases skip on the host.
- Scheduled generated admission/independent-oracle suite: 752 passed.
- Scheduled native runtime/lifecycle suite: 47 passed.
- Memcheck and synccheck: six automatic-selector boundary cases each, zero errors.
- Complete independently reconverged molecular E/F/finite-difference suite:
  six passed, covering RKS/UKS and multiple orbital bases.
- Original-protocol 3/12-atom populations: all 24 complete E/F calls pass with
  **no pair-selector environment override**. The moving 48/16/32 grid, five warm
  replays per geometry and 1800-second whole-point deadlines are unchanged.
  Existing matching independent reference populations supply accuracy only;
  these are not new reference timing comparisons.

The boundary cases spread a small physical pair domain across
65,535/65,536/65,537 nominal points, independently checking all six force-seed
channels including screened, signed and active positive-zero weights. This
keeps the test work bounded without reducing the production grid admission.
No permanent source-body fingerprints or new backend-observability ABI are
introduced.

Compiler dependency, default-promotion inventory and runtime ownership checks
pass. The source-only CUDA ownership report passes. Its optional generated-build
census encounters an existing overlapping registration for
`generated/generated_rccsdt_cuda.cu`; the exact failure log is retained, and no
unrelated ownership family is changed here. The relevant VV10 shard classifies
existing launch glue as runtime, not physical scientific-code retirement.

## Interpretation

This campaign verifies automatic execution and independent numerical acceptance;
its timings are not a new repeated/interleaved performance comparison. The
historical opt-in speedups, semantic work counts, complete consumer measurements
and original-limit 96-atom failures remain unchanged in
`../vv10-warp-packet-20261015/` and
`../vv10-warp-packet-96-20261015/`. In particular, both 96-atom arms timed out
in `moved/0` after 6/12 calls: default promotion does not resolve that endpoint.
Their `default_enabled=false` fields describe those exact historical sources.

`summary.json` reports current phase medians and observable SCF work;
`evidence.json.xz` retains exact raw endpoint journals/references, XML,
sanitizer summaries, source patch, compact build/cache and scheduler receipts.
The full compiler warning log stays in ignored `.artifacts/vv10-default/`;
its digest, successful command and Release/cache provenance remain in the archive.
Two setup-only deployment failures are separate from accepted observations.
The generated current CUDA ownership report is deliberately not committed.
Run `python benchmarks/results/vv10-packet-default-20261015/verify.py` to audit
the archive, independent gates, semantic work and complete phase populations.

To keep the unchanged aggregate evidence budget, an earlier redundant README
report snapshot moves to existing Git-object recovery; its exact revision,
blob, size and checksum remain in the original README manifest. No current
scientific sample, failed endpoint or oracle is removed and nothing is uploaded
to an external archive.
