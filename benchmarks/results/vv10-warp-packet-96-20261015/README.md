# Original-protocol 96-atom VV10 packet follow-up

## Result and boundary

The opt-in packet schedule reduces the observed complete cold energy-plus-force
interval from **1325.684 to 1277.316 s** (48.368 s, 3.65%). It does **not** resolve
the original 96-atom endpoint: both arms reach the unchanged **1800-second
whole-population deadline in `moved/0`**, after returning only cold plus five
warm calls, **6/12 calls per arm**. Neither changed-geometry endpoint nor any
moved-warm call completes. There is no full-population speedup or acceptance.

| Original-protocol phase | `parallel` | `parallel-packet` |
| --- | ---: | ---: |
| Cold complete E/F, seconds | 1325.684 | 1277.316 |
| Five warm calls, median seconds | 59.591 | 59.689 |
| Moved | interrupted at whole-point deadline | interrupted at whole-point deadline |
| Moved-warm | not reached | not reached |

Each cold result performs 36 SCF iterations, 36 Fock builds and four SCF pair
sweeps, with one SCF pair-cache hit. Each warm result performs one iteration,
one Fock build, zero SCF pair sweeps and one SCF pair-cache hit. This schedule
does not improve the approximately 60-second warm path. These are observable
SCF counts, not a measurement of force-component times or active scalar pair
counts. No bottleneck attribution to a particular unprofiled force kernel is
made. Work sums in `summary.json` cover returned records only; zeros for an
incomplete phase do not mean the interrupted call performed no work.

All 12 returned calls across the two arms pass the independent gates. Maximum
absolute energy error is 8.641e-12 Eh and maximum force-component error is
2.335e-11 Eh/Bohr, below the unchanged 1e-8/1e-7 gates. Both accuracy-only
geometry receipts were retained previously from independently reconverged
GPU4PySCF default atomic guesses, without native inputs. Their complete force
vectors, convergence residuals and provenance are retained here, checked against
the exact basis bytes and protocol before either arm starts. They supply no
reference timing sample; the moved oracle does not qualify a native result
that never returns.

## Reproduction and provenance

- Base: master `5f80ba629b5f20412215cdc690f8fa458ac3fdf8`, with the opt-in packet
  patch and qualifications in `../vv10-warp-packet-20261015/`.
- Both arms use the same source-matched Release library. The source identity is
  `acaa3866a5dbfe3822d4cb2b01c07e3aca2cacd253c6a8efb8ddcbed3d2884ea`; the
  library SHA-256 is
  `b8d9febe0e9da4ea9e3e7bb3c6597b1ecdf1b444ba72f6df0c8499d1bc8a61a5`.
- Exclusive n4 RTX 5090, Slurm 951, eight requested CPUs/task and eight numerical
  library threads, finite `01:05:00` allocation. The two independent native
  point processes each receive the original 1800-second deadline. The allocation
  itself does not expire; the driver exits nonzero after recording both timeouts.
- WB97M-V/RKS, spherical def2-SVP, 768 AOs, no density fitting, FP64, moving
  48/16/32 grid, 2,359,296 nominal points, and five warm replays per geometry.
  Defaults remain normal except the explicit pair selector. There is no reduced
  grid, repeat count, preliminary density, reference production work, or longer
  benchmark-point deadline.
- The retained `endpoint.sh` validates provenance/oracles, runs both arms even
  after a control timeout, and saves outcomes/cache/Slurm receipts. Invoke it
  through the retained environment wrapper using finite `srun`; do not execute
  GPU work directly. No compiler-cache counters change during this campaign;
  the existing stationary artifact cache is reused, never cleared. The single
  fixed-order off/on cold comparison is descriptive, not an interleaved,
  repeated estimate of a causal cold-start speedup.
- Two early pre-endpoint setup failures (helper import and unreduced basis
  identity) are retained separately. They do not contribute timing observations.

`evidence.json.xz` retains raw native journals, original finite-time
outcomes, both independent E/F receipts, oracle-origin hashes, drivers, compiler
cache statistics, full deployment inventory and Slurm receipts. The original
3/12-atom evidence remains unchanged. Its byte-identical `deployment.json` is
shared from that archive and authenticated by SHA-256 rather than duplicated;
the verifier reconstructs the same complete source/runtime inventory. Run
`python benchmarks/results/vv10-warp-packet-96-20261015/verify.py` to recompute
gates, medians, partial populations and semantic work from this archive.

Default selection remains unchanged. Revisit after profiling the warm and
changed-geometry paths and demonstrating complete original-limit populations;
the packet kernel alone is insufficient evidence for promotion at 96 atoms.
