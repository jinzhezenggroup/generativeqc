# Bounded LR value-source fixture on H100

The standalone fixture passes strict FP64 numerical, coverage, work, resource and
sanitizer gates for Cartesian SPD total orders 3–5. Its measured source is
`1fa3ba2f9d579e868103bfd5877c83cc5d225f5d`, tree
`4cb25a54dd8cad13bb03947103e30a7bb84a7144`, over base `6987eea5259563250d1e81684ac998b727b8a513`.
This is a fixture-only capability. Native integration, actual molecular source
replay, WB96 endpoint performance, production default promotion and the all30
goal remain unqualified/open.

The compiler reuses the existing materialized primitive-pair LR recurrence with
a compile-time density-free canonical value sink. Its ordinary Fock wrapper and
production ownership remain established. The bounded one-task uint16 map uses
at most 32/96/192 slots at the incumbent 32-lane schedule; unsupported requests
and malformed maps refuse. No production lease, molecular task map, selector,
capacity increase, SCF/force change or alternative recurrence is added.

## Measured scope

One finite H100 job, `gqc-h100-wb-lr-fixture-1106-v1`, on physical
`GPU-24ec451f-eb55-820b-9107-7314fac7c0ba` (SM90, MIG disabled):

- All 3,120 ordinary cases and 6,240 graph replays pass: 9,360 reports.
- Each of memcheck, racecheck and synccheck covers the fixed pre-data 264-case
  slice, with 264 ordinary executions and 528 graph replays. This is 792
  reports/tool, not sanitizer coverage of all 3,120 cases.
- The existing CPU reducer rechecks all 126,792 retained ordinary source values
  against frozen independent Libcint inputs and all 558,960 raw-K entries per
  contraction view. Maximum candidate error is `1.0103029524088925e-14`;
  candidate/scalar difference is `4.440892098500626e-16`; candidate raw-K error
  is `2.060851489460447e-15`.
- Both original component bounds remain `2e-12 + 2e-10*abs(reference)` and
  `3e-11 + 3e-11*abs(reference)`. Guards, holes, one-write/zero publication and
  actual recurrence work pass. The 624 zero-live cases do zero work; 624
  nonempty LR(0) cases publish exact zeros.
- Candidate order3/4/5 registers are 148/202/248; reported local/stack bytes are
  512 each and spill stores/loads are zero. Attribute shared bytes are
  6,584/6,864/7,312; linked values are 7,608/7,888/8,336. Both views pass 16KiB.
  The 1,024-byte difference remains visible without a causal or occupancy claim.

[RESULTS.md](RESULTS.md) retains every metric and all nine resource rows.
[result-review.md](result-review.md) is the independent review of the completed
source-bound result. The older proposed Agent Note is retained byte-for-byte as
pre-execution rationale, including previous adverse evidence and the native
cache-ownership gap. Its earlier pending-device wording is superseded only for
this standalone scope by these completed records.

## Compact offline report audit

Run with standard Python, without NumPy, chemistry, native code or GPU:

```sh
python benchmarks/results/h100-wb-lr-values-phase-a-20261106/replay.py
```

The reader authenticates retained records and checks source/tree/input/tool,
object/executable bindings, complete reported populations and work, numerical
maxima/gates, resource views and strict terminal sanitizer summaries. It also
verifies the bounded archived qualifier/reducer source without executing it.
It does **not** reconstruct raw device arrays, rerun the original reducer or
independently repeat Libcint/CUDA qualification. The source-bound reduction is
retained exactly, with SHA256
`c6a0f59ce45815b21a53a3c7ac9513c99c778f758a011d01c0b9570a799b493f`.

`cohort.json` inventories the small records and the original raw-artifact root.
The 31 raw stream hashes, compiler commands, object/executable/tool identities,
full fixture/reference receipt and fixed sanitizer selection remain available
in these records. Large arrays, binaries, generated headers and full build
streams stay in the original owner artifacts. Hashes identify them; they do not
make those omitted raw arrays recoverable from this compact publication alone.
The generic nvcc pre-sm75 deprecation warning is retained in `link-warning.txt` (original `link.stderr` bytes); the
actual compile command pins compute90/sm90.

## Integration boundary

The publication preparation applies the five reviewed source files unchanged to
main `7109b3d208d23352f266e2bf7440fdfcc71df974`. That whole tree is not relabeled
as the measured source or a new H100 qualification. The shared-Rys
fixture and VV10 changes retain their separate measured scopes. These synthetic results establish no
WB96 source cost, endpoint speedup, default promotion or completed all30 win.
Any native source replay must first prove a compatible already-live primitive
cache owner and charge mapping/preparation before a separate integration decision.

Publication synchronization then merges main `13c11aa026558ce437e4a9509f4e6f488017ab57`
(VV10 SCF feature-sweep PR #3024) normally. That upstream change touches none
of the five measured files or this evidence directory. Their bytes remain
source-bound to `1fa3`; no new full-tree device qualification is inferred.

A later normal synchronization merges main `9fbb47085cb1f8b2abdb9fef6f729cdee95317f3`
(shared-Rys fixture PR #3023), preserving its source and evidence independently.
The external publication-branch host bridge shim repair is preserved. The five
measured WB source files remain exact `1fa3` bytes. The generated Libxc inventory
is refreshed for the combined current inputs while retaining its historical
measurements as stale; no new Libxc, H100 or endpoint qualification is inferred.

Selected JSON receipts are stored as lossless `.json.xz` files. `cohort.json`
retains each original logical filename, decoded length and SHA256, separately
from the stored path/length/hash. The bounded reader decodes them before the
unchanged scientific report audit; original failure bytes remain unchanged.
